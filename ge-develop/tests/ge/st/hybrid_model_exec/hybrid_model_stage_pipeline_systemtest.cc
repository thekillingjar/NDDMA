/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include <gtest/gtest.h>
#include "runtime/rt.h"
#include "framework/executor/ge_executor.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "utils/data_buffers.h"
#include "register/op_tiling_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "graph/manager/mem_manager.h"
#include "common/dump/dump_manager.h"
#include "common/share_graph.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "hybrid/executor/hybrid_model_pipeline_executor.h"
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory_impl.h"
#include "common/share_graph.h"
#include "op_impl/less_important_op_impl.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/ge_model_builder.h"
#include "faker/magic_ops.h"
#include "graph/operator_reg.h"
#include "graph/ge_attr_value.h"
#include "ge_local_context.h"
#include "core/utils/tensor_utils.h"

using namespace ge;
using namespace gert;
using namespace std;
using namespace optiling::utils;
using namespace ge::hybrid;

namespace {
std::map<int, int> global_counters;
std::map<std::string, std::vector<int32_t>> global_vs;
std::vector<gert::Tensor> InputData2GertTensors(const InputData &input_data) {
  std::vector<gert::Tensor> input_tensors;
  for (size_t i = 0; i < input_data.blobs.size(); ++i) {
    gert::Tensor tensor;
    tensor.MutableTensorData().SetAddr(input_data.blobs[i].data, nullptr);
    tensor.MutableTensorData().SetSize(input_data.blobs[i].length);
    tensor.MutableStorageShape().SetDimNum(input_data.shapes[i].size());
    for (size_t j = 0U; j < input_data.shapes[i].size(); ++j) {
      tensor.MutableStorageShape().SetDim(0, input_data.shapes[i][j]);
    }
    input_tensors.emplace_back(std::move(tensor));
  }
  return input_tensors;
}
}  // namespace

int FetchAddGlobal(int key) {
  return global_counters[key]++;
}

int CreateGlobalCounter(int key) {
  global_counters[key] = 0;
  return key;
}

void ClearGlobalCounters() {
  global_counters.clear();
}

int AppendGlobalValues(const char *node, int v) {
  static std::mutex mu;
  std::unique_lock<std::mutex> lk(mu);
  global_vs[node].emplace_back(v);
  return v;
}

std::map<std::string, std::vector<int32_t>> FetchClearGlobalValues() {
  auto v = global_vs;
  global_vs.clear();
  return v;
}

class Listener : public ModelListener {
  using Done = std::function<void(uint32_t, uint32_t, uint32_t, std::vector<gert::Tensor> &)>;

 public:
  explicit Listener(Done done) : done_(done) {}
  Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result_code,
                       std::vector<gert::Tensor> &outputs) override {
    done_(model_id, data_index, result_code, outputs);
    return SUCCESS;
  }
  Done done_;
};

namespace {
class SimpleAllocator {
 public:
  static gert::TensorData Alloc(size_t size, const std::string &name) {
    std::unique_lock<std::mutex> lk(mu_);
    size_t aligned_size = (size + 32) / 32 * 32;
    void *block = malloc(aligned_size);
    named_blocks[block].first = 1U;
    named_blocks[block].second = name;
    GELOGI("Alloc %zu real %zu for %s %p", aligned_size, size, name.c_str(), block);
    return gert::TensorData(block, SimpleAllocator::AddrManager, aligned_size, gert::TensorPlacement::kOnDeviceHbm);
  }

 private:
  static ge::graphStatus AddrManager(void *block, TensorOperateType operate_type, void **out) {
    if (block == nullptr) {
      GELOGE(ge::FAILED, "SimpleAllocator failed operator %d", operate_type);
      return ge::GRAPH_FAILED;
    }
    if (operate_type == kGetTensorAddress) {
      GELOGI("SimpleAllocator get address %p", block);
      *out = block;
      return ge::GRAPH_SUCCESS;
    }
    if (operate_type == kFreeTensor) {
      std::unique_lock<std::mutex> lk(mu_);
      GELOGI("Unref %s %p, ref count %zu", named_blocks[block].second.c_str(), block, named_blocks[block].first);
      if (--named_blocks[block].first == 0U) {
        GELOGI("Free %s %p", named_blocks[block].second.c_str(), block);
        free(block);
      }
      return ge::GRAPH_SUCCESS;
    }
    if (operate_type == kPlusShareCount) {
      std::unique_lock<std::mutex> lk(mu_);
      GELOGI("Ref %s %p, ref count %zu", named_blocks[block].second.c_str(), block, named_blocks[block].first);
      ++named_blocks[block].first;
      return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_FAILED;
  }
  static std::mutex mu_;
  static std::map<void *, std::pair<size_t, std::string>> named_blocks;
};
std::mutex SimpleAllocator::mu_;
std::map<void *, std::pair<size_t, std::string>> SimpleAllocator::named_blocks;

memory::CachingMemAllocator caching_mem_allocator_{0};
thread_local memory::SingleStreamL2Allocator single_stream_l2_allocator{&caching_mem_allocator_};
ge::graphStatus KernelFakeGetNext(KernelContext *ctx) {
  auto ext_ctx = reinterpret_cast<ExtendedKernelContext *>(ctx);
  auto input = ctx->GetInputPointer<gert::GertTensorData>(kFkiStart + 1);
  auto tensor_data = ctx->GetOutputPointer<gert::GertTensorData>(1);
  auto tensor_shape = ctx->GetOutputPointer<gert::StorageShape>(0);
  gert::TensorUtils::ShareTdToGtd(
      SimpleAllocator::Alloc(4, std::string(ext_ctx->GetComputeNodeInfo()->GetNodeName()) + "/FakeGetNext:0"),
      single_stream_l2_allocator, *tensor_data);
  auto counter_key = *static_cast<int32_t *>(input->GetAddr());
  *static_cast<int32_t *>(tensor_data->GetAddr()) =
      *static_cast<int32_t *>(input->GetAddr()) + FetchAddGlobal(counter_key);
  tensor_data->SetPlacement(input->GetPlacement());
  tensor_data->SetSize(tensor_data->GetSize());
  tensor_shape->MutableStorageShape().SetDimNum(1);
  tensor_shape->MutableStorageShape().SetDim(0, 1);
  tensor_shape->MutableOriginShape().SetDimNum(1);
  tensor_shape->MutableOriginShape().SetDim(0, 1);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus KernelFakeCollectAllInputs(KernelContext *ctx) {
  auto ext_ctx = reinterpret_cast<ExtendedKernelContext *>(ctx);
  for (size_t i = 0U; i < ext_ctx->GetComputeNodeInputNum(); i++) {
    auto input = ctx->GetInputPointer<gert::GertTensorData>(kFkiStart + i + ext_ctx->GetComputeNodeInputNum());
    auto input_shape = ctx->GetInputPointer<gert::StorageShape>(kFkiStart + i);
    auto tensor_data = ctx->GetOutputPointer<gert::GertTensorData>(i + ext_ctx->GetComputeNodeInputNum());
    auto tensor_shape = ctx->GetOutputPointer<gert::StorageShape>(i);
    tensor_shape->MutableStorageShape() = input_shape->GetStorageShape();
    tensor_shape->MutableOriginShape() = input_shape->GetOriginShape();
    gert::TensorUtils::ShareTdToGtd(
        SimpleAllocator::Alloc(input->GetSize(), std::string(ext_ctx->GetComputeNodeInfo()->GetNodeName()) +
                                                     "/FakeCollectAllInputs:" + std::to_string(i)),
        single_stream_l2_allocator, *tensor_data);
    *static_cast<int32_t *>(tensor_data->GetAddr()) =
        AppendGlobalValues((std::string(ext_ctx->GetComputeNodeInfo()->GetNodeName()) + std::to_string(i)).c_str(),
                           *static_cast<int32_t *>(input->GetAddr()));
    tensor_data->SetPlacement(input->GetPlacement());
    tensor_data->SetSize(input->GetSize());
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

class StestHybridRt2PipelineExecutor : public testing::Test {
 protected:
  void SetUp() override {
    setenv("ENABLE_RUNTIME_V2", "1", 0);

    stub_ = std::unique_ptr<GertRuntimeStub>(new GertRuntimeStub());

    MagicOpFakerBuilder("FakeGetNext").KernelFunc(KernelFakeGetNext).Build(*stub_);
    MagicOpFakerBuilder("FakeCollectAllInputs").KernelFunc(KernelFakeCollectAllInputs).Build(*stub_);
  }
  void TearDown() override {
    unsetenv("ENABLE_RUNTIME_V2");
    ClearGlobalCounters();
    FetchClearGlobalValues();
    hybrid::NpuMemoryAllocator::Finalize();
    ge::MemManager::Instance().Finalize();
    stub_.reset(nullptr);
  }
  std::unique_ptr<GertRuntimeStub> stub_;
};

/**
 * 用例描述：测试两个Stage串联执行功能，验证输出Tensor的值符合预期
 * ┌──────────────┐   ┌──────────────────────┐
 * │  FakeGetNext │   │ FakeCollectAllInputs │
 * └──────┬───────┘   └───────────┬──────────┘
 *  ┌─────┴─────┐         ┌───────┴───────┐
 *  │  Stage_1  ├─────────►    Stage_2    │
 *  └───────────┘         └───────────────┘
 * 预置条件：
 * 1.用于模拟GetNext的FakeGetNext算子，该算子接收一个输入base，并输出base+0,base+1...的数据，即第N次调用，输出base+(N-1)
 * 2.用与模拟网络计算的FakeCollectAllInputs算子，该算子将每次执行的输入append到一个全局队列中，并输出输入的值，用于观测多Step时每个Step的输入
 * 3.用于存储Stage输入的全局变量global_vs（string:vector<int>），记录Stage名称与输入序列的映射关系
 * 测试步骤
 * 1.构造一张包含2个Stage（PartitionedCall）的计算图，Stage 1的子图包含一个FakeGetNext算子，Stage2的子图包含一个FakeCollectAllInputs算子
 * 2.在用例范围内，对FakeGetNext和FakeCollectAllInputs进行Kernel注册，Kernel计算逻辑与预置条件中的描述一致
 * 3.为计算图创建一个RT2 PipeLine执行器
 * 4.构造网络输入，即FakeGetNext的base输入，用例设置base=2，因此FakeGetNext的输出依次为2，3，4，...，经过Stage2的FakeCollectAllInputs计算后，
 * 输出仍然为2，3，4，... 5.执行前3个Step，检测其结果依次为2，3，4
 *
 * 预期结果
 * 1.执行成功
 * 2.每个Step的输出结果与预期值一致
 */
TEST_F(StestHybridRt2PipelineExecutor, test_2stage_no_pipe) {
  auto graph = ShareGraph::Build2StageGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  HybridModel hybrid_model(ge_root_model);

  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);
  ASSERT_EQ(executor.Init(), SUCCESS);

  const int32_t kGeneratorBase = CreateGlobalCounter(2);
  int32_t feed = kGeneratorBase;
  InputData inputs;
  inputs.blobs.emplace_back(ge::DataBuffer(&feed, sizeof(feed)));
  inputs.shapes.emplace_back(std::vector<int64_t>{1});
  std::shared_ptr<RunArgs> args;

  bool reached = false;
  int32_t result = 0;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        if (!outputs.empty()) {
          result = *static_cast<int32_t *>(outputs[0].GetAddr());
        }
        reached = true;
      });

  executor.Start(listener);

  for (int32_t i = 0U; i < 3; i++) {
    args = std::make_shared<RunArgs>();
    args->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(args);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }

    ASSERT_TRUE(reached);
    EXPECT_EQ(result, kGeneratorBase + i);
    reached = false;
  }
  executor.Stop();
}

/**
 * 用例描述：测试两个串联Stage流水100次执行，验证输出Tensor的值符合预期 & Stage每个Step的输出符合预期
 * ┌──────────────┐   ┌──────────────────────┐
 * │  FakeGetNext │   │ FakeCollectAllInputs │
 * └──────┬───────┘   └───────────┬──────────┘
 *  ┌─────┴─────┐         ┌───────┴───────┐
 *  │  Stage_1  ├─────────►    Stage_2    │
 *  └───────────┘         └───────────────┘
 * 预置条件：
 * 1.用于模拟GetNext的FakeGetNext算子，该算子接收一个输入base，并输出base+0,base+1...的数据，即第N次调用，输出base+(N-1)
 * 2.用与模拟网络计算的FakeCollectAllInputs算子，该算子将每次执行的输入append到一个全局队列中，并输出输入的值，用于观测多Step时每个Step的输入
 * 3.用于存储Stage输入的全局变量global_vs（string:vector<int>），记录Stage名称与输入序列的映射关系
 *
 * 测试步骤
 * 1.构造一张包含2个Stage（PartitionedCall）的计算图，Stage1
 * 的子图包含一个FakeGetNext算子，Stage2的子图包含一个FakeCollectAllInputs算子
 * 2.在用例范围内，对FakeGetNext和FakeCollectAllInputs进行Kernel注册，Kernel计算逻辑与预置条件中的描述一致
 * 3.为计算图创建一个RT2 PipeLine执行器
 * 4.构造网络输入，即FakeGetNext的base输入，用例设置base=2，因此FakeGetNext的输出依次为2，3，4，...
 * 5.设置流水次数的小循环option的值为100，每次执行100步流水，
 *      - 第一次Execute时，预期模型的输出为101，即第100次FakeGetNext的输出值99 + 2 = 101
 *      - 第二次Execute时，预期模型的输出为201，即第200次FakeGetNext的输出值199 + 2 = 201
 *      - 第三次Execute时，预期模型的输出为301，即第300次FakeGetNext的输出值299 + 2 = 301
 * 6.校验全局Vector中的元素数量为300（因为总计执行了300次）
 * 7.校验全局vector中的数据依次为2，3，4，...,301（表示Stage2整个执行周期中，处理的数据序列）
 *
 * 预期结果
 * 1.执行成功
 * 2.模型输出与预期值一致
 * 3.每个Step的输出结果与预期值一致
 */
TEST_F(StestHybridRt2PipelineExecutor, test_2stage_pipeline_100_steps) {
  auto graph = ShareGraph::Build2StageGraph();
  graph->SetNeedIteration(true);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  HybridModel hybrid_model(ge_root_model);

  const size_t kPipeLoop = 100;
  GetThreadLocalContext().SetGraphOption({{ge::ATTR_NAME_ITERATORS_PER_LOOP, std::to_string(kPipeLoop)}});
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);
  ASSERT_EQ(executor.Init(), SUCCESS);
  std::string iterations_loop;
  ge::GetThreadLocalContext().GetOption(ge::ATTR_NAME_ITERATORS_PER_LOOP, iterations_loop);
  EXPECT_EQ(iterations_loop, std::to_string(kPipeLoop));

  const int32_t kGeneratorBase = CreateGlobalCounter(2);
  int32_t feed = kGeneratorBase;
  InputData inputs;
  inputs.blobs.emplace_back(ge::DataBuffer(&feed, sizeof(feed)));
  inputs.shapes.emplace_back(std::vector<int64_t>{1});
  OutputData outputs;
  std::shared_ptr<RunArgs> args;

  bool reached = false;
  int32_t result = 0;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        if (!outputs.empty()) {
          result = *static_cast<int32_t *>(outputs[0].GetAddr());
        }
        reached = true;
      });

  executor.Start(listener);

  for (int32_t i = 0U; i < 3; i++) {
    args = std::make_shared<RunArgs>();
    args->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(args);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }

    ASSERT_TRUE(reached);
    EXPECT_EQ(result, kGeneratorBase + ((i + 1) * kPipeLoop - 1));
    reached = false;
  }

  auto stage2_input_values = FetchClearGlobalValues();
  ASSERT_EQ(stage2_input_values.size(), 1);
  EXPECT_EQ(stage2_input_values.begin()->second.size(), 3 * kPipeLoop);  // Stage loop has run 3 times
  for (size_t i = 0U; i < stage2_input_values.begin()->second.size(); i++) {
    EXPECT_EQ(stage2_input_values.begin()->second[i], kGeneratorBase + static_cast<int>(i));
  }

  GetThreadLocalContext().SetGraphOption({{ge::ATTR_NAME_ITERATORS_PER_LOOP, ""}});
  executor.Stop();
}

/**
 * 用例描述：测试Stage1对2场景流水100次执行，验证输出Tensor的值符合预期 & Stage每个Step的输出符合预期
*                      ┌──────────────────────┐
*                      │ FakeCollectAllInputs │
* ┌──────────────┐     └───────────┬──────────┘
* │  FakeGetNext │                 │
* └──────┬───────┘         ┌───────┴───────┐
*        │           ┌─────►    Stage_2    │
*  ┌─────┴─────┐     │     └───────────────┘
*  │  Stage_1  ├─────┤     ┌───────────────┐
*  └───────────┘     └─────►    Stage_3    │
*                          └───────┬───────┘
*                       ┌──────────┴───────────┐
*                       │ FakeCollectAllInputs │
*                       └──────────────────────┘
 * 预置条件：
 * 1.用于模拟GetNext的FakeGetNext算子，该算子接收一个输入base，并输出base+0,base+1...的数据，即第N次调用，输出base+(N-1)
 * 2.用与模拟网络计算的FakeCollectAllInputs算子，该算子将每次执行的输入append到一个全局队列中，并输出输入的值，用于观测多Step时每个Step的输入
 * 3.用于存储Stage输入的全局变量global_vs（string:vector<int>），记录Stage名称与输入序列的映射关系
 *
 * 测试步骤
 * 1.构造一张包含3个Stage（PartitionedCall）的计算图，Stage1的子图包含一个FakeGetNext算子，
 *   Stage2和Stage3的子图包含一个FakeCollectAllInputs算子，Stage1的输出同时给到Stage2和Stage3，该模型有两个输出。
 * 2.在用例范围内，对FakeGetNext和FakeCollectAllInputs进行Kernel注册，Kernel计算逻辑与预置条件中的描述一致
 * 3.为计算图创建一个RT2 PipeLine执行器
 * 4.构造网络输入，即FakeGetNext的base输入，用例设置base=2，因此FakeGetNext的输出依次为2，3，4，...
 * 5.设置流水次数的小循环option的值为100，每次执行100步流水，
 *      - 第一次Execute时，预期模型的输出分别为101，101，即第100次FakeGetNext的输出值99 + 2 = 101
 *      - 第二次Execute时，预期模型的输出为201，201，即第200次FakeGetNext的输出值199 + 2 = 201
 *      - 第三次Execute时，预期模型的输出为301，301，即第300次FakeGetNext的输出值299 + 2 = 301
 * 6.校验全局GlobalValues中，Stage2和Stage3的元素数量均为300（因为总计执行了300次）
 * 7.校验全局GlobalValues中，Stage2和Stage3的元素数据依次为2，3，4，...,301（表示整个执行周期中，每个处理的数据序列）
 *
 * 预期结果
 * 1.执行成功
 * 2.模型输出与预期值一致
 * 3.每个Step的输出结果与预期值一致
 */
TEST_F(StestHybridRt2PipelineExecutor, test_1to2_stage_pipeline_100_steps) {
  auto graph = ShareGraph::Build1to2StageGraph();
  graph->SetNeedIteration(true);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  HybridModel hybrid_model(ge_root_model);

  const size_t kPipeLoop = 100;
  GetThreadLocalContext().SetGraphOption({{ge::ATTR_NAME_ITERATORS_PER_LOOP, std::to_string(kPipeLoop)}});
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);
  ASSERT_EQ(executor.Init(), SUCCESS);
  std::string iterations_loop;
  ge::GetThreadLocalContext().GetOption(ge::ATTR_NAME_ITERATORS_PER_LOOP, iterations_loop);
  EXPECT_EQ(iterations_loop, std::to_string(kPipeLoop));

  const int32_t kGeneratorBase = CreateGlobalCounter(2);
  int32_t feed = kGeneratorBase;
  InputData inputs;
  inputs.blobs.emplace_back(ge::DataBuffer(&feed, sizeof(feed)));
  inputs.shapes.emplace_back(std::vector<int64_t>{1});
  OutputData outputs;
  std::shared_ptr<RunArgs> args;

  bool reached = false;
  int32_t result[2] = {0, 0};
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
          for (size_t i = 0U; i < outputs.size(); i++) {
            result[i] = *static_cast<int32_t *>(outputs[i].GetAddr());
          }
        reached = true;
      });

  executor.Start(listener);

  for (int32_t i = 0U; i < 3; i++) {
    args = std::make_shared<RunArgs>();
    args->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(args);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }

    ASSERT_TRUE(reached);
    EXPECT_EQ(result[0], kGeneratorBase + ((i + 1) * kPipeLoop - 1));
    EXPECT_EQ(result[1], kGeneratorBase + ((i + 1) * kPipeLoop - 1));
    reached = false;
  }

  auto stage_input_values = FetchClearGlobalValues();
  EXPECT_EQ(stage_input_values.size(), 2);
  for (auto &item : stage_input_values) {
    EXPECT_EQ(item.second.size(), 3 * kPipeLoop);  // Stage loop has run 3 times
    for (size_t i = 0U; i < item.second.size(); i++) {
      EXPECT_EQ(item.second[i], kGeneratorBase + static_cast<int>(i));
    }
  }

  GetThreadLocalContext().SetGraphOption({{ge::ATTR_NAME_ITERATORS_PER_LOOP, ""}});
  executor.Stop();
}

/**
 * 用例描述：测试Stage2对1场景流水100次执行，验证输出Tensor的值符合预期 & Stage每个Step的输出符合预期
 * ┌──────────────┐
 * │  FakeGetNext │       ┌──────────────────────┐
 * └──────┬───────┘       │ FakeCollectAllInputs │
 *  ┌─────┴─────┐         └───────────┬──────────┘
 *  │  Stage_1  ├───────┐     ┌───────┴───────┐
 *  └───────────┘       ├─────►    Stage_3    │
 *  ┌───────────┐       │     └───────────────┘
 *  │  Stage_2  ├───────┘
 *  └─────┬─────┘
 * ┌──────┴───────┐
 * │  FakeGetNext │
 * └──────────────┘
 * 预置条件：
 * 1.用于模拟GetNext的FakeGetNext算子，该算子接收一个输入base，并输出base+0,base+1...的数据，即第N次调用，输出base+(N-1)
 * 2.用与模拟网络计算的FakeCollectAllInputs算子，该算子将每次执行的输入append到一个全局队列中，并输出输入的值，用于观测多Step时每个Step的输入
 * 3.用于存储Stage输入的全局变量global_vs（string:vector<int>），记录Stage名称与输入序列的映射关系
 *
 * 测试步骤
 * 1.构造一张包含3个Stage（PartitionedCall）的计算图，Stage1和Stage2
 * 的子图包含一个FakeGetNext算子，Stage3的子图包含一个FakeCollectAllInputs算子，Stage1和Stage2分别为Stage3的输入0和输入1
 * 该模型有两个输出。
 * 2.在用例范围内，对FakeGetNext和FakeCollectAllInputs进行Kernel注册，Kernel计算逻辑与预置条件中的描述一致
 * 3.为计算图创建一个RT2 PipeLine执行器
 * 4.构造网络输入1，即Stage1的FakeGetNext的base1输入，用例设置base1=2，因此FakeGetNext的输出依次为2，3，4，...
 * 5.构造网络输入2，即Stage2的FakeGetNext的base2输入，用例设置base2=3，因此FakeGetNext的输出依次为3，4，5，...
 * 5.设置流水次数的小循环option的值为100，每次执行100步流水，
 *      - 第一次Execute时，预期模型的输出分别为101，102，即第100次FakeGetNext的输出值99 + 2|3 = 101|102
 *      - 第二次Execute时，预期模型的输出为201，202，即第200次FakeGetNext的输出值199 + 2|3 = 201|202
 *      - 第三次Execute时，预期模型的输出为301，302，即第300次FakeGetNext的输出值299 + 2|3 = 301|202
 * 6.校验全局GlobalValues中，Stage3的两个输入元素数量统计均为300（因为总计执行了300次）
 * 7.校验全局GlobalValues中，Stage3的输入1数据依次为2，3，4，...,301，输入2数据依次为3，4，5，...,302
 *
 * 预期结果
 * 1.执行成功
 * 2.模型输出与预期值一致
 * 3.每个Step的输出结果与预期值一致
 */
TEST_F(StestHybridRt2PipelineExecutor, test_2to1_stage_pipeline_100_steps) {
  auto graph = ShareGraph::Build2to1StageGraph();
  graph->SetNeedIteration(true);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  HybridModel hybrid_model(ge_root_model);

  const size_t kPipeLoop = 100;
  GetThreadLocalContext().SetGraphOption({{ge::ATTR_NAME_ITERATORS_PER_LOOP, std::to_string(kPipeLoop)}});
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);
  ASSERT_EQ(executor.Init(), SUCCESS);
  std::string iterations_loop;
  ge::GetThreadLocalContext().GetOption(ge::ATTR_NAME_ITERATORS_PER_LOOP, iterations_loop);
  EXPECT_EQ(iterations_loop, std::to_string(kPipeLoop));

  const int32_t kGenerator1Base = CreateGlobalCounter(2);
  const int32_t kGenerator2Base = CreateGlobalCounter(3);
  CreateGlobalCounter(kGenerator2Base);
  int32_t feed1 = kGenerator1Base;
  int32_t feed2 = kGenerator2Base;
  InputData inputs;
  inputs.blobs.emplace_back(ge::DataBuffer(&feed1, sizeof(feed1)));
  inputs.shapes.emplace_back(std::vector<int64_t>{1});
  inputs.blobs.emplace_back(ge::DataBuffer(&feed2, sizeof(feed2)));
  inputs.shapes.emplace_back(std::vector<int64_t>{1});
  OutputData outputs;
  std::shared_ptr<RunArgs> args;

  bool reached = false;
  int32_t result[2] = {1, 1};
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        for (size_t i = 0U; i < outputs.size(); i++) {
          result[i] = *static_cast<int32_t *>(outputs[i].GetAddr());
        }
        reached = true;
      });

  executor.Start(listener);

  for (int32_t i = 0U; i < 3; i++) {
    args = std::make_shared<RunArgs>();
    args->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(args);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }

    ASSERT_TRUE(reached);
    EXPECT_EQ(result[0], kGenerator1Base + ((i + 1) * kPipeLoop - 1));
    EXPECT_EQ(result[1], kGenerator2Base + ((i + 1) * kPipeLoop - 1));
    reached = false;
  }

  auto stage2_input_values = FetchClearGlobalValues();
  EXPECT_EQ(stage2_input_values.size(), 2);
  auto &stage3_input1_seq = stage2_input_values.begin()->second;
  auto &stage3_input2_seq = stage2_input_values.rbegin()->second;
  EXPECT_EQ(stage3_input1_seq.size(), 3 * kPipeLoop);  // Stage loop has run 3 times
  EXPECT_EQ(stage3_input2_seq.size(), 3 * kPipeLoop);  // Stage loop has run 3 times
  for (size_t i = 0U; i < stage3_input1_seq.size(); i++) {
    EXPECT_EQ(stage3_input1_seq[i], kGenerator1Base + static_cast<int>(i));
    EXPECT_EQ(stage3_input2_seq[i], kGenerator2Base + static_cast<int>(i));
  }

  GetThreadLocalContext().SetGraphOption({{ge::ATTR_NAME_ITERATORS_PER_LOOP, ""}});
  executor.Stop();
}

/**
 * 用例描述：测试两个串联Stage流水100次执行，同时存在Stage0的输出  同时给到多个Stage1的输入，验证输出Tensor的值符合预期 & Stage每个Step的输出符合预期
 * ┌──────────────┐     ┌──────────────────────┐
 * │  FakeGetNext │     │ FakeCollectAllInputs │
 * └──────┬───────┘     └───────────┬──────────┘
 *  ┌─────┴─────┐   ┌──────►┌───────┴───────┐
 *  │  Stage_1  ├───├──────►    Stage_2     │
 *  └───────────┘   └──────►└───────────────┘
 * 预置条件：
 * 1.用于模拟GetNext的FakeGetNext算子，该算子接收一个输入base，并输出base+0,base+1...的数据，即第N次调用，输出base+(N-1)
 * 2.用与模拟网络计算的FakeCollectAllInputs算子，该算子将每次执行的输入append到一个全局队列中，并输出输入的值，用于观测多Step时每个Step的输入
 * 3.用于存储Stage输入的全局变量global_vs（string:vector<int>），记录Stage名称与输入序列的映射关系
 *
 * 测试步骤
 * 1.构造一张包含2个Stage（PartitionedCall）的计算图，Stage1
 * 的子图包含一个FakeGetNext算子，Stage2的子图包含一个FakeCollectAllInputs算子
 * 2.在用例范围内，对FakeGetNext和FakeCollectAllInputs进行Kernel注册，Kernel计算逻辑与预置条件中的描述一致
 * 3.为计算图创建一个RT2 PipeLine执行器
 * 4.构造网络输入，即FakeGetNext的base输入，用例设置base=2，因此FakeGetNext的输出依次为2，3，4，...
 * 5.设置流水次数的小循环option的值为100，每次执行100步流水，
 *      - 第一次Execute时，预期模型的输出为101，101，101即第100次FakeGetNext的输出值99 + 2 = 101
 *      - 第二次Execute时，预期模型的输出为201，201，201即第200次FakeGetNext的输出值199 + 2 = 201
 *      - 第三次Execute时，预期模型的输出为301，301，301即第300次FakeGetNext的输出值299 + 2 = 301
 * 6.校验全局Vector中的元素数量为300（因为总计执行了300次）
 * 7.校验全局vector中的数据依次为2，3，4，...,301（表示Stage2整个执行周期中，处理的数据序列）
 *
 * 预期结果
 * 1.执行成功
 * 2.模型输出与预期值一致
 * 3.每个Step的输出结果与预期值一致
 */
TEST_F(StestHybridRt2PipelineExecutor, test_2stage_input_1toN_pipeline_100_steps) {
  auto graph = ShareGraph::Build2StageWith1ToNGraph();
  graph->SetNeedIteration(true);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  HybridModel hybrid_model(ge_root_model);

  const size_t kPipeLoop = 100;
  GetThreadLocalContext().SetGraphOption({{ge::ATTR_NAME_ITERATORS_PER_LOOP, std::to_string(kPipeLoop)}});
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  HybridModelAsyncExecutor executor(&hybrid_model);
  ASSERT_EQ(executor.Init(), SUCCESS);
  std::string iterations_loop;
  ge::GetThreadLocalContext().GetOption(ge::ATTR_NAME_ITERATORS_PER_LOOP, iterations_loop);
  EXPECT_EQ(iterations_loop, std::to_string(kPipeLoop));

  const int32_t kGeneratorBase = CreateGlobalCounter(2);
  int32_t feed = kGeneratorBase;
  InputData inputs;
  inputs.blobs.emplace_back(ge::DataBuffer(&feed, sizeof(feed)));
  inputs.shapes.emplace_back(std::vector<int64_t>{1});
  OutputData outputs;
  std::shared_ptr<RunArgs> args;

  bool reached = false;
  std::vector<int32_t> result;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>(
      [&](uint32_t model_id, uint32_t data_index, uint32_t result_code, std::vector<gert::Tensor> &outputs) {
        result.clear();
        for (size_t i = 0U; i < outputs.size(); i++) {
          result.emplace_back(*static_cast<int32_t *>(outputs[i].GetAddr()));
        }
        reached = true;
      });

  executor.Start(listener);

  for (int32_t i = 0U; i < 3; i++) {
    args = std::make_shared<RunArgs>();
    args->input_tensor = std::move(InputData2GertTensors(inputs));
    executor.EnqueueData(args);

    size_t kMaxWaitSeconds = 5U;
    for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
      if (reached) {
        break;
      }
      sleep(1);
    }

    ASSERT_TRUE(reached);
    ASSERT_EQ(result.size(), 3U);
    for (auto &value : result) {
      EXPECT_EQ(value, kGeneratorBase + ((i + 1) * kPipeLoop - 1));
    }
    reached = false;
  }

  auto stage2_input_values = FetchClearGlobalValues();
  ASSERT_EQ(stage2_input_values.size(), 3);
  for (auto &stage2_input_value : stage2_input_values) {
    EXPECT_EQ(stage2_input_value.second.size(), 3 * kPipeLoop);  // Stage loop has run 3 times
    for (size_t i = 0U; i < stage2_input_value.second.size(); i++) {
      EXPECT_EQ(stage2_input_value.second[i], kGeneratorBase + static_cast<int>(i));
    }
  }

  GetThreadLocalContext().SetGraphOption({{ge::ATTR_NAME_ITERATORS_PER_LOOP, ""}});
  executor.Stop();
}
