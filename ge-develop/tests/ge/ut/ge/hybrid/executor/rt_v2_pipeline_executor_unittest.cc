/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "macro_utils/dt_public_scope.h"
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/executor/hybrid_model_rt_v2_executor.h"
#include "hybrid/executor/runtime_v2/rt_v2_pipeline_executor.h"
#include "hybrid/executor/runtime_v2/rt_v2_utils.h"
#include "graph/ge_context.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/types.h"
#include "common/share_graph.h"
#include "op_impl/less_important_op_impl.h"
#include "stub/gert_runtime_stub.h"
#include "faker/ge_model_builder.h"
#include "faker/magic_ops.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "core/utils/tensor_utils.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"

using namespace ge;
using namespace hybrid;
using namespace gert;

class SimpleAllocator {
 public:
  static gert::TensorData Alloc(size_t size, const std::string &name = "") {
    std::unique_lock<std::mutex> lk(mu_);
    size_t aligned_size = (size + 32) / 32 * 32;
    void *block = malloc(aligned_size);
    named_blocks_[block].first = 1U;
    named_blocks_[block].second = name;
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
      GELOGI("Unref %s %p, ref count %zu", named_blocks_[block].second.c_str(), block, named_blocks_[block].first);
      if (--named_blocks_[block].first == 0U) {
        GELOGI("Free %s %p", named_blocks_[block].second.c_str(), block);
        free(block);
      }
      return ge::GRAPH_SUCCESS;
    }
    if (operate_type == kPlusShareCount) {
      std::unique_lock<std::mutex> lk(mu_);
      GELOGI("Ref %s %p, ref count %zu", named_blocks_[block].second.c_str(), block, named_blocks_[block].first);
      ++named_blocks_[block].first;
      return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_FAILED;
  }
  static std::mutex mu_;
  static std::map<void *, std::pair<size_t, std::string>> named_blocks_;
};
std::mutex SimpleAllocator::mu_;
std::map<void *, std::pair<size_t, std::string>> SimpleAllocator::named_blocks_;


class UtestRtV2PipelineExecutor : public testing::Test {
 protected:
  void SetUp() {
    setenv("ENABLE_RUNTIME_V2", "1", 0);
    stub_ = std::unique_ptr<GertRuntimeStub>(new GertRuntimeStub());
    MagicOpFakerBuilder("FakeGetNext").KernelFunc([](KernelContext *) { return GRAPH_SUCCESS; }).Build(*stub_);
    MagicOpFakerBuilder("FakeCollectAllInputs").KernelFunc([](KernelContext *) { return GRAPH_SUCCESS; }).Build(*stub_);
  }
  void TearDown() {
    unsetenv("ENABLE_RUNTIME_V2");
    stub_.reset(nullptr);
  }

 private:
  std::unique_ptr<GertRuntimeStub> stub_;
};

class Listener : public ModelListener {
 public:
  Listener(std::function<void()> done) : done_(done) {}
  Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result_code,
                       std::vector<gert::Tensor> &outputs) override {
    done_();
    return SUCCESS;
  }
  std::function<void()> done_;
};

namespace {
std::vector<NodePtr> GetGraphOrderedDataNodes(ComputeGraphPtr graph) {
  std::map<size_t, NodePtr> index_2_datas;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() != ge::DATA) {
      continue;
    }
    size_t index = 0U;
    GE_ASSERT(ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_INDEX, index));
    index_2_datas[index] = node;
  }
  std::vector<NodePtr> ordered_datas;
  for (auto &index_2_data : index_2_datas) {
    ordered_datas.emplace_back(index_2_data.second);
  }
  return ordered_datas;
}

std::vector<NodePtr> GetGraphStageNodes(ComputeGraphPtr graph) {
  std::vector<NodePtr> stage_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == ge::PARTITIONEDCALL) {
      stage_nodes.emplace_back(node);
    }
  }
  return stage_nodes;
}

void ModelDescToTensors(const gert::ModelIoDesc *desc, size_t num, std::vector<gert::Tensor> &tensors) {
  tensors.resize(num);
  for (size_t i = 0U; i < num; i++) {
    auto &holder = tensors[i];
    holder.SetData(gert::TensorData(nullptr));
    holder.SetDataType(static_cast<ge::DataType>(desc[i].GetDataType()));
    holder.SetOriginFormat(desc[i].GetOriginFormat());
    holder.SetStorageFormat(desc[i].GetStorageFormat());
  }
}
}  // namespace

// 测试多种Stage节点组合情况下的Stage创建能力
TEST_F(UtestRtV2PipelineExecutor, test_create_pipeline_executor) {
  for (auto graph :
       {ShareGraph::Build2StageGraph(), ShareGraph::Build1to2StageGraph(), ShareGraph::Build2to1StageGraph()}) {
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.BuildGeRootModel();
    RtSession session;
    auto executor = RtV2PipelineExecutor::Create(ge_root_model, &session);
    ASSERT_NE(executor, nullptr);

    size_t num_inputs = 0U;
    const ModelIoDesc *inputs_desc = executor->GetAllInputsDesc(num_inputs);
    auto data_nodes = GetGraphOrderedDataNodes(graph);
    ASSERT_EQ(data_nodes.size(), num_inputs);
    for (size_t i = 0U; i < num_inputs; i++) {
      auto &compute_io_desc = data_nodes[i]->GetOpDesc()->GetOutputDesc(0);
      auto &model_io_desc = inputs_desc[i];
      ASSERT_EQ(compute_io_desc.GetDataType(), model_io_desc.GetDataType());
      ASSERT_EQ(compute_io_desc.GetFormat(), model_io_desc.GetStorageFormat());
      ASSERT_EQ(compute_io_desc.GetOriginFormat(), model_io_desc.GetOriginFormat());
      GeShape model_io_shape;
      RtShapeAsGeShape(model_io_desc.GetStorageShape(), model_io_shape);
      ASSERT_EQ(compute_io_desc.GetShape(), model_io_shape);
      GeShape model_io_origin_shape;
      RtShapeAsGeShape(model_io_desc.GetOriginShape(), model_io_origin_shape);
      ASSERT_EQ(compute_io_desc.GetOriginShape(), model_io_origin_shape);
    }

    size_t num_outputs = 0U;
    const ModelIoDesc *outputs_desc = executor->GetAllOutputsDesc(num_outputs);
    auto netoutput = graph->FindFirstNodeMatchType(NETOUTPUT);
    ASSERT_NE(netoutput, nullptr);
    auto graph_outputs_desc = netoutput->GetOpDesc()->GetAllInputsDesc();
    ASSERT_EQ(graph_outputs_desc.size(), num_outputs);
    for (size_t i = 0U; i < num_outputs; i++) {
      auto &compute_io_desc = netoutput->GetOpDesc()->GetInputDesc(i);
      auto &model_io_desc = outputs_desc[i];
      ASSERT_EQ(compute_io_desc.GetDataType(), model_io_desc.GetDataType());
      ASSERT_EQ(compute_io_desc.GetFormat(), model_io_desc.GetStorageFormat());
      ASSERT_EQ(compute_io_desc.GetOriginFormat(), model_io_desc.GetOriginFormat());
      GeShape model_io_shape;
      RtShapeAsGeShape(model_io_desc.GetStorageShape(), model_io_shape);
      ASSERT_EQ(compute_io_desc.GetShape(), model_io_shape);
      GeShape model_io_origin_shape;
      RtShapeAsGeShape(model_io_desc.GetOriginShape(), model_io_origin_shape);
      ASSERT_EQ(compute_io_desc.GetOriginShape(), model_io_origin_shape);
    }

    ASSERT_EQ(executor->stage_executors_.size(), GetGraphStageNodes(graph).size());
  }
}

// 测试多种Stage节点组合情况下的Stage加载后的卸载能力
TEST_F(UtestRtV2PipelineExecutor, test_load_unload_pipeline_executor) {
  for (auto graph :
       {ShareGraph::Build2StageGraph(), ShareGraph::Build1to2StageGraph(), ShareGraph::Build2to1StageGraph()}) {
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.BuildGeRootModel();
    RtSession session;
    auto executor = RtV2PipelineExecutor::Create(ge_root_model, &session);
    ASSERT_NE(executor, nullptr);

    ModelExecuteArg execute_args;
    ModelLoadArg load_args(nullptr);
    ASSERT_EQ(executor->Load(execute_args, load_args), SUCCESS);
    ASSERT_EQ(executor->stage_executors_.size(), GetGraphStageNodes(graph).size());
    for (size_t i = 0U; i < executor->stage_executors_.size(); i++) {
      ASSERT_EQ(executor->stage_executors_[i]->worker_.joinable(), (i != 0U));
    }
    ASSERT_EQ(executor->Unload(), SUCCESS);
  }
}

// 测试Pipeline executor只加载不显示卸载时，可以正常退出
TEST_F(UtestRtV2PipelineExecutor, test_pipeline_executor_exit_safely_without_unload) {
  auto graph = ShareGraph::Build2StageGraph();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  RtSession session;
  auto executor = RtV2PipelineExecutor::Create(ge_root_model, &session);
  ASSERT_NE(executor, nullptr);

  ModelExecuteArg execute_args;
  ModelLoadArg load_args(nullptr);
  ASSERT_EQ(executor->Load(execute_args, load_args), SUCCESS);
}

// 此条用例主要验证Stage发生任何异常时，线程能正常退出。
TEST_F(UtestRtV2PipelineExecutor, test_pipeline_executor_execute_exit_safely_when_stage_error) {
  auto graph = ShareGraph::Build2StageGraph();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  RtSession session;
  auto executor = RtV2PipelineExecutor::Create(ge_root_model, &session);
  ASSERT_NE(executor, nullptr);

  ModelExecuteArg execute_args;
  ModelLoadArg load_args(nullptr);
  ASSERT_EQ(executor->Load(execute_args, load_args), SUCCESS);

  std::vector<gert::Tensor> inputs_holder;
  std::vector<gert::Tensor *> inputs;
  size_t num_inputs = 0U;
  const ModelIoDesc *inputs_desc = executor->GetAllInputsDesc(num_inputs);
  ModelDescToTensors(inputs_desc, num_inputs, inputs_holder);
  for (auto &input_holder : inputs_holder) {
    inputs.emplace_back(&input_holder);
  }

  std::vector<gert::Tensor> outputs_holder;
  std::vector<gert::Tensor *> outputs;
  size_t num_outputs = 0U;
  const ModelIoDesc *outputs_desc = executor->GetAllOutputsDesc(num_outputs);
  ModelDescToTensors(outputs_desc, num_outputs, outputs_holder);
  for (auto &output_holder : outputs_holder) {
    outputs.emplace_back(&output_holder);
  }
  RtV2ExecutorInterface::RunConfig config(1U);
  ASSERT_NE(executor->Execute(execute_args, inputs.data(), inputs.size(), outputs.data(), outputs.size(), config), SUCCESS);
  ASSERT_EQ(executor->Unload(), SUCCESS);
}

namespace {
memory::CachingMemAllocator caching_mem_allocator_{0};
memory::SingleStreamL2Allocator single_stream_l2_allocator_{&caching_mem_allocator_};
ge::graphStatus OwnedMemoryKernel(KernelContext *ctx) {
  auto ext_ctx = reinterpret_cast<ExtendedKernelContext *>(ctx);
  for (size_t i = 0U; i < ext_ctx->GetComputeNodeInputNum(); i++) {
    auto tensor_data = ctx->GetOutputPointer<gert::GertTensorData>(i + ext_ctx->GetComputeNodeInputNum());
    gert::TensorUtils::ShareTdToGtd(SimpleAllocator::Alloc(64), single_stream_l2_allocator_, *tensor_data);
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

// 测试Pipeline的执行能力，执行精度的正确性，通过ST用例覆盖
TEST_F(UtestRtV2PipelineExecutor, test_pipeline_executor_execute_success) {
  MagicOpFakerBuilder("FakeGetNext").KernelFunc(OwnedMemoryKernel).Build(*stub_);
  MagicOpFakerBuilder("FakeCollectAllInputs").KernelFunc(OwnedMemoryKernel).Build(*stub_);

  auto graph = ShareGraph::Build2StageGraph();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  RtSession session;
  auto executor = RtV2PipelineExecutor::Create(ge_root_model, &session);
  ASSERT_NE(executor, nullptr);

  ModelExecuteArg execute_args;
  ModelLoadArg load_args(nullptr);
  ASSERT_EQ(executor->Load(execute_args, load_args), SUCCESS);

  std::vector<gert::Tensor> inputs_holder;
  std::vector<gert::Tensor *> inputs;
  size_t num_inputs = 0U;
  const ModelIoDesc *inputs_desc = executor->GetAllInputsDesc(num_inputs);
  ModelDescToTensors(inputs_desc, num_inputs, inputs_holder);
  for (auto &input_holder : inputs_holder) {
    inputs.emplace_back(&input_holder);
  }

  std::vector<gert::Tensor> outputs_holder;
  std::vector<gert::Tensor *> outputs;
  size_t num_outputs = 0U;
  const ModelIoDesc *outputs_desc = executor->GetAllOutputsDesc(num_outputs);
  ModelDescToTensors(outputs_desc, num_outputs, outputs_holder);
  for (auto &output_holder : outputs_holder) {
    outputs.emplace_back(&output_holder);
  }
  RtV2ExecutorInterface::RunConfig config(1U);
  ASSERT_EQ(executor->Execute(execute_args, inputs.data(), inputs.size(), outputs.data(), outputs.size(), config), SUCCESS);
  ASSERT_EQ(executor->Unload(), SUCCESS);
}
TEST_F(UtestRtV2PipelineExecutor, test_pipeline_executor_create_with_allocator_success) {
  MagicOpFakerBuilder("FakeGetNext").KernelFunc(OwnedMemoryKernel).Build(*stub_);
  MagicOpFakerBuilder("FakeCollectAllInputs").KernelFunc(OwnedMemoryKernel).Build(*stub_);

  auto graph = ShareGraph::Build2StageGraph();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  ge::DevResourceAllocator allocator;
  RtSession session;
  auto executor = RtV2PipelineExecutor::Create(ge_root_model, allocator, &session);
  ASSERT_NE(executor, nullptr);

}
