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
#include <memory>
#include <numeric>
#include <string>
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "graph/load/model_manager/model_manager.h"
#include "runtime/base.h"
#include "ge/ge_api.h"
#include "ge/ge_api_error_codes.h"
#include "ge/ge_graph_compile_summary.h"
#include "graph/execute/model_executor.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/load/model_manager/model_utils.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "utils/mock_ops_kernel_builder.h"
#include "utils/taskdef_builder.h"
#include "stub/gert_runtime_stub.h"
#include "graph/manager/mem_manager.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "exe_graph/runtime/tensor.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "common/global_variables/diagnose_switch.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/utils/attr_utils.h"


namespace ge {
namespace {
class ExternalAllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    void *mem = nullptr;
    (void)rtMalloc(&mem, size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
    malloc_cnt++;
    return new (std::nothrow) MemBlock(*this, mem, size);
  }
  MemBlock *MallocAdvise(size_t size, void *addr) override {
    malloc_advise_cnt++;
    return Malloc(size);
  }
  void Free(MemBlock *block) override {
    if (block != nullptr) {
      rtFree(block->GetAddr());
      delete block;
    }
  }
  uint32_t GetMallocCnt() {
    return malloc_cnt;
  }
  uint32_t GetMallocAdviseCnt() {
    return malloc_advise_cnt;
  }
 private:
  uint32_t malloc_cnt = 0;
  uint32_t malloc_advise_cnt = 0;
};

void MockGenerateTask() {
  auto aicore_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");
    ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
    ge::OpKernelBinPtr tbeKernelPtr = std::make_shared<ge::OpKernelBin>("test_tvm", std::move(buffer));
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  auto rts_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    return SUCCESS;
  };

  MockForGenerateTask("AIcoreEngine", aicore_func);
  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("RTSLib", rts_func);
}

void ConstructInputOutputTensor(std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs,
                                size_t output_num = 1U) {
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  TensorDesc desc_1(Shape({1, 2, 3, 4}));
  desc_1.SetPlacement(kPlacementDevice);
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> input_data_2(1, 0);
  TensorDesc desc_2(Shape({1}));
  desc_2.SetPlacement(kPlacementDevice);
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(bool));
  inputs.emplace_back(input_tensor_2);

  for (size_t i = 0; i < output_num; ++i) {
    std::vector<uint8_t> output_data_1(96, 0xff);
    TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
    output_desc_1.SetPlacement(kPlacementDevice);
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);
  }
  return;
}
void ConstructInputOutputTensorForMultiBatch(std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs,
                                             size_t output_num = 1U) {
  std::vector<int32_t> input_data_1(1, 0);
  TensorDesc desc_1(Shape({1, 1, 1}));
  desc_1.SetPlacement(kPlacementDevice);
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> input_data_2(8, 0);
  TensorDesc desc_2(Shape({2, 2, 2}));
  desc_2.SetPlacement(kPlacementDevice);
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_2);

  std::vector<int32_t> input_data_3(1, 0);
  TensorDesc desc_3(Shape({1, 1, 1}));
  desc_3.SetPlacement(kPlacementDevice);
  ge::Tensor input_tensor_3{desc_3};
  input_tensor_3.SetData(reinterpret_cast<uint8_t *>(input_data_3.data()), input_data_3.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_3);

  std::vector<int32_t> input_data_4(1, 0);
  TensorDesc desc_4(Shape({1, 1, 1}));
  desc_4.SetPlacement(kPlacementDevice);
  ge::Tensor input_tensor_4{desc_4};
  input_tensor_4.SetData(reinterpret_cast<uint8_t *>(input_data_4.data()), input_data_4.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_4);

  for (size_t i = 0; i < output_num; ++i) {
    std::vector<uint8_t> output_data_1(8, 0xff);
    TensorDesc output_desc_1(Shape({2, 2, 2}));
    output_desc_1.SetPlacement(kPlacementDevice);
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);
  }
  return;
}

void ConstructInputOutputGertTensorForMultiBatch(std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs,
                                                 size_t output_num = 1U) {
  inputs.resize(4);
  std::vector<int32_t> input_data_1(1, 0);;
  inputs[0] =  {{{1, 1, 1}, {1, 1, 1}},                       // shape
                {ge::FORMAT_NCHW, ge::FORMAT_NCHW, {}},      // format
                gert::kOnDeviceHbm,                          // placement
                ge::DT_INT32,                                // data type
                (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(8, 0);
  inputs[1] =  {{{2, 2, 2}, {2, 2, 2}},                      // shape
                {ge::FORMAT_NCHW, ge::FORMAT_NCHW, {}},      // format
                gert::kOnDeviceHbm,                          // placement
                ge::DT_INT32,                                // data type
                (void *) input_data_2.data()};

  std::vector<int32_t> input_data_3(1, 0);
  inputs[2] =  {{{1, 1, 1}, {1, 1, 1}},                      // shape
                {ge::FORMAT_NCHW, ge::FORMAT_NCHW, {}},      // format
                gert::kOnDeviceHbm,                          // placement
                ge::DT_INT32,                                // data type
                (void *) input_data_3.data()};

  std::vector<int32_t> input_data_4(1, 0);
  inputs[3] =  {{{1, 1, 1}, {1, 1, 1}},                      // shape
                {ge::FORMAT_NCHW, ge::FORMAT_NCHW, {}},      // format
                gert::kOnDeviceHbm,                          // placement
                ge::DT_INT32,                                // data type
                (void *) input_data_4.data()};

  outputs.resize(output_num);
  for (size_t i = 0; i < output_num; ++i) {
    std::vector<uint8_t> output_data_1(8, 0xff);
    outputs[i] =  {{{2, 2, 2}, {2, 2, 2}},                      // shape
                   {ge::FORMAT_NCHW, ge::FORMAT_NCHW, {}},      // format
                   gert::kOnDeviceHbm,                          // placement
                   ge::DT_INT32,                                // data type
                   (void *) input_data_4.data()};
  }
  return;
}

}

using namespace gert;
class ExternalAllocatorGraphTest : public testing::Test {
 protected:
  void SetUp() {
    ModelManager::GetInstance().ClearAicpuSo();
    MockGenerateTask();
  }
  void TearDown() {
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    OpsKernelBuilderRegistry::GetInstance().Unregister("RTSLib");
  }
};

/*
 * 用例描述: 加载执行静态地址可刷新模型，注册外置allocator

 * 预置条件：
 * 1. 构造用例01中的模型
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    MEMORY_OPTIMIZATION_POLICY=GE:kMemoryPriority
 * 3. 模型编译
 * 4. 注册external allocator
 * 5. 模型执行
 * 6. 模型执行（通过allocator申请内存）
 *
 * 预期结果：
 * 1. 模型执行成功
 */
TEST_F(ExternalAllocatorGraphTest, malloc_const_feature_success) {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  MemManager::Instance().Finalize();
}

TEST_F(ExternalAllocatorGraphTest, dynamic_multi_batch_run_success) {
  GertRuntimeStub runtime_stub;

  GeRunningEnvFaker ge_env;
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("MultiDims", multi_dims).Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeOp(DATA).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp("RefData").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(VARIABLE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
  ge_env.Install(FakeOp(MEMCPYASYNC).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(STREAMMERGE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(STREAMSWITCH).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(IDENTITY).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::INPUT_SHAPE.c_str(), "bed:-1,-1,-1;cat:-1,-1,2;dad:1,1,1;ear:1,1,1");
  options.emplace(ge::DYNAMIC_NODE_TYPE.c_str(), "1");
  options.emplace(ge::kDynamicDims, "1,1,1,2,2;2,2,2,2,2");
  Session session(options);
  auto graph = ShareGraph::MultiBatchGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorForMultiBatch(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  MemManager::Instance().Finalize();
  ge_env.InstallDefault();
}

TEST_F(ExternalAllocatorGraphTest, dynamic_multi_batch_run_and_execute_success) {
  GertRuntimeStub runtime_stub;

  GeRunningEnvFaker ge_env;
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("MultiDims", multi_dims).Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeOp(DATA).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp("RefData").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(VARIABLE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
  ge_env.Install(FakeOp(MEMCPYASYNC).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(STREAMMERGE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(STREAMSWITCH).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(IDENTITY).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace(ge::INPUT_SHAPE.c_str(), "bed:-1,-1,-1;cat:-1,-1,2;dad:1,1,1;ear:1,1,1");
  options.emplace(ge::DYNAMIC_NODE_TYPE.c_str(), "1");
  options.emplace(ge::kDynamicDims, "1,1,1,2,2;2,2,2,2,2");
  Session session(options);
  auto graph = ShareGraph::MultiBatchGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensorForMultiBatch(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  ConstructInputOutputGertTensorForMultiBatch(gert_inputs, gert_outputs);
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  MemManager::Instance().Finalize();
  ge_env.InstallDefault();
}

/*
 * 用例描述: 加载执行静态地址可刷新模型，注册外置allocator

 * 预置条件：
 * 1. 构造用例01中的模型
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=0
 *    MEMORY_OPTIMIZATION_POLICY=GE:kMemoryPriority
 * 3. 模型编译
 * 4. 注册external allocator
 * 5. 模型执行
 * 6. 模型执行（通过allocator申请内存）
 *
 * 预期结果：
 * 1. 模型执行成功
 */
TEST_F(ExternalAllocatorGraphTest, malloc_const_feature_success_unrefresh) {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == false);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  MemManager::Instance().Finalize();
}

TEST_F(ExternalAllocatorGraphTest, host_input_success) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.socVersion", "Ascend910B1");
  options.emplace("ge.exec.hostInputIndexes", "0");

  Session session(options);
  dlog_setlevel(0,0,0);
  auto graph = ShareGraph::BuildSwitchMergeGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindNode("data_1");
  std::vector<int64_t> shape_vec{10, 32, 4, 4};
  ge::GeShape shape_desc = ge::GeShape(shape_vec);
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape_desc);
  auto switch_node = compute_graph->FindNode("switch_1");
  switch_node->GetOpDesc()->MutableInputDesc(1)->SetShape(shape_desc);

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  {
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Tensor> outputs;
    std::vector<int32_t> input_data_1(10 * 32 * 4 * 4, 0);
    TensorDesc desc_1(Shape({10, 32, 4, 4}));
    desc_1.SetPlacement(kPlacementHost);
    ge::Tensor input_tensor_1{desc_1};
    input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
    inputs.emplace_back(input_tensor_1);

    std::vector<int32_t> input_data_2(1, 0);
    TensorDesc desc_2(Shape({1}));
    desc_2.SetPlacement(kPlacementDevice);
    ge::Tensor input_tensor_2{desc_2};
    input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(bool));
    inputs.emplace_back(input_tensor_2);

    std::vector<uint8_t> output_data_1(96, 0xff);
    TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
    output_desc_1.SetPlacement(kPlacementDevice);
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);

    ge::diagnoseSwitch::DisableDumper();
    runtime_stub.Clear();
    setenv("GE_DAVINCI_MODEL_PROFILING", "2", true);
    runtime_stub.Clear();
    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
    EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
    EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
    EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  }

  unsetenv("GE_DAVINCI_MODEL_PROFILING");
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  dlog_setlevel(0,3,0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));

  MemManager::Instance().Finalize();
}

TEST_F(ExternalAllocatorGraphTest, host_input_with_feature_base_refreshable_success) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.socVersion", "Ascend910B1");
  options.emplace("ge.exec.hostInputIndexes", "0");

  Session session(options);
  dlog_setlevel(0,0,0);
  auto graph = ShareGraph::BuildSwitchMergeGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindNode("data_1");
  std::vector<int64_t> shape_vec{10, 32, 4, 4};
  ge::GeShape shape_desc = ge::GeShape(shape_vec);
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape_desc);
  auto switch_node = compute_graph->FindNode("switch_1");
  switch_node->GetOpDesc()->MutableInputDesc(1)->SetShape(shape_desc);

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  {
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Tensor> outputs;
    std::vector<int32_t> input_data_1(10 * 32 * 4 * 4, 0);
    TensorDesc desc_1(Shape({10, 32, 4, 4}));
    desc_1.SetPlacement(kPlacementHost);
    ge::Tensor input_tensor_1{desc_1};
    input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
    inputs.emplace_back(input_tensor_1);

    std::vector<int32_t> input_data_2(1, 0);
    TensorDesc desc_2(Shape({1}));
    desc_2.SetPlacement(kPlacementDevice);
    ge::Tensor input_tensor_2{desc_2};
    input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(bool));
    inputs.emplace_back(input_tensor_2);

    std::vector<uint8_t> output_data_1(96, 0xff);
    TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
    output_desc_1.SetPlacement(kPlacementDevice);
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);

    ge::diagnoseSwitch::DisableDumper();
    runtime_stub.Clear();
    setenv("GE_DAVINCI_MODEL_PROFILING", "2", true);
    runtime_stub.Clear();
    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  }

  unsetenv("GE_DAVINCI_MODEL_PROFILING");
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
  dlog_setlevel(0,3,0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));

  MemManager::Instance().Finalize();
}


TEST_F(ExternalAllocatorGraphTest, host_input_with_kernel_bin_success) {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.socVersion", "Ascend910B1");
  options.emplace("ge.exec.hostInputIndexes", "0");

  Session session(options);
  dlog_setlevel(0,0,0);
  auto graph = ShareGraph::BuildSwitchMergeGraph();

  // 构造校大的host内存，走非pice流程
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto data_node = compute_graph->FindNode("data_1");
  std::vector<int64_t> shape_vec{10, 32, 4, 4};
  ge::GeShape shape_desc = ge::GeShape(shape_vec);
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape_desc);
  auto switch_node = compute_graph->FindNode("switch_1");
  switch_node->GetOpDesc()->MutableInputDesc(1)->SetShape(shape_desc);

  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  char ld_library_path_env[MMPA_MAX_PATH] = {};
  (void)mmGetEnv("LD_LIBRARY_PATH", &(ld_library_path_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  std::string new_ld_library_path(ld_library_path_env);
  new_ld_library_path = new_ld_library_path + ":" + "runtime/lib64";
  char_t *kEnvAutoUseUcMemory = "LD_LIBRARY_PATH";
  (void)mmSetEnv(kEnvAutoUseUcMemory, new_ld_library_path.c_str(), 1);
  std::string base_path = "runtime/lib64";
  std::string save_path = base_path + "/UpdateModelParam_dav_2201.o";
  DEF_GRAPH(g1) {
      CHAIN(NODE("cons1", "Const")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    };
  ge::Graph graph_to_file = ToGeGraph(g1);
  graph_to_file.SaveToFile(save_path.c_str());

  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  {
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Tensor> outputs;
    std::vector<int32_t> input_data_1(10 * 32 * 4 * 4, 0);
    TensorDesc desc_1(Shape({10, 32, 4, 4}));
    desc_1.SetPlacement(kPlacementHost);
    ge::Tensor input_tensor_1{desc_1};
    input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
    inputs.emplace_back(input_tensor_1);

    std::vector<int32_t> input_data_2(1, 0);
    TensorDesc desc_2(Shape({1}));
    desc_2.SetPlacement(kPlacementDevice);
    ge::Tensor input_tensor_2{desc_2};
    input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(bool));
    inputs.emplace_back(input_tensor_2);

    std::vector<uint8_t> output_data_1(96, 0xff);
    TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
    output_desc_1.SetPlacement(kPlacementDevice);
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);

    ge::diagnoseSwitch::DisableDumper();
    runtime_stub.Clear();
    setenv("GE_DAVINCI_MODEL_PROFILING", "2", true);
    runtime_stub.Clear();
    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
    EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
    EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
    EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  }

  unsetenv("GE_DAVINCI_MODEL_PROFILING");
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  dlog_setlevel(0,3,0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  remove(save_path.c_str());
  MemManager::Instance().Finalize();
}


TEST_F(ExternalAllocatorGraphTest, host_input_with_kernel_bin_pcie_success) {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.socVersion", "Ascend910B1");
  options.emplace("ge.exec.hostInputIndexes", "1");

  Session session(options);
  dlog_setlevel(0,0,0);
  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  char ld_library_path_env[MMPA_MAX_PATH] = {};
  (void)mmGetEnv("LD_LIBRARY_PATH", &(ld_library_path_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  std::string new_ld_library_path(ld_library_path_env);
  new_ld_library_path = new_ld_library_path + ":" + "runtime/lib64";
  char_t *kEnvAutoUseUcMemory = "LD_LIBRARY_PATH";
  (void)mmSetEnv(kEnvAutoUseUcMemory, new_ld_library_path.c_str(), 1);
  std::string base_path = "runtime/lib64";
  std::string save_path = base_path + "/UpdateModelParam_dav_2201.o";
  DEF_GRAPH(g1) {
      CHAIN(NODE("cons1", "Const")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    };
  ge::Graph graph_to_file = ToGeGraph(g1);
  graph_to_file.SaveToFile(save_path.c_str());

  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  {
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Tensor> outputs;
    std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
    TensorDesc desc_1(Shape({1, 2, 3, 4}));
    desc_1.SetPlacement(kPlacementDevice);
    ge::Tensor input_tensor_1{desc_1};
    input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
    inputs.emplace_back(input_tensor_1);

    std::vector<int32_t> input_data_2(1, 0);
    TensorDesc desc_2(Shape({1}));
    desc_2.SetPlacement(kPlacementHost);
    ge::Tensor input_tensor_2{desc_2};
    input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(bool));
    inputs.emplace_back(input_tensor_2);

    std::vector<uint8_t> output_data_1(96, 0xff);
    TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
    output_desc_1.SetPlacement(kPlacementDevice);
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);

    ge::diagnoseSwitch::DisableDumper();
    runtime_stub.Clear();
    setenv("GE_DAVINCI_MODEL_PROFILING", "2", true);
    runtime_stub.Clear();
    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
    EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
    EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
    EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  }

  {
    std::vector<gert::Tensor> gert_inputs;
    std::vector<gert::Tensor> gert_outputs;
    gert_inputs.resize(2);
    gert_outputs.resize(1);
    std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
    gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnDeviceHbm,                                // placement
                              ge::DT_INT32,                              // data type
                              (void *) input_data_1.data()};

    std::vector<int32_t> input_data_2(1, 666);
    gert_inputs[1] = {{{1}, {1}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnHost,                                // placement
                              ge::DT_BOOL,                              // data type
                              (void *) input_data_2.data()};

    std::vector<int32_t> output_data(1 * 2 * 3 * 4, 666);
    gert_outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                                // placement
                          ge::DT_INT32,                              // data type
                          nullptr};
    gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
        gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>(
            {gert::ProfilingType::kTaskTime, gert::ProfilingType::kDevice}));
    EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
  }

  unsetenv("GE_DAVINCI_MODEL_PROFILING");
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  dlog_setlevel(0,3,0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  remove(save_path.c_str());
  MemManager::Instance().Finalize();
}

TEST_F(ExternalAllocatorGraphTest, malloc_const_feature_addr_refresh_op_success_unrefresh) {
  GertRuntimeStub runtime_stub;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.exec.frozenInputIndexes", "1000");
  options.emplace("ge.socVersion", "Ascend910B1");
  Session session(options);
  dlog_setlevel(0,0,0);
  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  char ld_library_path_env[MMPA_MAX_PATH] = {};
  (void)mmGetEnv("LD_LIBRARY_PATH", &(ld_library_path_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  std::string new_ld_library_path(ld_library_path_env);
  new_ld_library_path = new_ld_library_path + ":" + "runtime/lib64";
  char_t *kEnvAutoUseUcMemory = "LD_LIBRARY_PATH";
  (void)mmSetEnv(kEnvAutoUseUcMemory, new_ld_library_path.c_str(), 1);
  std::string base_path = "runtime/lib64";
  std::string save_path = base_path + "/UpdateModelParam_dav_2201.o";
  DEF_GRAPH(g1) {
      CHAIN(NODE("cons1", "Const")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    };
  ge::Graph graph_to_file = ToGeGraph(g1);
  graph_to_file.SaveToFile(save_path.c_str());

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == false);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));
  setenv("GE_DAVINCI_MODEL_PROFILING", "2", true);
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);

  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 666);
  gert_inputs[1] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_2.data()};

  std::vector<int32_t> output_data(1 * 2 * 3 * 4, 666);
  gert_outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                        {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                        gert::kOnDeviceHbm,                                // placement
                        ge::DT_INT32,                              // data type
                        nullptr};
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>(
          {gert::ProfilingType::kTaskTime, gert::ProfilingType::kDevice}));
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
  unsetenv("GE_DAVINCI_MODEL_PROFILING");
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  dlog_setlevel(0,3,0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  remove(save_path.c_str());
  MemManager::Instance().Finalize();
}


TEST_F(ExternalAllocatorGraphTest, malloc_const_feature_addr_refresh_op_success_overflow_dump_enabled) {
  GertRuntimeStub runtime_stub;
  gert::GlobalDumper::GetInstance()->ClearInnerExceptionDumpers();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kOverflowDump}));
  runtime_stub.Clear();
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "0");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.exec.frozenInputIndexes", "1000");
  options.emplace("ge.socVersion", "Ascend910B1");
  Session session(options);
  dlog_setlevel(0,0,0);
  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  char ld_library_path_env[MMPA_MAX_PATH] = {};
  (void)mmGetEnv("LD_LIBRARY_PATH", &(ld_library_path_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  std::string new_ld_library_path(ld_library_path_env);
  new_ld_library_path = new_ld_library_path + ":" + "runtime/lib64";
  char_t *kEnvAutoUseUcMemory = "LD_LIBRARY_PATH";
  (void)mmSetEnv(kEnvAutoUseUcMemory, new_ld_library_path.c_str(), 1);
  std::string base_path = "runtime/lib64";
  std::string save_path = base_path + "/UpdateModelParam_dav_2201.o";
  DEF_GRAPH(g1) {
      CHAIN(NODE("cons1", "Const")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    };
  ge::Graph graph_to_file = ToGeGraph(g1);
  graph_to_file.SaveToFile(save_path.c_str());

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_TRUE(is_refreshable == false);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();
  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));

  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogRegex("Davinci model add addr refresh kernel");
  EXPECT_FALSE(find_log > 0);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
  runtime_stub.Clear();

  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 666);
  gert_inputs[1] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_2.data()};

  std::vector<int32_t> output_data(1 * 2 * 3 * 4, 666);
  gert_outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                        {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                        gert::kOnDeviceHbm,                                // placement
                        ge::DT_INT32,                              // data type
                        nullptr};
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>(
          {gert::ProfilingType::kTaskTime, gert::ProfilingType::kDevice}));
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
  EXPECT_NE(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  dlog_setlevel(0,3,0);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  remove(save_path.c_str());
  MemManager::Instance().Finalize();
}
/*
 * 用例描述: 加载执行静态地址可刷新模型，使用外部申请的地址，注册外置allocator用于申请输出内存

 * 预置条件：
 * 1. 构造用例01中的模型
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    MEMORY_OPTIMIZATION_POLICY=GE:kMemoryPriority
 * 3. 模型编译
 * 4. 不申请outputs内存
 * 5. 注册external allocator
 * 6. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 */
TEST_F(ExternalAllocatorGraphTest, malloc_outputs_success) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphConstMemoryBase(graph_id, weight_mem.data(), weight_size));
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 1);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_NE(outputs[0].GetData(), nullptr);

  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 0);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_NE(outputs[0].GetData(), nullptr);
  EXPECT_EQ(SUCCESS, session.UnregisterExternalAllocator(stream));
  MemManager::Instance().Finalize();
}

/*
 * 用例描述: 加载执行静态地址可刷新模型，使用外部申请的地址，注册外置allocator用于申请输出内存

 * 预置条件：
 * 1. 构造用例01中的模型
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    MEMORY_OPTIMIZATION_POLICY=GE:kMemoryPriority
 * 3. 模型编译
 * 4. 申请部分outputs内存
 * 5. 注册external allocator
 * 6. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 */
TEST_F(ExternalAllocatorGraphTest, malloc_part_outputs_success) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraphWithTwoOutputs();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  ge::Tensor output_tensor_1(output_desc);
  outputs.emplace_back(output_tensor_1);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 3);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  EXPECT_EQ(outputs.size(), 2);
  EXPECT_NE(outputs[0].GetData(), nullptr);
  EXPECT_NE(outputs[1].GetData(), nullptr);

  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 4);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 1);
  EXPECT_EQ(outputs.size(), 2);
  EXPECT_NE(outputs[0].GetData(), nullptr);
  EXPECT_NE(outputs[1].GetData(), nullptr);
  // release ConstMemBlock
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
}

/*
 * 用例描述: 加载执行静态地址可刷新模型，使用外部申请的地址，注册外置allocator用于申请输出内存

 * 预置条件：
 * 1. 构造用例01中的模型
 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_FEATURE_BASE_REFRESHABLE=1
 *    MEMORY_OPTIMIZATION_POLICY=GE:kMemoryPriority
 * 3. 模型编译
 * 4. 申请2个outputs内存
 * 5. 注册external allocator
 * 6. 模型执行
 *
 * 预期结果：
 * 1. 模型执行成功
 */
TEST_F(ExternalAllocatorGraphTest, malloc_two_outputs_success) {
  GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = ShareGraph::BuildSwitchMergeGraphWithTwoOutputs();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);

  std::vector<uint8_t> weight_mem(weight_size, 0);
  std::vector<uint8_t> feature_mem(feature_size, 0);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 0);
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  rtStream_t stream = (void *)0x10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  EXPECT_EQ(SUCCESS, session.RegisterExternalAllocator(stream, external_allocator));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 4);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 0);
  EXPECT_EQ(outputs.size(), 2);
  EXPECT_NE(outputs[0].GetData(), nullptr);
  EXPECT_NE(outputs[1].GetData(), nullptr);

  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocCnt(), 5);
  EXPECT_EQ(dynamic_cast<ExternalAllocatorUtStub *>(external_allocator.get())->GetMallocAdviseCnt(), 1);
  EXPECT_EQ(outputs.size(), 2);
  EXPECT_NE(outputs[0].GetData(), nullptr);
  EXPECT_NE(outputs[1].GetData(), nullptr);
  EXPECT_EQ(SUCCESS, session.RemoveGraph(graph_id));
}
}  // namespace ge
