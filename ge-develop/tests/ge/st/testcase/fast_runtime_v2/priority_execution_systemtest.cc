/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/gert_api.h"
#include <gtest/gtest.h>
#include "lowering/model_converter.h"

#include "common/share_graph.h"
#include "common/helper/model_helper.h"
#include "common/helper/model_parser_base.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "faker/ge_model_builder.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/fake_value.h"
#include "faker/aicpu_taskdef_faker.h"
#include "faker/space_registry_faker.h"
#include "aicore/launch_kernel/ai_core_launch_kernel.h"
#include "stub/gert_runtime_stub.h"
#include "check/memory_allocation_checker.h"
#include "check/executor_statistician.h"
#include "graph/debug/ge_attr_define.h"

namespace gert {
namespace {
const char *const TransDataStubName = "TransDataStubBin";
const char *const TransData13StubName = "TransData17StubBin";
const char *const DynamicAtomicStubName = "DynamicAtomicBin";
const char *const DynamicRnnv3StubName = "DynamicRNNV3StubBin";
const char *const AddStubName = "AddStubBin";
const char *const MulStubName = "MulStubBin";
const int64_t kExpansion = 10;
const int64_t kDecrease = 5;
struct ExeGraph2Model {
  ge::ExecuteGraphPtr exe_graph;
  ge::GeRootModelPtr root_model;
};
ExeGraph2Model GenerateLstmpExeGraph() {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  GE_DUMP(graph, "LstmpST_ComputeGraph");
  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("TransData",
                      AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName).WithHandle())
          .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName))
          .BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  GE_ASSERT_NOTNULL(exe_graph);
  ge::DumpGraph(exe_graph.get(), "LstmpST_ExecuteGraph1");
  return {exe_graph, ge_root_model};
}

int64_t GetPriority(const ge::ExecuteGraph *main_graph, const std::string &key_name) {
  int64_t priority = -1;
  for (const auto &node : main_graph->GetAllNodes()) {
    if (node->GetName().find(key_name, 0) == 0) {
      (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "priority", priority);
      break;
    }
  }
  return priority;
}

void CheckKernelPriority(const ge::ComputeGraphPtr &graph, const ge::ExecuteGraph *exe_graph,
                         const std::string &compute_node_name, const std::string &kernel_type,
                         const int64_t decrease = 0) {
  auto launch_priority = GetPriority(exe_graph, kernel_type + "_" + compute_node_name);
  ASSERT_NE(launch_priority, -1);
  auto node_id = graph->FindNode(compute_node_name)->GetOpDesc()->GetId();
  auto expect_priority = (node_id * kExpansion) + (kExpansion - 1) - decrease;
  ASSERT_EQ(expect_priority, launch_priority);
}

void CheckAicpuLaunchTfKernelPriority(const ge::ComputeGraphPtr &graph, const ge::ExecuteGraph *exe_graph,
                                      const std::string &compute_node_name) {
  auto launch_priority = GetPriority(exe_graph, "AicpuLaunchTfKernel_" + compute_node_name);
  ASSERT_NE(launch_priority, -1);
  auto memcpy_launch_priority = GetPriority(exe_graph, "AicpuLaunchTfKernel_" + compute_node_name + "_FakeMemCpy");
  ASSERT_NE(memcpy_launch_priority, -1);
  auto node_id = graph->FindNode(compute_node_name)->GetOpDesc()->GetId();
  ASSERT_EQ((node_id * kExpansion) + (kExpansion - 1), launch_priority);
  ASSERT_EQ(launch_priority, memcpy_launch_priority);
}

void CheckComputeNodeOrderEqualToExecuteNodeOrder(const ge::ComputeGraphPtr &graph, const ExecutorStatistician *ess) {
  std::vector<int64_t> compute_node_order;
  compute_node_order.emplace_back(graph->FindNode("transdata_10")->GetOpDesc()->GetId());
  compute_node_order.emplace_back(graph->FindNode("transdata_8")->GetOpDesc()->GetId());
  compute_node_order.emplace_back(graph->FindNode("transdata_4")->GetOpDesc()->GetId());
  compute_node_order.emplace_back(graph->FindNode("transdata_13")->GetOpDesc()->GetId());
  compute_node_order.emplace_back(graph->FindNode("transdata_15")->GetOpDesc()->GetId());
  compute_node_order.emplace_back(graph->FindNode("transdata_17")->GetOpDesc()->GetId());
  auto pre_compute_node_id = compute_node_order[0];
  for (size_t i = 1; i < compute_node_order.size(); ++i) {
    ASSERT_TRUE(pre_compute_node_id < compute_node_order[i]);
    pre_compute_node_id = compute_node_order[i];
  }

  std::vector<int64_t> execute_node_order;
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_10_AtomicClean", "AtomicLaunchKernelWithFlag"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_10", "LaunchKernelWithHandle"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_8_AtomicClean", "AtomicLaunchKernelWithFlag"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_8", "LaunchKernelWithHandle"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_4_AtomicClean", "AtomicLaunchKernelWithFlag"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_4", "LaunchKernelWithHandle"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_13_AtomicClean", "AtomicLaunchKernelWithFlag"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_13", "LaunchKernelWithHandle"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_15_AtomicClean", "AtomicLaunchKernelWithFlag"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_15", "LaunchKernelWithHandle"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_17_AtomicClean", "AtomicLaunchKernelWithFlag"));
  execute_node_order.emplace_back(
      ess->GetExecuteIndexByNodeNameAndKernelType("transdata_17", "LaunchKernelWithHandle"));
  auto pre_execute_node_id = execute_node_order[0];
  for (size_t i = 1; i < execute_node_order.size(); ++i) {
    ASSERT_TRUE(pre_execute_node_id < execute_node_order[i]);
    pre_execute_node_id = execute_node_order[i];
  }
}

ge::graphStatus StubLaunch_UpdateOutputShape(KernelContext *context) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  if (strcmp(extend_context->GetComputeNodeInfo()->GetNodeType(), "NonZero") != 0) {
    return ge::GRAPH_SUCCESS;
  }
  auto shape_buffer =
      context->GetInputValue<gert::TensorData *>(static_cast<size_t>(kernel::InputCommon::kShapeBufferAddr));
  uint32_t g_stub_output_shape[8 + 1] = {4, 1, 2, 3, 1};
  GE_ASSERT_EOK(
      memcpy_s(shape_buffer->GetAddr(), shape_buffer->GetSize(), g_stub_output_shape, sizeof(g_stub_output_shape)));
  return ge::GRAPH_SUCCESS;
}
}  // namespace
class PriorityExecutionST : public testing::Test {};
/**
 * 用例描述： 当需要做内存操作时，空闲内存的释放总是在内存申请之前进行
 *
 * 预置条件：
 * 1. Fake LSTMP网络中的算子的整套实现
 *
 * 测试步骤：
 * 1. 构造lstmp网络
 * 2. lowering成执行图、加载成基于优先级调度的执行器
 * 3. 向执行器注入subscriber
 * 4. 执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 基于算法检查：在一块内存无人使用后，接下来的内存相关kernel一定是先释放后申请
 * 3. 基于算法检查：在一块内存申请后，接下来的launch必然会使用到这块内存
 */
TEST_F(PriorityExecutionST, Memory_FreeBeforeAlloc) {
  auto exe_graph_2_model = GenerateLstmpExeGraph();
  auto exe_graph = exe_graph_2_model.exe_graph;
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, exe_graph_2_model.root_model);
  ASSERT_NE(model_executor, nullptr);

  GertRuntimeStub runtime_stub;

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto checker = model_executor->GetSubscribers().AddSubscriber<MemoryAllocationChecker>(exe_graph);
  ASSERT_NE(checker, nullptr);

  auto outputs = FakeTensors({2048}, 3);
  auto inputs = FakeTensors({2048}, 3);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  checker->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_TRUE(checker->CheckFreeEarlyEnough());
  ASSERT_TRUE(checker->CheckAllocLatelyEnough());

  checker->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(checker->CheckFreeEarlyEnough());
  ASSERT_TRUE(checker->CheckAllocLatelyEnough());

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  runtime_stub.Clear();
  rtStreamDestroy(stream);
}

/**
 * 用例描述： 校验AicoreLaunch的优先级
 *
 * 预置条件：
 * 1. 构造包含aicore算子的计算图
 *
 * 测试步骤：
 * 1. 构造计算图
 * 2. lowering成执行图、加载成基于优先级调度的执行器
 * 3. 执行
 *
 * 预期结果：
 * 1. Launch的优先级为计算图算子的id * 10 + 9，附属新增的Atomic计算节点的Launch优先级为id * 10 + 9 - 5
 * 2. 执行成功
 */
TEST_F(PriorityExecutionST, Check_Priority_With_Aicore_Atomic_success) {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("TransData",
                      AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName).WithHandle())
          .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName))
          .BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  CheckKernelPriority(graph, exe_graph.get(), "transdata_10", "LaunchKernelWithHandle");
  CheckKernelPriority(graph, exe_graph.get(), "transdata_10", "AtomicLaunchKernelWithFlag", kDecrease);
  CheckKernelPriority(graph, exe_graph.get(), "transdata_8", "LaunchKernelWithHandle");
  CheckKernelPriority(graph, exe_graph.get(), "transdata_8", "AtomicLaunchKernelWithFlag", kDecrease);
  CheckKernelPriority(graph, exe_graph.get(), "transdata_4", "LaunchKernelWithHandle");
  CheckKernelPriority(graph, exe_graph.get(), "transdata_4", "AtomicLaunchKernelWithFlag", kDecrease);
  CheckKernelPriority(graph, exe_graph.get(), "transdata_13", "LaunchKernelWithHandle");
  CheckKernelPriority(graph, exe_graph.get(), "transdata_13", "AtomicLaunchKernelWithFlag", kDecrease);
  CheckKernelPriority(graph, exe_graph.get(), "transdata_15", "LaunchKernelWithHandle");
  CheckKernelPriority(graph, exe_graph.get(), "transdata_15", "AtomicLaunchKernelWithFlag", kDecrease);
  CheckKernelPriority(graph, exe_graph.get(), "transdata_17", "LaunchKernelWithHandle");
  CheckKernelPriority(graph, exe_graph.get(), "transdata_17", "AtomicLaunchKernelWithFlag", kDecrease);

  auto main_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main");
  ASSERT_NE(main_node, nullptr);
  auto main_graph = ge::FastNodeUtils::GetSubgraphFromNode(main_node, 0);
  ASSERT_NE(main_graph, nullptr);
  auto priority1 = GetPriority(main_graph, "SelectL2Allocator_cell_state_float32");
  ASSERT_NE(priority1, -1);
  auto priority2 = GetPriority(main_graph, "AtomicLaunchKernelWithFlag_transdata_10_AtomicClean");
  // SelectAllocator被最高优先级的target节点标定
  ASSERT_EQ(priority1, priority2);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto ess = StartExecutorStatistician(model_executor);

  GertRuntimeStub runtime_stub;

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 3);
  auto inputs = FakeTensors({2048}, 3);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  CheckComputeNodeOrderEqualToExecuteNodeOrder(graph, ess);

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  CheckComputeNodeOrderEqualToExecuteNodeOrder(graph, ess);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  runtime_stub.Clear();
  rtStreamDestroy(stream);
}

/**
 * 用例描述： 校验AicpuLaunch的优先级
 *
 * 预置条件：
 * 1. 构造包含aicpu算子的计算图
 *
 * 测试步骤：
 * 1. 构造计算图
 * 2. lowering成执行图、加载成基于优先级调度的执行器
 * 3. 执行
 *
 * 预期结果：
 * 1. Launch的优先级为计算图算子的id * 10 + 9，附属新增的MemCpy计算节点的Launch优先级为id * 10 + 9
 * 2. 执行成功
 */
TEST_F(PriorityExecutionST, Check_Priority_With_Aicpu_MemCpy_success) {
  auto graph = ShareGraph::Aicpu4thGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  graph->FindNode("add2")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  CheckAicpuLaunchTfKernelPriority(graph, exe_graph.get(), "add1");

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048, 1, 1, 1}, 1);
  auto inputs = FakeTensors({2048, 1, 1, 1}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述： 校验静态子图Launch的优先级
 *
 * 预置条件：
 * 1. 构造包含静态子图的计算图
 *
 * 测试步骤：
 * 1. 构造计算图
 * 2. lowering成执行图、加载成基于优先级调度的执行器
 * 3. 执行
 *
 * 预期结果：
 * 1. 静态子图Launch的优先级为计算图算子的id * 10 + 9
 * 2. 执行成功
 */
TEST_F(PriorityExecutionST, Check_Priority_With_DavinciModelExecute_success) {
  auto graph = ShareGraph::BuildDynamicAndStaticGraph();
  graph->TopologicalSorting();
  auto ge_model_builder = GeModelBuilder(graph);
  auto ge_root_model = ge_model_builder.BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  CheckKernelPriority(graph, exe_graph.get(), "known_op", "DavinciModelExecute");

  auto executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(executor, nullptr);

  ge::graphStatus ret = executor->Load();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2, 2}, 2);
  auto inputs = FakeTensors({2, 2}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ret = executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size());
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  executor->UnLoad();
  rtStreamDestroy(stream);
}

/**
 * 用例描述： 校验第三类算子Launch和SyncStream的优先级
 *
 * 预置条件：
 * 1. 构造包含第三类算子的计算图
 *
 * 测试步骤：
 * 1. 构造计算图
 * 2. lowering成执行图、加载成基于优先级调度的执行器
 * 3. 执行
 *
 * 预期结果：
 * 1. 第三类算子Launch和SyncStream的优先级为计算图算子的id * 10 + 9
 * 2. 执行成功
 */
TEST_F(PriorityExecutionST, Check_Priority_With_ThirdNode_success) {
  auto graph = ShareGraph::BuildAiCoreThirdClassNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .AddTaskDef("NonZero", AiCoreTaskDefFaker("NonZeroStubBin").WithHandle())
      .BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  CheckKernelPriority(graph, exe_graph.get(), "add1", "LaunchKernelWithHandle");
  CheckKernelPriority(graph, exe_graph.get(), "nonzero", "LaunchKernelWithHandle");
  CheckKernelPriority(graph, exe_graph.get(), "nonzero", "SyncStream");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevel(1);
  runtime_stub.GetKernelStub().StubTiling().SetUp("LaunchKernelWithHandle", StubLaunch_UpdateOutputShape);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({1, 2, 3, 4}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex("output original shapes") != -1);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex("output storage shapes") != -1);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述： 校验不带图展开的om在加载阶段将动态子图展开
 *
 * 预置条件：
 * 1. 用ShareGraph::BuildDynamicAndStaticGraph()构造包含动态子图的om.
 *
 * 测试步骤：
 * 1. LoadExecutorFromFile
 * 2. 执行
 *
 * 预期结果：
 * 1. 根图里只有unkown_op，没有mul
 * 2. 执行节点里有mul的launch节点
 */
TEST_F(PriorityExecutionST, LoadFromOm_Check_Priority_With_DavinciModelExecute_success) {
  gert::CreateVersionInfo();
  std::string case_dir = __FILE__;
  std::size_t idx = case_dir.find_last_of("/");
  case_dir = case_dir.substr(0, idx);
  std::string om_path = case_dir + "/../../st_run_data/origin_model/dynamic_and_static_graph_without_unfold.om";
  ge::ModelData model_data;
  ge::ModelParserBase::LoadFromFile(om_path.c_str(), -1, model_data);
  ge::ModelHelper model_helper;
  auto ret = model_helper.LoadRootModel(model_data);
  ASSERT_EQ(ret, ge::SUCCESS);
  auto ge_root_model = model_helper.GetGeRootModel();
  ASSERT_NE(ge_root_model, nullptr);
  auto root_graph = ge_root_model->GetRootGraph();

  for (auto node : root_graph->GetAllNodes()) {
    if (node->GetName() == "add") {
      auto op_desc = node->GetOpDesc();
      // cust aicpu kernel
      const char kernel_bin[] = "test";
      vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
      ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("add", std::move(buffer));
      op_desc->SetExtAttr(ge::OP_EXTATTR_CUSTAICPU_KERNEL, kernel_bin_ptr);
    }
  }

  ge::NodePtr unknown_op_node = root_graph->FindNode("unknown_op");
  ASSERT_NE(unknown_op_node, nullptr);
  ge::NodePtr mul_node = root_graph->FindNode("mul");
  ASSERT_EQ(mul_node, nullptr);

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  auto executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(executor, nullptr);

  auto ess = StartExecutorStatistician(executor);

  ret = executor->Load();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2, 2}, 2);
  auto inputs = FakeTensors({2, 2}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ret = executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size());
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  ASSERT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("mul", "AicpuLaunchCCKernel"), 1);

  executor->UnLoad();
  rtStreamDestroy(stream);
  gert::DestroyVersionInfo();
  delete[] static_cast<ge::char_t *>(model_data.model_data);
  model_data.model_data = nullptr;
}
}  // namespace gert