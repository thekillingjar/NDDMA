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
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/fake_value.h"
#include "faker/ge_model_builder.h"
#include "stub/gert_runtime_stub.h"
#include "lowering/model_converter.h"
#include "aicpu_task_struct.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "faker/global_data_faker.h"
#include "lowering/graph_converter.h"
#include "runtime/model_v2_executor.h"
#include "hybrid/executor/cann_tracing_profiler.h"
#include "common/global_variables/diagnose_switch.h"
#include "register/op_tiling_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "hybrid/executor/hybrid_model_rt_v2_executor.h"
#include "faker/model_desc_holder_faker.h"

namespace gert {
class CannProfilerST : public bg::BgTest {
 public:
  static void BuildExecutorInner(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option) {
    auto graph = ShareGraph::BuildSingleNodeGraph();
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
    ModelConverter::Args args{option, nullptr, nullptr, nullptr, nullptr};
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(graph);
    ge::ModelData model_data{};
    model_data.om_name = "test";
    RtSession session;
    model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model, &session);
    ASSERT_NE(model_executor, nullptr);
  }

  static void BuildExecutorInner2(std::unique_ptr<ModelV2Executor> &model_executor, const LoweringOption &option) {
    auto graph = ShareGraph::BuildTwoAddNodeGraph();
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).
        AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
    ModelConverter::Args args{option, nullptr, nullptr, nullptr, nullptr};
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    GertRuntimeStub fakeRuntime;
    fakeRuntime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(graph);
    ge::ModelData model_data{};
    model_data.om_name = "test";
    RtSession session;
    model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model, &session);
    ASSERT_NE(model_executor, nullptr);
  }

  static std::unique_ptr<ModelV2Executor> BuildExecutorForTrace(LoweringOption option = {}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildExecutorInner2(model_executor, option);
    return model_executor;
  }

  static std::unique_ptr<ModelV2Executor> BuildExecutor(LoweringOption option = {}) {
    std::unique_ptr<ModelV2Executor> model_executor;
    BuildExecutorInner(model_executor, option);
    return model_executor;
  }

 protected:
  void TearDown() {
    ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0);
  }
};

/**
 * 用例描述：单算子用例执行时打开profiling的api开关，统计上报信息数量
 *
 * 预置条件:
 * 1.构造单算子执行器
 *
 * 测试步骤：
 * 1. 使能cann trace profiling的Api开关
 * 2. 构造单执行器执行
 *
 * 预期结果：
 * 1. 执行成功
 */
TEST_F(CannProfilerST, CannTracingProfiler_Test_single_op) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTrainingTrace});
  auto model_executor = CannProfilerST::BuildExecutor();
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  SubscriberExtendInfo extend_info;
  extend_info.executor = model_executor.get();
  extend_info.node_names_to_attrs["add1"] = {true, true, 20002};
  model_executor->GetSubscribers().AddSubscriber<CannTracingProfiler>(kMainExeGraph, extend_info);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
            reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：图模式用例执行时打开profiling的api开关，统计上报信息数量
 *
 * 预置条件:
 * 1.构造图模式执行器
 *
 * 测试步骤：
 * 1. 使能cann trace profiling的Api开关
 * 2. 构造单执行器执行
 *
 * 预期结果：
 * 1. 执行成功
 */
TEST_F(CannProfilerST, CannTracingProfiler_Test_graph) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTrainingTrace});
  auto model_executor = CannProfilerST::BuildExecutorForTrace();
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  SubscriberExtendInfo extend_info;
  extend_info.executor = model_executor.get();
  extend_info.node_names_to_attrs["add1"] = {true, false, 20002};
  extend_info.node_names_to_attrs["add2"] = {true, true, -1};
  model_executor->GetSubscribers().AddSubscriber<CannTracingProfiler>(kMainExeGraph, extend_info);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
            reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：异常场景，没有注册CannTracingProfiler的场景
 *
 * 预置条件:
 * 1.构造图模式执行器
 *
 * 测试步骤：
 * 1. 使能cann trace profiling的Api开关
 * 2. 构造执行器执行
 *
 * 预期结果：
 * 1. 执行成功
 */
TEST_F(CannProfilerST, CannTracingProfiler_Test_profiler_null_ptr) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTrainingTrace});
  auto model_executor = CannProfilerST::BuildExecutorForTrace();
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  SubscriberExtendInfo extend_info;
  extend_info.executor = model_executor.get();
  extend_info.executor->GetExeGraphExecutor(kMainExeGraph)->SetExecutionData(nullptr, nullptr);
  extend_info.node_names_to_attrs["add1"] = {true, false, 20002};
  extend_info.node_names_to_attrs["add2"] = {true, true, -1};
  auto subscriber = model_executor->GetSubscribers().AddSubscriber<CannTracingProfiler>(
      kMainExeGraph, extend_info);
  subscriber->OnExecuteEvent(0, nullptr, kExecuteStart, nullptr, 0);

  auto execution_data_holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[sizeof(ExecutionData)]);
  auto execution_data = reinterpret_cast<ExecutionData *>(execution_data_holder.get());
  std::unique_ptr<Node> launch_node = std::make_unique<Node>();
  launch_node->node_id = 0;
  launch_node->context.kernel_extend_info = nullptr;
  // build fake execution data with 2 exe nodes
  execution_data->base_ed.node_num = 1;
  std::vector<Node *> nodes{launch_node.get()};
  execution_data->base_ed.nodes = nodes.data();
  extend_info.executor->GetExeGraphExecutor(kMainExeGraph)->SetExecutionData(execution_data, nullptr);
  subscriber->Init();
}
}  // namespace gert
