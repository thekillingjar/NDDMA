/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/builder/graph_executor_builder.h"
#include <gtest/gtest.h>
#include <ge_runtime_stub/include/faker/model_desc_holder_faker.h>
#include "common/bg_test.h"
#include "common/exe_graph.h"
#include "common/share_graph.h"
#include "faker/exe_graph_model_level_data_faker.h"
#include "faker/global_data_faker.h"
#include "stub/gert_runtime_stub.h"

#include "exe_graph/runtime/context_extend.h"
#include "lowering/graph_converter.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
namespace gert {
namespace {
ge::FastNode* FindData(const ge::ExecuteGraphPtr &graph, int64_t index) {
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Data") {
      int64_t feed_index = std::numeric_limits<int64_t>::min();
      ge::AttrUtils::GetInt(node->GetOpDescPtr(), "index", feed_index);
      if (feed_index == index) {
        return node;
      }
    }
  }
  return nullptr;
}
}  // namespace
class GraphExecutorBuilderUT : public bg::BgTestAutoCreate3StageFrame {};

/*
 *              NetOutput
 *             /      |
 *          Bar20     |
 *          /   \     |
 *     Foo21    OutputData
 *     /   \
 * data0    data1
 */
bg::ExeGraph BuildGraph1() {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto data1 = bg::ValueHolder::CreateFeed(1);
  auto foo21 = bg::ValueHolder::CreateSingleDataOutput("Foo21", {data0, data1});
  auto od = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto bar20 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Bar20", {foo21, od});
  auto main_frame = bg::ValueHolder::PopGraphFrame({od}, {bar20}, "NetOutput");
  auto root_frame = bg::ValueHolder::PopGraphFrame();
  return bg::ExeGraph(root_frame->GetExecuteGraph());
}

bg::ExeGraph BuildGraph2(size_t launch_num) {
  while (bg::ValueHolder::PopGraphFrame() != nullptr)
    ;
  auto compute_graph = ShareGraph::BuildLotsOfNodes(launch_num);
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  return bg::ExeGraph(exe_graph);
}

bg::ExeGraph BuildGraph3(size_t launch_num) {
  while (bg::ValueHolder::PopGraphFrame() != nullptr)
    ;
  auto compute_graph = ShareGraph::BuildLotsOfNodes(launch_num);
  compute_graph->TopologicalSorting();
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "Add") {
      ge::AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", "test");
      ge::AttrUtils::SetBool(node->GetOpDesc(), ge::kPartiallySupported, true);
      ge::AttrUtils::SetStr(node->GetOpDesc(), ge::kAICpuKernelLibName, ge::kEngineNameAiCpuTf);
    }
  }
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, true).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  return bg::ExeGraph(exe_graph);
}

bg::ExeGraph BuildThirdClassGraph() {
  while (bg::ValueHolder::PopGraphFrame() != nullptr)
    ;
  auto graph = ShareGraph::BuildAiCoreThirdClassNodeGraph();
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data =
      GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).FakeWithHandleAiCore("NonZero", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  return bg::ExeGraph(exe_graph);
}

/*
 *             NetOutput
 *           c/   c|    c\
 *      Bar20   Bar20    Bar20
 *       | |      | |     | |
 *       | |      | |     | +-----------------+
 *       | |      | +-----|-----------------+ |
 *       | +------|-------|---------------+ | |
 *       |        |     Foo21             | | |
 *       |        |     |   \           OutputData
 *       \    +---+-----+  InnerData
 *        \   |
 *        Foo22
 *       /   |
 *   Foo11   |  Foo11
 *    |      \  /
 *  data0   data1
 */

TEST_F(GraphExecutorBuilderUT, TopoExecutionData_Ok) {
  auto exe_graph = BuildGraph1();
  auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetMainGraph()).Build();
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  ExeGraphExecutor executor;

  ASSERT_EQ(GraphExecutorBuilder({reinterpret_cast<ContinuousBuffer *>(mld.compute_node_info.get()),
                                  reinterpret_cast<ContinuousBuffer *>(mld.kernel_extend_info.get())},
                                 exe_graph.GetMainGraph(), &mld.symbols_to_value)
                .Build(ExecutorType::kTopologicalPriority, executor),
            ge::GRAPH_SUCCESS);

  auto execution_data = static_cast<const ExecutionData *>(executor.GetExecutionData());
  ASSERT_NE(execution_data, nullptr);

  auto foo21 = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.GetMainGraph().get(), "Foo21");
  auto bar20 = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.GetMainGraph().get(), "Bar20");
  auto data0 = FindData(exe_graph.GetMainGraph(), 0);
  auto data1 = FindData(exe_graph.GetMainGraph(), 1);
  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.GetMainGraph().get(), "NetOutput");
  ASSERT_NE(foo21, nullptr);
  ASSERT_NE(bar20, nullptr);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  ASSERT_NE(netoutput, nullptr);

  ASSERT_EQ(execution_data->base_ed.node_num, 2u);

  // check foo21 node
  EXPECT_STREQ(static_cast<const KernelExtendInfo *>(execution_data->base_ed.nodes[0]->context.kernel_extend_info)
                   ->GetKernelName(),
               foo21->GetName().c_str());
  EXPECT_STREQ(static_cast<const KernelExtendInfo *>(execution_data->base_ed.nodes[0]->context.kernel_extend_info)
                   ->GetKernelType(),
               foo21->GetType().c_str());

  EXPECT_STREQ(
      static_cast<const ComputeNodeInfo *>(execution_data->base_ed.nodes[0]->context.compute_node_info)->GetNodeName(),
      "FakeName");
  EXPECT_STREQ(
      static_cast<const ComputeNodeInfo *>(execution_data->base_ed.nodes[0]->context.compute_node_info)->GetNodeType(),
      "FakeType");

  ASSERT_EQ(execution_data->base_ed.nodes[0]->context.input_size, 2U);
  ASSERT_EQ(execution_data->base_ed.nodes[0]->context.output_size, 1U);
  // EXPECT_EQ(execution_data->base_ed.nodes[0]->context.values[0], mld.gav.nodes_to_input_values[foo21.get()].at(0));
  // EXPECT_EQ(execution_data->base_ed.nodes[0]->context.values[1], mld.gav.nodes_to_input_values[foo21.get()].at(1));
  // EXPECT_EQ(execution_data->base_ed.nodes[0]->context.values[2], mld.gav.nodes_to_output_values[foo21.get()].at(0));

  // check bar20 node
  EXPECT_STREQ(static_cast<const KernelExtendInfo *>(execution_data->base_ed.nodes[1]->context.kernel_extend_info)
                   ->GetKernelName(),
               bar20->GetName().c_str());
  EXPECT_STREQ(static_cast<const KernelExtendInfo *>(execution_data->base_ed.nodes[1]->context.kernel_extend_info)
                   ->GetKernelType(),
               bar20->GetType().c_str());

  EXPECT_STREQ(
      static_cast<const ComputeNodeInfo *>(execution_data->base_ed.nodes[1]->context.compute_node_info)->GetNodeName(),
      "FakeName");
  EXPECT_STREQ(
      static_cast<const ComputeNodeInfo *>(execution_data->base_ed.nodes[1]->context.compute_node_info)->GetNodeType(),
      "FakeType");

  ASSERT_EQ(execution_data->base_ed.nodes[1]->context.input_size, 2U);
  ASSERT_EQ(execution_data->base_ed.nodes[1]->context.output_size, 0U);
  // EXPECT_EQ(execution_data->base_ed.nodes[1]->context.values[0], mld.gav.nodes_to_input_values[bar20.get()].at(0));
  // EXPECT_EQ(execution_data->base_ed.nodes[1]->context.values[1], mld.gav.nodes_to_input_values[bar20.get()].at(1));

  // check graph inputs and outputs
  ASSERT_EQ(execution_data->base_ed.input_num, 2U);
  // EXPECT_EQ(execution_data->base_ed.input_values[0], mld.gav.nodes_to_output_values[data0.get()].at(0));
  // EXPECT_EQ(execution_data->base_ed.input_values[1], mld.gav.nodes_to_output_values[data1.get()].at(0));

  ASSERT_EQ(execution_data->base_ed.output_num, 1U);
  // EXPECT_EQ(execution_data->base_ed.output_values[0], mld.gav.nodes_to_input_values[netoutput.get()].at(0));

  // check indgree
  EXPECT_EQ(execution_data->node_indegrees[0], 0);
  EXPECT_EQ(execution_data->node_indegrees[1], 1);
  EXPECT_EQ(execution_data->node_indegrees_backup[0], 0);
  EXPECT_EQ(execution_data->node_indegrees_backup[1], 1);

  // check start nodes
  ASSERT_EQ(execution_data->start_num, 1U);
  ASSERT_EQ(execution_data->start_nodes[0], execution_data->base_ed.nodes[0]);

  // check watcher
  ASSERT_EQ(execution_data->node_watchers[0]->watch_num, 1U);
  ASSERT_EQ(execution_data->node_watchers[0]->node_ids[0], 1);

  ASSERT_EQ(execution_data->node_watchers[1]->watch_num, 0U);
}

TEST_F(GraphExecutorBuilderUT, ExecutorSelection_SelectSeqPriority_WhenNoIfOrWhile) {
  auto exe_graph = BuildGraph2(50);
  auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetRootGraph()).Build();
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_INFO);
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  stub.GetSlogStub().Clear();
  ExeGraphExecutor executor;
  ASSERT_EQ(GraphExecutorBuilder({reinterpret_cast<ContinuousBuffer *>(mld.compute_node_info.get()),
                                  reinterpret_cast<ContinuousBuffer *>(mld.kernel_extend_info.get())},
                                 exe_graph.GetMainGraph(), &mld.symbols_to_value)
                .Build(executor),
            ge::GRAPH_SUCCESS);

  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, "Select executor type sequential-executor for graph Main_"), -1);
}
TEST_F(GraphExecutorBuilderUT, ExecutorSelection_SelectTopo_WhenIfAnd4Launch) {
  auto exe_graph = BuildGraph3(2);
  ASSERT_NE(exe_graph.GetRootGraph(), nullptr);
  auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetRootGraph()).Build();
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_INFO);
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  stub.GetSlogStub().Clear();
  ExeGraphExecutor executor;
  ASSERT_EQ(GraphExecutorBuilder({reinterpret_cast<ContinuousBuffer *>(mld.compute_node_info.get()),
                                  reinterpret_cast<ContinuousBuffer *>(mld.kernel_extend_info.get())},
                                 exe_graph.GetMainGraph(), &mld.symbols_to_value)
                .Build(executor),
            ge::GRAPH_SUCCESS);

  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, "Select executor type general-topo-executor for graph Main_"), -1);
}
TEST_F(GraphExecutorBuilderUT, ExecutorSelection_SelectTopoPriority_WhenIfAnd6Launch) {
  auto exe_graph = BuildGraph3(3);
  auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetRootGraph()).Build();
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_INFO);
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  stub.GetSlogStub().Clear();
  ExeGraphExecutor executor;
  ASSERT_EQ(GraphExecutorBuilder({reinterpret_cast<ContinuousBuffer *>(mld.compute_node_info.get()),
                                  reinterpret_cast<ContinuousBuffer *>(mld.kernel_extend_info.get())},
                                 exe_graph.GetMainGraph(), &mld.symbols_to_value)
                .Build(executor),
            ge::GRAPH_SUCCESS);

  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, "Select executor type priority-topo-executor for graph Main_"), -1);
}

TEST_F(GraphExecutorBuilderUT, ExecutorSelection_SelectTopoMultiThread_WhenMultiThreadOption) {
  auto exe_graph = BuildGraph3(3);
  auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetRootGraph()).Build();
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_INFO);
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  stub.GetSlogStub().Clear();
  ExeGraphExecutor executor;
  auto option = MultiThreadExecutorOption(kLeastThreadNumber);
  ASSERT_EQ(GraphExecutorBuilder({reinterpret_cast<ContinuousBuffer *>(mld.compute_node_info.get()),
                                  reinterpret_cast<ContinuousBuffer *>(mld.kernel_extend_info.get())},
                                 exe_graph.GetMainGraph(), &mld.symbols_to_value)
                .ExecutorOpt(option)
                .Build(executor),
            ge::GRAPH_SUCCESS);

  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, "Select executor type multi-thread-topo-executor for graph Main_"),
            -1);
}
}  // namespace gert