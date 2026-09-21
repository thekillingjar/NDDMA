/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/builder/equivalent_data_edges.h"
#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "exe_graph/runtime/tiling_context.h"
#include "faker/space_registry_faker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "exe_graph/lowering/value_holder.h"
#include "graph/fast_graph/edge.h"
#include "graph_builder/bg_condition.h"
#include "graph/utils/execute_graph_utils.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "lowering/graph_converter.h"

namespace gert {
namespace {
constexpr uint32_t kInputEnd = 0U;
constexpr uint32_t kOutputEnd = 1U;
}  // namespace
class EquivalentDataEdgesUT : public bg::BgTest {
 public:
  // 0~31bit表示id, 32~62bit表示index, 63bit表示direction
  struct EndPoint {
    uint64_t id : 32;
    uint64_t index : 31;
    uint64_t direction : 1;

    std::string ToString() const {
      std::stringstream ss;
      ss << "EndPoint(" << id << ", " << index << ", " << direction << ") ";
      return ss.str();
    }
  };

  void CheckSameSymbol(const std::vector<uint64_t> &symbols) {
    ASSERT_GT(symbols.size(), 1U);
    const auto symbol = *symbols.begin();
    for (const auto s : symbols) {
      EXPECT_EQ(s, symbol);
    }
  }

  // 返回值：{symbol总个数，target_symbol的个数}
  std::pair<size_t, size_t> CalcSymbolSize(const ge::ExecuteGraph *const graph,
                                           uint64_t target_symbol = ge::kInvalidSymbol) {
    size_t size_of_target_symbol = 0U;
    std::unordered_set<uint64_t> symbols_set;
    for (const auto n : graph->GetAllNodes()) {
      for (size_t i = 0U; i < n->GetDataOutNum(); ++i) {
        symbols_set.emplace(n->GetExtendInfo()->GetOutputSymbol(i));

        if (target_symbol == n->GetExtendInfo()->GetOutputSymbol(i)) {
          ++size_of_target_symbol;
        }
      }
      for (size_t i = 0U; i < n->GetDataInNum(); ++i) {
        if (target_symbol == n->GetExtendInfo()->GetInputSymbol(i)) {
          ++size_of_target_symbol;
        }
        symbols_set.emplace(n->GetExtendInfo()->GetInputSymbol(i));
      }
    }
    return {symbols_set.size(), size_of_target_symbol};
  }
};

TEST_F(EquivalentDataEdgesUT, InOutEqOk1) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("node1", "Const")->NODE("node2", "FindInferShapeFunc")->NODE("NetOutput", "NetOutput"));
  };

  auto graph = ge::ToExecuteGraph(g1);
  graph->TopologicalSorting();

  EquivalentDataEdges eq;
  ASSERT_EQ(eq.UpdateEquivalentEdges(graph.get()), ge::GRAPH_SUCCESS);
  auto node1 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node1");
  auto node2 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node2");
  auto netoutput_node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "NetOutput");
  auto all_nodes = graph->GetAllNodes();
  EXPECT_EQ(node1->GetExtendInfo()->GetOutputSymbol(0U), node2->GetExtendInfo()->GetInputSymbol(0U));
  EXPECT_EQ(node2->GetExtendInfo()->GetOutputSymbol(0U), netoutput_node->GetExtendInfo()->GetInputSymbol(0U));
  EXPECT_EQ(CalcSymbolSize(graph.get()).first, 2U);
}

TEST_F(EquivalentDataEdgesUT, InOutEqOk2) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("node1", "Const")->NODE("node2", "FindInferShapeFunc")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("node3", "Const")
              ->NODE("node4", "FindTilingFunc")
              ->NODE("node5", "TilingParse")
              ->EDGE(0, 1)
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("node6", "Const")->EDGE(0, 1)->NODE("node5", "TilingParse"));
  };

  auto graph = ge::ToExecuteGraph(g1);
  graph->TopologicalSorting();

  EquivalentDataEdges eq;
  ASSERT_EQ(eq.UpdateEquivalentEdges(graph.get()), ge::GRAPH_SUCCESS);
  auto node1 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node1");
  auto node2 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node2");
  auto node3 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node3");
  auto node4 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node4");
  auto node5 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node5");
  auto node6 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node6");
  auto netoutput_node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "NetOutput");

  EXPECT_EQ(node1->GetExtendInfo()->GetOutputSymbol(0U), node2->GetExtendInfo()->GetInputSymbol(0U));
  EXPECT_EQ(node2->GetExtendInfo()->GetOutputSymbol(0U), netoutput_node->GetExtendInfo()->GetInputSymbol(0U));
  EXPECT_EQ(node3->GetExtendInfo()->GetOutputSymbol(0U), node4->GetExtendInfo()->GetInputSymbol(0U));
  EXPECT_EQ(node4->GetExtendInfo()->GetOutputSymbol(0U), node5->GetExtendInfo()->GetInputSymbol(0U));
  EXPECT_EQ(node6->GetExtendInfo()->GetOutputSymbol(0U), node5->GetExtendInfo()->GetInputSymbol(1U));
  EXPECT_EQ(node5->GetExtendInfo()->GetOutputSymbol(0U), netoutput_node->GetExtendInfo()->GetInputSymbol(1U));
  EXPECT_EQ(CalcSymbolSize(graph.get()).first, 6U);
}

TEST_F(EquivalentDataEdgesUT, OneOutMultipleIn) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("node1", "Const")->NODE("node2", "FindInferShapeFunc")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("node1", "Const")
              ->EDGE(0, 0)
              ->NODE("node4", "FindTilingFunc")
              ->NODE("node5", "TilingParse")
              ->EDGE(0, 1)
              ->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("node6", "Const")->EDGE(0, 1)->NODE("node5", "TilingParse"));
  };

  auto graph = ge::ToExecuteGraph(g1);
  graph->TopologicalSorting();

  EquivalentDataEdges eq;
  ASSERT_EQ(eq.UpdateEquivalentEdges(graph.get()), ge::GRAPH_SUCCESS);
  auto node1 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node1");
  auto node2 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node2");
  auto node4 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node4");
  auto node5 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node5");
  auto node6 = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "node6");
  auto netoutput_node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "NetOutput");

  CheckSameSymbol({node1->GetExtendInfo()->GetOutputSymbol(0U), node2->GetExtendInfo()->GetInputSymbol(0U),
                   node4->GetExtendInfo()->GetInputSymbol(0U)});
  CheckSameSymbol({node2->GetExtendInfo()->GetOutputSymbol(0U), netoutput_node->GetExtendInfo()->GetInputSymbol(0U)});
  CheckSameSymbol({node4->GetExtendInfo()->GetOutputSymbol(0U), node5->GetExtendInfo()->GetInputSymbol(0U)});
  CheckSameSymbol({node6->GetExtendInfo()->GetOutputSymbol(0U), node5->GetExtendInfo()->GetInputSymbol(1U)});
  CheckSameSymbol({node5->GetExtendInfo()->GetOutputSymbol(0U), netoutput_node->GetExtendInfo()->GetInputSymbol(1U)});
  EXPECT_EQ(CalcSymbolSize(graph.get()).first, 5U);
}

TEST_F(EquivalentDataEdgesUT, MultiGraphs) {
  DEF_GRAPH(Init) {
    CHAIN(NODE("node1", "Const")->NODE("node2", "FindInferShapeFunc")->NODE("InnerNetOutput", "InnerNetOutput"));
    CHAIN(NODE("node1", "Const")
              ->EDGE(0, 0)
              ->NODE("node4", "FindTilingFunc")
              ->NODE("node5", "TilingParse")
              ->EDGE(0, 1)
              ->NODE("InnerNetOutput", "InnerNetOutput"));
    CHAIN(NODE("node6", "Const")->EDGE(0, 1)->NODE("node5", "TilingParse"));
  };

  auto init_graph = ge::ToExecuteGraph(Init);
  DEF_GRAPH(Main) {
    CHAIN(NODE("idata1", "InnerData")->NODE("node1", "InferShape")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("idata2", "InnerData")->NODE("node2", "Tiling")->EDGE(0, 1)->NODE("NetOutput", "NetOutput"));
  };

  auto main_graph = ge::ToExecuteGraph(Main);
  ge::AttrUtils::SetInt(ge::ExecuteGraphUtils::FindNodeFromAllNodes(main_graph.get(), "idata1")->GetOpDescPtr(),
                        "index", 0);
  ge::AttrUtils::SetInt(ge::ExecuteGraphUtils::FindNodeFromAllNodes(main_graph.get(), "idata2")->GetOpDescPtr(),
                        "index", 1);

  DEF_GRAPH(rg) {
    CHAIN(NODE("Init", "Init")->EDGE(0, 0)->NODE("Main", "Main"));
    CHAIN(NODE("Init", "Init")->EDGE(1, 1)->NODE("Main", "Main"));
  };
  auto root_graph = ge::ToExecuteGraph(rg);
  root_graph->AddSubGraph(init_graph);
  root_graph->AddSubGraph(main_graph);
  init_graph->SetParentGraph(root_graph.get());
  init_graph->SetParentNode(ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), "Init"));
  ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), "Init")->GetOpDescPtr()->AddSubgraphName("Init");
  ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), "Init")
      ->GetOpDescPtr()
      ->SetSubgraphInstanceName(0, "Init");
  main_graph->SetParentGraph(root_graph.get());
  main_graph->SetParentNode(ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), "Main"));
  ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), "Main")->GetOpDescPtr()->AddSubgraphName("Main");
  ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), "Main")
      ->GetOpDescPtr()
      ->SetSubgraphInstanceName(0, "Main");

  root_graph->TopologicalSorting();
  EquivalentDataEdges eq;
  ASSERT_EQ(eq.UpdateEquivalentEdges(root_graph.get()), ge::GRAPH_SUCCESS);
  CheckSameSymbol(
      {ge::ExecuteGraphUtils::FindNodeFromAllNodes(init_graph.get(), "node2")->GetExtendInfo()->GetOutputSymbol(0U),
       ge::ExecuteGraphUtils::FindNodeFromAllNodes(init_graph.get(), "InnerNetOutput")
           ->GetExtendInfo()
           ->GetInputSymbol(0U),
       ge::ExecuteGraphUtils::FindNodeFromAllNodes(main_graph.get(), "idata1")->GetExtendInfo()->GetOutputSymbol(0U),
       ge::ExecuteGraphUtils::FindNodeFromAllNodes(main_graph.get(), "node1")->GetExtendInfo()->GetInputSymbol(0U),
       ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), "Init")->GetExtendInfo()->GetOutputSymbol(0U),
       ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), "Main")->GetExtendInfo()->GetInputSymbol(0U)});
}

/* TilingData和LaunchArgs不再使用RefFrom输出, 替换为图上连边的表达
TEST_F(EquivalentDataEdgesUT, RefOk) {
  // todo这里用了端到端用例构造ref场景，后面考虑换成手动构造，保证用例的稳定性
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                              .SetModelDescHolder(&model_desc_holder)
                              .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  exe_graph->TopologicalSorting();

  EquivalentDataEdges eq;
  ASSERT_EQ(eq.UpdateEquivalentEdges(exe_graph.get()), ge::GRAPH_SUCCESS);

  auto alloc_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "AllocLaunchArg");
  ASSERT_EQ(alloc_nodes.size(), 1);
  auto alloc_node = alloc_nodes[0];

  auto tiling_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "Tiling");
  ASSERT_EQ(tiling_nodes.size(), 1);
  auto tiling_node = tiling_nodes[0];

  auto launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "LaunchKernelWithHandle");
  ASSERT_EQ(launch_nodes.size(), 1);
  auto launch_node = launch_nodes[0];

  auto optimize_host_inputs_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "CopyFlowLaunch");
  ASSERT_EQ(optimize_host_inputs_nodes.size(), 1);
  auto optimize_host_inputs_node = optimize_host_inputs_nodes[0];

  EXPECT_EQ(alloc_node->GetOutEdgesSizeByIndex(static_cast<int32_t>(AllocLaunchArgOutputs::kTilingDataBase)), 0U);
  EXPECT_EQ(alloc_node->GetDataOutNum(), static_cast<size_t>(AllocLaunchArgOutputs::kNum));

  EXPECT_EQ(alloc_node->GetExtendInfo()->GetOutputSymbol(static_cast<size_t>(AllocLaunchArgOutputs::kTilingDataBase)),
            tiling_node->GetExtendInfo()->GetOutputSymbol(static_cast<size_t>(TilingContext::kOutputTilingData)));

  CheckSameSymbol({alloc_node->GetExtendInfo()->GetOutputSymbol(static_cast<size_t>(AllocLaunchArgOutputs::kRtArg)),
                   launch_node->GetExtendInfo()->GetOutputSymbol(0U),
                   optimize_host_inputs_node->GetExtendInfo()->GetOutputSymbol(0U)});
}
*/

TEST_F(EquivalentDataEdgesUT, InnerDataLinkToNetOutput) {
  bg::ValueHolder::PushGraphFrame();
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto holder1 = bg::If<bg::ValueHolder>(
      data0,
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto launch = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", {data0});
        return {data0, launch};
      },
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto launch = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {data0});
        return {data0, launch};
      });

  auto holder2 = bg::If<bg::ValueHolder>(
      data0,
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto launch = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithHandle", {data0});
        return {data0, launch};
      },
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto launch = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {data0});
        return {data0, launch};
      });
  auto add1 = bg::ValueHolder::CreateDataOutput("Add", holder1, 1);
  auto add2 = bg::ValueHolder::CreateDataOutput("Add", holder2, 1);
  std::vector<bg::ValueHolderPtr> inputs = add1;
  for (auto add : add2) {
    inputs.emplace_back(add);
  }
  auto out = bg::ValueHolder::CreateDataOutput("Add", inputs, 2);

  auto frame = bg::ValueHolder::PopGraphFrame(out, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();

  EquivalentDataEdges eq;
  ASSERT_EQ(eq.UpdateEquivalentEdges(graph.get()), ge::GRAPH_SUCCESS);

  // 最终图中的edge应该要存在symbol，对应到外部的Data节点的输出edge上
  auto data_out_symbol = data0->GetFastNode()->GetExtendInfo()->GetOutputSymbol(0U);
  const auto pair = CalcSymbolSize(graph.get(), data_out_symbol);
  ASSERT_EQ(pair.first, 13);
  ASSERT_EQ(pair.second, 23U);
}
}  // namespace gert
