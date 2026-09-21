/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/utils/inner_data_remover.h"
#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "graph_builder/bg_condition.h"
#include "common/ge_inner_attrs.h"
#include "graph/utils/graph_dump_utils.h"

namespace gert {
namespace bg {
class InnerDataRemoverForSubgraphsUT : public BgTestAutoCreateFrame {
 public:
  /*
   * |                               |
   * | InnerData InnerData InnerData |   Main Graph
   * +-------------------------------+
   *        \      |       /
   *       +-----------------+
   *       | InnerNetOutput  |
   *       |  |   |      |   |
   *       |  |   |      |   |  Init Graph
   *       |  |   c2 -> Foo  |
   *       |  |         |    |
   *       | c1 --------+    |
   *       +-----------------+
   */
  std::vector<ValueHolderPtr> InitMain(size_t init_out_count) {
    init_outputs = ValueHolder::CreateDataOutput("Init", {}, init_out_count);
    ValueHolder::PushGraphFrame(init_outputs[0], "Init");

    std::vector<ValueHolderPtr> cs;
    for (size_t i = 0U; i < init_out_count; ++i) {
      cs.push_back(ValueHolder::CreateConst("Hello", 5));
    }
    if (init_out_count > 1) {
      auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {cs[0], cs[1]});
      auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {cs[0], cs[1]});
      cs[0] = foo1;
      cs[1] = foo2;
    }
    ValueHolder::PopGraphFrame(cs, {});

    auto main_node = ValueHolder::CreateVoid<bg::ValueHolder>("Main", init_outputs);
    ValueHolder::PushGraphFrame(main_node, "Main");
    std::vector<ValueHolderPtr> inner_datas_on_main;
    for (size_t i = 0U; i < init_out_count; ++i) {
      inner_datas_on_main.emplace_back(CreateInnerData(static_cast<int32_t>(i)));
    }
    return inner_datas_on_main;
  }

  std::vector<ValueHolderPtr> InitSubgraph(int32_t init_out_count) {
    auto inner_datas_on_main = InitMain(init_out_count);
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", inner_datas_on_main);
    ValueHolder::PushGraphFrame(foo1, "Subgraph");
    std::vector<ValueHolderPtr> subgraph_inner_data;
    subgraph_inner_data.reserve(init_out_count);
    for (int32_t i = 0; i < init_out_count; ++i) {
      subgraph_inner_data.push_back(CreateInnerData(i));
    }
    return subgraph_inner_data;
  }

  std::vector<ValueHolderPtr> InitSubgraph(int32_t init_out_count, int32_t subgraph_in_count) {
    auto inner_datas_on_main = InitMain(init_out_count);
    auto foo1 = ValueHolder::CreateSingleDataOutput(
        "Foo", {inner_datas_on_main.begin(), inner_datas_on_main.begin() + subgraph_in_count});
    ValueHolder::PushGraphFrame(foo1, "Subgraph");
    std::vector<ValueHolderPtr> subgraph_inner_data;
    subgraph_inner_data.reserve(subgraph_in_count);
    for (int32_t i = 0; i < subgraph_in_count; ++i) {
      subgraph_inner_data.push_back(CreateInnerData(i));
    }
    return subgraph_inner_data;
  }
  std::vector<ValueHolderPtr> InitSubgraphAllUseInParent(int32_t init_out_count) {
    auto inner_datas_on_main = InitMain(init_out_count);
    auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", inner_datas_on_main);
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", inner_datas_on_main);
    ValueHolder::PushGraphFrame(foo1, "Subgraph");
    std::vector<ValueHolderPtr> subgraph_inner_data;
    subgraph_inner_data.reserve(init_out_count);
    for (int32_t i = 0; i < init_out_count; ++i) {
      subgraph_inner_data.push_back(CreateInnerData(i));
    }
    return subgraph_inner_data;
  }
  static ValueHolderPtr CreateInnerData(int32_t index) {
    auto data = ValueHolder::CreateSingleDataOutput("InnerData", {});
    GE_ASSERT_NOTNULL(data);
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(data->GetFastNode()->GetOpDescBarePtr(), "index", index));
    return data;
  }

  static void AllInnerDataIndexesContinuous(const ge::ExecuteGraph *graph) {
    std::set<int32_t> indexes;
    for (const auto &node : graph->GetDirectNode()) {
      if (!IsInnerDataType(node->GetTypePtr())) {
        continue;
      }
      int32_t index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "index", index));
      ASSERT_TRUE(indexes.emplace(index).second);
    }
    ASSERT_EQ(static_cast<size_t>(*indexes.rbegin()) + 1, indexes.size());
  }
  static void InnerDataNumMatch(const ge::ExecuteGraph *graph, size_t expect_num) {
    std::set<int32_t> indexes;
    for (const auto &node : graph->GetDirectNode()) {
      if (!IsInnerDataType(node->GetTypePtr())) {
        continue;
      }
      int32_t index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "index", index));
      ASSERT_TRUE(indexes.emplace(index).second);
    }
    ASSERT_EQ(indexes.size(), expect_num);
  }

  static void ConnectFromInitNode(int32_t src_index, const ge::FastNode *dst_node, int32_t dst_index) {
    auto top_in_data_edge = dst_node->GetInDataEdgeByIndex(dst_index);
    ASSERT_NE(top_in_data_edge, nullptr);
    while (true) {
      auto tmp_src_node = top_in_data_edge->src;
      ASSERT_NE(tmp_src_node, nullptr);
      if (tmp_src_node->GetType() != "InnerData") {
        break;
      }
      int32_t tmp_index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(tmp_src_node->GetOpDescBarePtr(), "index", tmp_index));

      auto tmp_owner_graph = tmp_src_node->GetExtendInfo()->GetOwnerGraphBarePtr();
      ASSERT_NE(tmp_owner_graph, nullptr);
      auto tmp_parent_node = tmp_owner_graph->GetParentNodeBarePtr();
      ASSERT_NE(tmp_parent_node, nullptr);

      top_in_data_edge = tmp_parent_node->GetInDataEdgeByIndex(tmp_index);
      ASSERT_NE(top_in_data_edge, nullptr)
          << "Failed to get in anchor from node " << tmp_parent_node->GetName() << ", index " << tmp_index;
    }

    auto top_src_node = top_in_data_edge->src;
    ASSERT_NE(top_src_node, nullptr);
    ASSERT_EQ(top_src_node->GetType(), "Init");
    ASSERT_EQ(top_in_data_edge->src_output, src_index);
  }

  // copy from BgConditionUT
  ge::ExecuteGraph *FindSubgraph(const std::vector<const ge::FastNode *> &nodes, const std::string &subgraph_name) {
    auto root_graph = ge::ExecuteGraphUtils::FindRootGraph(nodes[0]->GetExtendInfo()->GetOwnerGraphBarePtr());
    for (const auto node : nodes) {
      auto &names_to_index = node->GetOpDescBarePtr()->GetSubgraphNameIndexes();
      auto iter = names_to_index.find(subgraph_name);
      if (iter != names_to_index.end()) {
        return root_graph->GetSubGraph(node->GetOpDescBarePtr()->GetSubgraphInstanceName(iter->second));
      }
    }
    return nullptr;
  }

  std::vector<ge::FastNode *> FindNodesByType(const ge::ExecuteGraph *graph, const std::string &type) {
    std::vector<ge::FastNode *> nodes;
    for (const auto node : graph->GetDirectNode()) {
      if (node->GetType() == type) {
        nodes.push_back(node);
      }
    }
    return nodes;
  }

  void CheckCondGraph(const ge::FastNode *if_node, size_t branch_count) {
    ASSERT_EQ(if_node->GetOpDescBarePtr()->GetSubgraphNameIndexes().size(), branch_count + 1);
    auto cond_graph = FindSubgraph({if_node}, ge::kConditionGraph);
    ASSERT_NE(cond_graph, nullptr);
    ASSERT_EQ(ExeGraphSummaryChecker(cond_graph)
                  .StrictDirectNodeTypes({{"InnerData", 1},
                                          {"SwitchNotify", 1},
                                          {"WatcherPlaceholder", 1},
                                          {"BranchPivot", branch_count},
                                          {"BranchDone", branch_count},
                                          {"WaitAnyone", 1}}),
              "success");
    auto data = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "InnerData");
    ASSERT_EQ(FastNodeTopoChecker(data).StrictConnectTo(0, {{"SwitchNotify"}}), "success");

    // check count of BranchPivot and BranchDone
    auto p_nodes = FindNodesByType(cond_graph, "BranchPivot");
    auto d_nodes = FindNodesByType(cond_graph, "BranchDone");
    ASSERT_EQ(p_nodes.size(), branch_count);
    ASSERT_EQ(d_nodes.size(), branch_count);

    // check Attr "branch" for nodes BranchPivot
    std::set<int64_t> expected_indexes;
    for (size_t i = 0; i < branch_count; ++i) {
      expected_indexes.insert(static_cast<int64_t>(i) + 1);
    }
    std::set<int64_t> indexes;
    for (const auto &p_node : p_nodes) {
      int64_t branch_index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(p_node->GetOpDescBarePtr(), ge::kRelativeBranch, branch_index));
      ASSERT_TRUE(indexes.insert(branch_index).second) << "Duplicated branch index " << branch_index;
    }
    ASSERT_EQ(indexes, expected_indexes);

    // check Attr "branch" for nodes BranchDone
    indexes.clear();
    for (const auto &d_node : d_nodes) {
      int64_t branch_index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(d_node->GetOpDescBarePtr(), ge::kRelativeBranch, branch_index));
      ASSERT_TRUE(indexes.insert(branch_index).second) << "Duplicated branch index " << branch_index;
    }
    ASSERT_EQ(indexes, expected_indexes);

    // check edges
    auto sn = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "SwitchNotify");
    ASSERT_EQ(FastNodeTopoChecker(sn).StrictConnectFrom({{"InnerData"}}), "success");
    ASSERT_EQ(FastNodeTopoChecker(sn).StrictConnectTo(0, {{"WatcherPlaceholder"}}), "success");
    for (size_t i = 0; i < branch_count; ++i) {
      ASSERT_EQ(FastNodeTopoChecker(sn).StrictConnectTo(i + 1, {{"BranchPivot"}}), "success");
      ASSERT_TRUE(FastNodeTopoChecker(sn)
                      .OutChecker()
                      .DataTo(p_nodes[i]->GetName())
                      .CtrlToByType("BranchDone")
                      .CtrlToByType("WaitAnyone")
                      .IsOk());
    }
  }

  std::vector<ValueHolderPtr> init_outputs;
};

TEST_F(InnerDataRemoverForSubgraphsUT, RemoveSucessAndIndexCorrect_OneGraphRemoveLast) {
  auto inner_datas_on_main = InitMain(4);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {inner_datas_on_main[0], inner_datas_on_main[1]});
  ASSERT_NE(foo1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({foo1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  std::vector<ge::ExecuteGraph *> graphs = {graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get()).StrictAllNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(graph.get());
  auto owner_graph = foo1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  size_t in_data_edge_size = owner_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize();
  ASSERT_EQ(in_data_edge_size, 2U);
  ConnectFromInitNode(0, foo1->GetFastNode(), 0);
  ConnectFromInitNode(1, foo1->GetFastNode(), 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT, RemoveSucessAndIndexCorrect_OneGraphRemoveFirst) {
  auto inner_datas_on_main = InitMain(4);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {inner_datas_on_main[2], inner_datas_on_main[3]});
  ASSERT_NE(foo1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({foo1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  std::vector<ge::ExecuteGraph *> graphs = {graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get()).StrictAllNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(graph.get());
  auto owner_graph = foo1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  size_t in_data_edge_size = owner_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize();
  ASSERT_EQ(in_data_edge_size, 2U);
  ConnectFromInitNode(2, foo1->GetFastNode(), 0);
  ConnectFromInitNode(3, foo1->GetFastNode(), 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT, RemoveSucessAndIndexCorrect_OneGraphRemoveMid) {
  auto inner_datas_on_main = InitMain(4);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {inner_datas_on_main[0], inner_datas_on_main[3]});
  ASSERT_NE(foo1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({foo1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  std::vector<ge::ExecuteGraph *> graphs = {graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get()).StrictAllNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(graph.get());
  auto owner_graph = foo1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  size_t in_data_edge_size = owner_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize();
  ASSERT_EQ(in_data_edge_size, 2U);
  ConnectFromInitNode(0, foo1->GetFastNode(), 0);
  ConnectFromInitNode(3, foo1->GetFastNode(), 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT, RemoveSucessAndIndexCorrect_RemoveFirstMidLast) {
  auto inner_datas_on_main = InitMain(7);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {inner_datas_on_main[2], inner_datas_on_main[4]});
  ASSERT_NE(foo1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({foo1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  std::vector<ge::ExecuteGraph *> graphs = {graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get()).StrictAllNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(graph.get());
  auto owner_graph = foo1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  size_t in_data_edge_size = owner_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize();
  ASSERT_EQ(in_data_edge_size, 2U);
  ConnectFromInitNode(2, foo1->GetFastNode(), 0);
  ConnectFromInitNode(4, foo1->GetFastNode(), 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT, RemoveSucessAndIndexCorrect_UnusedInnerDataInSubgraphAtLast) {
  auto inputs = InitSubgraph(4);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {inputs[0], inputs[1]});
  ASSERT_NE(bar1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({bar1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  std::vector<ge::ExecuteGraph *> graphs = {graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get()).StrictAllNodeTypes({{"InnerData", 2}, {"Bar", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(graph.get());
  auto owner_graph = bar1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  size_t in_data_edge_size = owner_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize();
  ASSERT_EQ(in_data_edge_size, 2U);
  ConnectFromInitNode(0, bar1->GetFastNode(), 0);
  ConnectFromInitNode(1, bar1->GetFastNode(), 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT, RemoveSucessAndIndexCorrect_UnusedInnerDataInSubgraphAtBeginMidAndEnd) {
  auto inputs = InitSubgraph(7);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {inputs[2], inputs[4]});
  ASSERT_NE(bar1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({bar1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  std::vector<ge::ExecuteGraph *> graphs = {graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get()).StrictAllNodeTypes({{"InnerData", 2}, {"Bar", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(graph.get());
  auto owner_graph = bar1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  size_t in_data_edge_size = owner_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize();
  ASSERT_EQ(in_data_edge_size, 2U);
  ConnectFromInitNode(2, bar1->GetFastNode(), 0);
  ConnectFromInitNode(4, bar1->GetFastNode(), 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT, UnusedInnerDataParentRemoved_UnusedInnerDataInSubgraphAtBeginMidAndEnd) {
  auto inputs = InitSubgraph(7);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {inputs[2], inputs[4]});
  ASSERT_NE(bar1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({bar1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto parent_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(parent_frame, nullptr);
  auto parent_graph = parent_frame->GetExecuteGraph();
  ASSERT_NE(parent_graph, nullptr);
  std::vector<ge::ExecuteGraph *> graphs = {graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ExeGraphSummaryChecker(parent_graph.get()).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}}),
            "success");
  AllInnerDataIndexesContinuous(parent_graph.get());

  ASSERT_EQ(parent_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize(), 2);
  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(parent_graph.get(), "Foo");
  ASSERT_NE(foo1_node, nullptr);
  ConnectFromInitNode(2, foo1_node, 0);
  ConnectFromInitNode(4, foo1_node, 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT,
       RemoveSucessAndIndexCorrect_UnusedInnerDataInSubgraphAtBeginMidAndEndInputsParentAndSubgraphs1) {
  auto inputs = InitSubgraph(9, 7);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {inputs[2], inputs[4]});
  ASSERT_NE(bar1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({bar1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto parent_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(parent_frame, nullptr);
  auto parent_graph = parent_frame->GetExecuteGraph();
  ASSERT_NE(parent_graph, nullptr);
  std::vector<ge::ExecuteGraph *> graphs = {graph.get(), parent_graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get()).StrictAllNodeTypes({{"InnerData", 2}, {"Bar", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(graph.get());
  auto owner_graph = bar1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  size_t in_data_edge_size = owner_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize();
  ASSERT_EQ(in_data_edge_size, 2U);
  ConnectFromInitNode(2, bar1->GetFastNode(), 0);
  ConnectFromInitNode(4, bar1->GetFastNode(), 1);

  ASSERT_EQ(ExeGraphSummaryChecker(parent_graph.get()).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}}),
            "success");
  AllInnerDataIndexesContinuous(parent_graph.get());
  ASSERT_EQ(parent_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize(), 2U);
  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(parent_graph.get(), "Foo");
  ASSERT_NE(foo1_node, nullptr);
  ConnectFromInitNode(2, foo1_node, 0);
  ConnectFromInitNode(4, foo1_node, 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT,
       RemoveSucessAndIndexCorrect_UnusedInnerDataInSubgraphAtBeginMidAndEndInputsParentAndSubgraphs2) {
  auto inputs = InitSubgraph(9, 7);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {inputs[2], inputs[4]});
  ASSERT_NE(bar1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({bar1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto parent_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(parent_frame, nullptr);
  auto parent_graph = parent_frame->GetExecuteGraph();
  ASSERT_NE(parent_graph, nullptr);
  std::vector<ge::ExecuteGraph *> graphs = {parent_graph.get(), graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get()).StrictAllNodeTypes({{"InnerData", 2}, {"Bar", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(graph.get());
  auto owner_graph = bar1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  size_t in_data_edge_size = owner_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize();
  ASSERT_EQ(in_data_edge_size, 2U);
  ConnectFromInitNode(2, bar1->GetFastNode(), 0);
  ConnectFromInitNode(4, bar1->GetFastNode(), 1);

  ASSERT_EQ(ExeGraphSummaryChecker(parent_graph.get()).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}}),
            "success");
  AllInnerDataIndexesContinuous(parent_graph.get());
  ASSERT_EQ(parent_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize(), 2U);
  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(parent_graph.get(), "Foo");
  ASSERT_NE(foo1_node, nullptr);
  ConnectFromInitNode(2, foo1_node, 0);
  ConnectFromInitNode(4, foo1_node, 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT, ParentInnerDataRemains_MultipleUseOnParent) {
  auto inputs = InitSubgraphAllUseInParent(7);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {inputs[2], inputs[4]});
  ASSERT_NE(bar1, nullptr);
  auto frame = ValueHolder::PopGraphFrame({bar1}, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto parent_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(parent_frame, nullptr);
  auto parent_graph = parent_frame->GetExecuteGraph();
  ASSERT_NE(parent_graph, nullptr);

  // check subgraph is ok
  std::vector<ge::ExecuteGraph *> graphs = {graph.get()};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get()).StrictAllNodeTypes({{"InnerData", 2}, {"Bar", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(graph.get());
  auto owner_graph = bar1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  size_t in_data_edge_size = owner_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize();
  ASSERT_EQ(in_data_edge_size, 2U);
  ConnectFromInitNode(2, bar1->GetFastNode(), 0);
  ConnectFromInitNode(4, bar1->GetFastNode(), 1);

  ASSERT_EQ(
      ExeGraphSummaryChecker(parent_graph.get()).StrictDirectNodeTypes({{"InnerData", 7}, {"Foo2", 1}, {"Foo", 1}}),
      "success");
  AllInnerDataIndexesContinuous(parent_graph.get());
  ASSERT_EQ(parent_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize(), 7U);
  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(parent_graph.get(), "Foo");
  ASSERT_NE(foo1_node, nullptr);
  ConnectFromInitNode(2, foo1_node, 0);
  ConnectFromInitNode(4, foo1_node, 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT, RemoveSucessAndIndexCorrect_RemoveInSingleBranchCondition) {
  auto inputs = InitMain(7);
  auto if_holder = bg::If<ValueHolder>(
      inputs[2],
      [&]() -> std::vector<ValueHolderPtr> {
        ValueHolder::CreateSingleDataOutput("Foo", {inputs[2], inputs[4]});
        ValueHolder::CreateSingleDataOutput("Bar", inputs);
        return {};
      },
      nullptr);

  auto parent_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(parent_frame, nullptr);
  auto parent_graph = parent_frame->GetExecuteGraph();
  ASSERT_NE(parent_graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(parent_graph.get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  auto foo_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Foo");
  ASSERT_NE(foo_node, nullptr);
  auto bar_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Bar");
  ASSERT_NE(bar_node, nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::IsolateNode(bar_node, {}), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(then_graph, bar_node), ge::GRAPH_SUCCESS);
  std::vector<ge::ExecuteGraph *> graphs = {then_graph};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  CheckCondGraph(if_node, 2);

  // subgraphs in if
  ASSERT_EQ(
      ExeGraphSummaryChecker(then_graph).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  AllInnerDataIndexesContinuous(then_graph);
  ASSERT_EQ(foo_node->GetAllInDataEdgesSize(), 2U);
  ConnectFromInitNode(2, foo_node, 0);
  ConnectFromInitNode(4, foo_node, 1);

  // parent graph
  ASSERT_EQ(ExeGraphSummaryChecker(parent_graph.get()).StrictDirectNodeTypes({{"InnerData", 2}, {"If", 1}}), "success");
  AllInnerDataIndexesContinuous(parent_graph.get());
  ASSERT_EQ(parent_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize(), 2U);
  ConnectFromInitNode(2, if_node, 0);
  ConnectFromInitNode(4, if_node, 1);
}
TEST_F(InnerDataRemoverForSubgraphsUT, Remians_OtherBranchUses1) {
  auto inputs = InitMain(9);
  auto if_holder = bg::If<bg::ValueHolder>(
      inputs[2],
      [&]() -> std::vector<ValueHolderPtr> {
        ValueHolder::CreateSingleDataOutput("Bar", inputs);
        ValueHolder::CreateSingleDataOutput("Foo", {inputs[2], inputs[6]});
        return {};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        ValueHolder::CreateSingleDataOutput("Bar", inputs);
        ValueHolder::CreateSingleDataOutput("Foo", {inputs[3], inputs[8]});
        return {};
      });

  auto parent_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(parent_frame, nullptr);
  auto parent_graph = parent_frame->GetExecuteGraph();
  ASSERT_NE(parent_graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(parent_graph.get(), "If");
  ASSERT_NE(if_node, nullptr);

  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  auto then_foo = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Foo");
  ASSERT_NE(then_foo, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  auto else_foo = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "Foo");
  ASSERT_NE(else_foo, nullptr);

  auto bar_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Bar");
  ASSERT_NE(bar_node, nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::IsolateNode(bar_node, {}), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(then_graph, bar_node), ge::GRAPH_SUCCESS);
  bar_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "Bar");
  ASSERT_NE(bar_node, nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::IsolateNode(bar_node, {}), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(else_graph, bar_node), ge::GRAPH_SUCCESS);
  std::vector<ge::ExecuteGraph *> graphs = {then_graph, else_graph};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  CheckCondGraph(if_node, 2);

  // then graph in if
  ASSERT_EQ(
      ExeGraphSummaryChecker(then_graph).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  InnerDataNumMatch(then_graph, 2);
  ASSERT_EQ(then_foo->GetAllInDataEdgesSize(), 2);
  ConnectFromInitNode(2, then_foo, 0);
  ConnectFromInitNode(6, then_foo, 1);

  // else graph in if
  ASSERT_EQ(
      ExeGraphSummaryChecker(else_graph).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  InnerDataNumMatch(then_graph, 2);
  ASSERT_EQ(else_foo->GetAllInDataEdgesSize(), 2U);
  ConnectFromInitNode(3, else_foo, 0);
  ConnectFromInitNode(8, else_foo, 1);

  // parent graph
  ASSERT_EQ(ExeGraphSummaryChecker(parent_graph.get()).StrictDirectNodeTypes({{"InnerData", 4}, {"If", 1}}), "success");
  InnerDataNumMatch(parent_graph.get(), 4);
  ASSERT_EQ(parent_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize(), 4U);
}
TEST_F(InnerDataRemoverForSubgraphsUT, Remians_OtherBranchUses2) {
  auto inputs = InitMain(9);
  auto if_holder = bg::If<bg::ValueHolder>(
      inputs[2],
      [&]() -> std::vector<ValueHolderPtr> {
        ValueHolder::CreateSingleDataOutput("Bar", inputs);
        ValueHolder::CreateSingleDataOutput("Foo", {inputs[2], inputs[6]});
        return {};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        ValueHolder::CreateSingleDataOutput("Foo", {inputs[3], inputs[8]});
        return {};
      });

  auto parent_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(parent_frame, nullptr);
  auto parent_graph = parent_frame->GetExecuteGraph();
  ASSERT_NE(parent_graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(parent_graph.get(), "If");
  ASSERT_NE(if_node, nullptr);
  ASSERT_EQ(if_node->GetAllInDataEdgesSize(), 9U);

  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  auto then_foo = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Foo");
  ASSERT_NE(then_foo, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  auto else_foo = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "Foo");
  ASSERT_NE(else_foo, nullptr);

  auto bar_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Bar");
  ASSERT_NE(bar_node, nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::IsolateNode(bar_node, {}), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(then_graph, bar_node), ge::GRAPH_SUCCESS);

  // GE_DUMP(parent_graph->GetParentGraphBarePtr(), "Test1");
  ge::DumpGraph(parent_graph->GetParentGraphBarePtr(), "Test1");

  InnerDataRemoverForSubgraphs remover(std::vector<ge::ExecuteGraph *>{then_graph, else_graph});
  bool changed = false;
  ASSERT_EQ(remover.RemoveUnusedInnerDataNodes(changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  CheckCondGraph(if_node, 2);

  // then graph in if
  ASSERT_EQ(
      ExeGraphSummaryChecker(then_graph).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  InnerDataNumMatch(then_graph, 2);
  ASSERT_EQ(then_foo->GetAllInDataEdgesSize(), 2U);
  ConnectFromInitNode(2, then_foo, 0);
  ConnectFromInitNode(6, then_foo, 1);

  // else graph in if
  ASSERT_EQ(
      ExeGraphSummaryChecker(else_graph).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  InnerDataNumMatch(then_graph, 2);
  ASSERT_EQ(else_foo->GetAllInDataEdgesSize(), 2U);
  ConnectFromInitNode(3, else_foo, 0);
  ConnectFromInitNode(8, else_foo, 1);

  // parent graph
  ASSERT_EQ(ExeGraphSummaryChecker(parent_graph.get()).StrictDirectNodeTypes({{"InnerData", 4}, {"If", 1}}), "success");
  InnerDataNumMatch(parent_graph.get(), 4);
  ASSERT_EQ(parent_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize(), 4U);

  changed = false;
  ASSERT_EQ(remover.RemoveUnusedInnerDataNodes(changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
TEST_F(InnerDataRemoverForSubgraphsUT, Success_IfInCase) {
  auto inputs = InitMain(9);
  auto case_holder = bg::Case<bg::ValueHolder>(
      inputs[1], {
                     [&]() -> std::vector<ValueHolderPtr> {
                       return bg::If<bg::ValueHolder>(
                           inputs[2],
                           [&]() -> std::vector<ValueHolderPtr> {
                             ValueHolder::CreateSingleDataOutput("Bar", inputs);
                             auto foo = ValueHolder::CreateSingleDataOutput("Foo", {inputs[2], inputs[6]});
                             return {foo};
                           },
                           [&]() -> std::vector<ValueHolderPtr> {
                             ValueHolder::CreateSingleDataOutput("Bar", inputs);
                             auto foo = ValueHolder::CreateSingleDataOutput("Foo", {inputs[3], inputs[8]});
                             return {foo};
                           });
                     },
                     [&]() -> std::vector<ValueHolderPtr> {
                       return {ValueHolder::CreateSingleDataOutput("Foo", {inputs[2], inputs[6]})};
                     },
                     [&]() -> std::vector<ValueHolderPtr> {
                       return {ValueHolder::CreateSingleDataOutput("Foo", {inputs[3], inputs[4]})};
                     },
                 });

  auto parent_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(parent_frame, nullptr);
  auto parent_graph = parent_frame->GetExecuteGraph();
  ASSERT_NE(parent_graph, nullptr);

  auto case_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(parent_graph.get(), "Case");
  ASSERT_NE(case_node, nullptr);
  auto branch0_graph = ge::FastNodeUtils::GetSubgraphFromNode(case_node, 1);
  ASSERT_NE(branch0_graph, nullptr);
  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(branch0_graph, "If");
  ASSERT_NE(if_node, nullptr);

  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  auto then_foo = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Foo");
  ASSERT_NE(then_foo, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  auto else_foo = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "Foo");
  ASSERT_NE(else_foo, nullptr);

  auto bar_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Bar");
  ASSERT_NE(bar_node, nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::IsolateNode(bar_node, {}), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(then_graph, bar_node), ge::GRAPH_SUCCESS);
  bar_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "Bar");
  ASSERT_NE(bar_node, nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::IsolateNode(bar_node, {}), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(else_graph, bar_node), ge::GRAPH_SUCCESS);
  std::vector<ge::ExecuteGraph *> graphs = {then_graph, else_graph};
  ASSERT_EQ(InnerDataRemoverForSubgraphs(graphs).RemoveUnusedInnerDataNodes(), ge::GRAPH_SUCCESS);
  CheckCondGraph(case_node, 3);
  CheckCondGraph(if_node, 2);

  // then graph in if
  ASSERT_EQ(
      ExeGraphSummaryChecker(then_graph).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  InnerDataNumMatch(then_graph, 2);
  ASSERT_EQ(then_foo->GetAllInDataEdgesSize(), 2U);
  ConnectFromInitNode(2, then_foo, 0);
  ConnectFromInitNode(6, then_foo, 1);

  // else graph in if
  ASSERT_EQ(
      ExeGraphSummaryChecker(else_graph).StrictDirectNodeTypes({{"InnerData", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  InnerDataNumMatch(then_graph, 2);
  ASSERT_EQ(else_foo->GetAllInDataEdgesSize(), 2);
  ConnectFromInitNode(3, else_foo, 0);
  ConnectFromInitNode(8, else_foo, 1);

  // parent graph
  ASSERT_EQ(ExeGraphSummaryChecker(parent_graph.get()).StrictDirectNodeTypes({{"InnerData", 6}, {"Case", 1}}),
            "success");
  InnerDataNumMatch(parent_graph.get(), 6);
  ASSERT_EQ(parent_graph->GetParentNodeBarePtr()->GetAllInDataEdgesSize(), 6U);
  ConnectFromInitNode(1, case_node, 0);
}

// todo 当前while必须由lowering生成，无法自定义的while子图，因此暂时不覆盖while用例
// TEST_F(InnerDataRemoverForSubgraphsUT, RemoveSucessAndIndexCorrect_InWhileNode) {}
// TEST_F(InnerDataRemoverForSubgraphsUT, DoNotRemove_WhileNodeVariantInput) {}
}  // namespace bg
}  // namespace gert
