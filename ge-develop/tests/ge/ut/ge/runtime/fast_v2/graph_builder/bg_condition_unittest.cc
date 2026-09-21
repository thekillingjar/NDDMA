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
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "graph_builder/bg_condition.h"
#include "common/bg_test.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "common/ge_inner_attrs.h"
#include "utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/execute_graph_adapter.h"
#include "graph/fast_graph/execute_graph.h"
#include "graph/utils/execute_graph_utils.h"

namespace gert {
namespace bg {
class ConditionUT : public BgTestAutoCreateFrame {
 public:
  ge::ExecuteGraph *FindSubgraph(const std::vector<ge::FastNode *> &nodes, const std::string &subgraph_name) {
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
  ge::FastNode *FindData(const ge::ExecuteGraph *graph, int32_t index) {
    for (const auto node : graph->GetDirectNode()) {
      if (node->GetType() != "Data") {
        continue;
      }
      int32_t data_index;
      if (!ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "index", data_index)) {
        continue;
      }
      if (data_index != index) {
        continue;
      }
      return node;
    }
    return nullptr;
  }
  void ConnectFromOuter(ge::FastNode *node, int32_t dst_index, const ge::FastNode *outer, int32_t src_index) {
    while (true) {
        auto in_edge = node->GetInDataEdgeByIndex(dst_index);
        ASSERT_NE(in_edge, nullptr);
        auto src_node = in_edge->src;
        ASSERT_NE(src_node, nullptr);
        if (src_node == outer) {
          return;
        }
        if (src_node->GetType() == "InnerData" || src_node->GetType() == "Data") {
        int32_t parent_index;
        ASSERT_TRUE(ge::AttrUtils::GetInt(src_node->GetOpDescBarePtr(), "index", parent_index));
        auto parent_graph = src_node->GetExtendInfo()->GetOwnerGraphBarePtr();
        ASSERT_NE(parent_graph, nullptr);
        auto parent_node = parent_graph->GetParentNodeBarePtr();
        ASSERT_NE(parent_node, nullptr);
        node = const_cast<ge::FastNode *>(parent_node);
        dst_index = parent_index;
      }
    }
  }

  void CheckCondGraph(ge::FastNode *if_node, size_t branch_count) {
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
    auto p_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(cond_graph, "BranchPivot");
    auto d_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(cond_graph, "BranchDone");
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

  void NodeOutHasGuarder(const ge::FastNode *if_node, int32_t out_index, bool has_guarder = true) {
    bool found_guarder = false;
    for (const auto edge : if_node->GetOutEdgesRefByIndex(out_index)) {
      if (edge == nullptr) {
        continue;
      }
      auto dst_node = edge->dst;
      ASSERT_NE(dst_node, nullptr);
      int32_t index = -1;
      if (!ge::AttrUtils::GetInt(dst_node->GetOpDescBarePtr(), kReleaseResourceIndex, index)) {
        continue;
      }
      auto in_edge = dst_node->GetInDataEdgeByIndex(index);
      ASSERT_NE(in_edge, nullptr);
      if (in_edge->src == edge->src && in_edge->src_output == edge->src_output) {
        found_guarder = true;
        break;
      }
    }
    ASSERT_EQ(found_guarder, has_guarder)
        << "The cond node output index " << out_index << " guarded status check failed";
  }
};

TEST_F(ConditionUT, If_Failed_CondIsNullptr) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(
      nullptr,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
        return {foo1, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data1});
        return {data0, foo2};
      });

  ASSERT_EQ(if_holder.size(), 0);
}
TEST_F(ConditionUT, If_Failed_ThenAndElseAreNullptr) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(data0, nullptr, nullptr);

  ASSERT_EQ(if_holder.size(), 0);
}
TEST_F(ConditionUT, If_OutputSizeSameWithBranch) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
        return {foo1, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data1});
        return {data0, foo2};
      });

  ASSERT_EQ(if_holder.size(), 2);
}

TEST_F(ConditionUT, If_IoConnectionCorrect) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {  // then branch
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
        return {foo1, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {  // else branch
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data1});
        return {data0, foo2};
      });

  auto frame = ValueHolder::PopGraphFrame(if_holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(if_node).StrictConnectFrom({data0, data1}), "success");
  ASSERT_EQ(FastNodeTopoChecker(if_node).StrictConnectTo(0, {{"NetOutput", 0}}), "success");
  ASSERT_EQ(FastNodeTopoChecker(if_node).StrictConnectTo(1, {{"NetOutput", 1}}), "success");
}

TEST_F(ConditionUT, If_CondGraphCorrect) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {data0, data1});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
        auto foo3 = ValueHolder::CreateSingleDataOutput("Foo1", {bar1, foo2});
        return {foo3, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data2});
        return {data0, foo2};
      });

  auto frame = ValueHolder::PopGraphFrame(if_holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  CheckCondGraph(if_node, 2);
  auto sn = ge::ExecuteGraphUtils::FindFirstNodeMatchType(FindSubgraph({if_node}, ge::kConditionGraph), "SwitchNotify");
  ConnectFromOuter(sn, 0, data0->GetFastNode(), 0);
  ASSERT_EQ(FastNodeTopoChecker(sn).StrictConnectTo(0, {{"WatcherPlaceholder"}}), "success");
}
TEST_F(ConditionUT, If_ThenElseGraphCorrect) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {data0, data1});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
        auto foo3 = ValueHolder::CreateSingleDataOutput("Foo1", {bar1, foo2});
        return {foo3, bar1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data2});
        return {foo2, foo2};
      });

  auto frame = ValueHolder::PopGraphFrame(if_holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);

  auto then_graph = FindSubgraph({if_node}, ge::kThenGraph);
  ASSERT_NE(then_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(then_graph)
                .StrictAllNodeTypes({{"InnerData", 3}, {"Foo1", 2}, {"Bar1", 1}, {"InnerNetOutput", 1}}),
            "success");
  auto netoutput_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput");
  ASSERT_NE(netoutput_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput_node).StrictConnectFrom({{"Foo1"}, {"Bar1"}}), "success");
  ASSERT_TRUE(FastNodeTopoChecker(netoutput_node)
                  .InChecker()
                  .DataFromByType("Foo1")
                  .DataFromByType("Bar1")
                  .DataFromByType("InnerData")
                  .IsOk());

  auto else_graph = FindSubgraph({if_node}, ge::kElseGraph);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph).StrictAllNodeTypes({{"InnerData", 2}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");
  netoutput_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput");
  ASSERT_NE(netoutput_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput_node).StrictConnectFrom({{"Foo2"}, {"Foo2"}}), "success");
  auto foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "Foo2");
  ConnectFromOuter(foo2_node, 0, FindData(graph, 0), 0);
  ConnectFromOuter(foo2_node, 1, FindData(graph, 2), 0);
}

TEST_F(ConditionUT, If_ReturnHolder_NoDataOutputs) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0});
        auto foo2_holder = ValueHolder::CreateVoid<bg::ValueHolder>("Foo2", {foo1, data2});
        return {};
      },
      nullptr);
  ASSERT_EQ(if_holder.size(), 1);  // ctrl holder in it
}
TEST_F(ConditionUT, If_Success_OnlyThenBranch) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0});
        auto foo2_holder = ValueHolder::CreateVoid<bg::ValueHolder>("Foo2", {foo1, data2});
        return {};
      },
      nullptr);
  ASSERT_EQ(if_holder.size(), 1);

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(graph).StrictDirectNodeTypes({{"Data", 3}, {"If", 1}}), "success");
  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);
  CheckCondGraph(if_node, 2);

  auto then_graph = FindSubgraph({if_node}, ge::kThenGraph);
  ASSERT_NE(then_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(then_graph)
                .StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");

  auto else_graph = FindSubgraph({if_node}, ge::kElseGraph);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph).StrictAllNodeTypes({{"InnerNetOutput", 1}}), "success");
}
TEST_F(ConditionUT, If_Success_OnlyElseBranch) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(data0, nullptr, [&]() -> std::vector<ValueHolderPtr> {
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0});
    auto foo2_holder = ValueHolder::CreateVoid<bg::ValueHolder>("Foo2", {foo1, data2});
    return {};
  });
  ASSERT_EQ(if_holder.size(), 1);

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(graph).StrictDirectNodeTypes({{"Data", 3}, {"If", 1}}), "success");
  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);
  CheckCondGraph(if_node, 2);

  auto then_graph = FindSubgraph({if_node}, ge::kElseGraph);
  ASSERT_NE(then_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(then_graph)
                .StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");

  auto else_graph = FindSubgraph({if_node}, ge::kThenGraph);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph).StrictAllNodeTypes({{"InnerNetOutput", 1}}), "success");
}
TEST_F(ConditionUT, If_FailedForOuputsNumNotMatch_OnlyOneBranchWithDataOut) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(data0, nullptr, [&]() -> std::vector<ValueHolderPtr> {
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0});
    auto foo2_holder = ValueHolder::CreateVoid<bg::ValueHolder>("Foo2", {foo1, data2});
    return {foo1};
  });

  ASSERT_TRUE(if_holder.empty());
}
TEST_F(ConditionUT, If_Failed_BranchOutputSizeDifferent) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto if_holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
        return {foo1, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data1});
        return {data0};
      });

  ASSERT_EQ(if_holder.size(), 0);
}
TEST_F(ConditionUT, Case_Failed_IndexIsNullptr) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto holder = Case<ValueHolder>(nullptr, {[&]() -> std::vector<ValueHolderPtr> {
                                 auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
                                 return {foo1, data1};
                               },
                               [&]() -> std::vector<ValueHolderPtr> {
                                 auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data1});
                                 return {data0, foo2};
                               }});

  ASSERT_EQ(holder.size(), 0);
}
TEST_F(ConditionUT, Case_Failed_SubgraphBuilderIsNullptr) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto holder = Case<ValueHolder>(data0, {[&]() -> std::vector<ValueHolderPtr> {
                               auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
                               return {foo1, data1};
                             },
                             nullptr,
                             [&]() -> std::vector<ValueHolderPtr> {
                               auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data1});
                               return {data0, foo2};
                             }});

  ASSERT_EQ(holder.size(), 0);
}
TEST_F(ConditionUT, Case_CondGraphCorrect_ThreeBranch) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  auto case_holder = Case<ValueHolder>(data0, {[&]() -> std::vector<ValueHolderPtr> {
                                    auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {data0, data1});
                                    auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
                                    auto foo3 = ValueHolder::CreateSingleDataOutput("Foo1", {bar1, foo2});
                                    return {foo3, data1};
                                  },
                                  [&]() -> std::vector<ValueHolderPtr> {
                                    auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data0, data2});
                                    return {data0, foo2};
                                  },
                                  [&]() -> std::vector<ValueHolderPtr> {
                                    auto foo3 = ValueHolder::CreateSingleDataOutput("Foo3", {data0, data2});
                                    return {data0, foo3};
                                  }});

  auto frame = ValueHolder::PopGraphFrame(case_holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto case_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Case");
  ASSERT_NE(case_node, nullptr);
  CheckCondGraph(case_node, 3);
  auto cond_graph = FindSubgraph({case_node}, ge::kConditionGraph);
  auto sn = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "SwitchNotify");
  ConnectFromOuter(sn, 0, data0->GetFastNode(), 0);
}
TEST_F(ConditionUT, If_MoveGuarderOut_AllBranchsHaveGuarders) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  ge::FastNode *foo2_node, *foo3_node, *foo4_node;
  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {data0, data1});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
        foo2_node = foo2->GetFastNode();
        ValueHolder::CreateVoidGuarder("FreeMemory", foo2, {});
        auto foo3 = ValueHolder::CreateSingleDataOutput("Foo1", {bar1, foo2});
        ValueHolder::CreateVoidGuarder("FreeMemory", foo3, {});
        foo3_node = foo3->GetFastNode();
        return {foo3, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo4 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data2});
        ValueHolder::CreateVoidGuarder("FreeMemory", foo4, {});
        foo4_node = foo4->GetFastNode();
        return {foo4, data1};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);
  NodeOutHasGuarder(if_node, 0, true);
  NodeOutHasGuarder(if_node, 1, false);

  // then graph
  ASSERT_EQ(
      ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1))
          .StrictAllNodeTypes({{"InnerData", 3}, {"Foo1", 2}, {"Bar1", 1}, {"FreeMemory", 1}, {"InnerNetOutput", 1}}),
      "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1)->FindNode(foo2_node->GetNodeToken()))
                .StrictConnectTo(0, {{"FreeMemory"}, {"Foo1"}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1)->FindNode(foo3_node->GetNodeToken()))
                .StrictConnectTo(0, {{"InnerNetOutput"}}),
            "success");

  // else graph
  ASSERT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2))
                .StrictAllNodeTypes({{"InnerData", 3}, {"Foo1", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2)->FindNode(foo4_node->GetNodeToken()))
                .StrictConnectTo(0, {{"InnerNetOutput"}}),
            "success");
}
TEST_F(ConditionUT, If_ElsePlusAndThenMoveOut_OnlyThenHaveGuarder) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  ge::FastNode *foo2_node, *foo4_node;
  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
        foo2_node = foo2->GetFastNode();
        ValueHolder::CreateVoidGuarder("FreeMemory", foo2, {});
        return {data1, foo2};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo4 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data2});
        foo4_node = foo4->GetFastNode();
        return {data1, foo4};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);
  NodeOutHasGuarder(if_node, 0, false);
  NodeOutHasGuarder(if_node, 1, true);

  // then graph
  ASSERT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1))
                .StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1)->FindNode(foo2_node->GetNodeToken()))
                .StrictConnectTo(0, {{"InnerNetOutput", 1}}),
            "success");

  // else graph
  ASSERT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2))
                .StrictAllNodeTypes({{"InnerData", 3}, {"Foo1", 1}, {"IdentityAddr", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2)->FindNode(foo4_node->GetNodeToken()))
                .StrictConnectTo(0, {{"IdentityAddr", 0}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(
                                ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2), "InnerNetOutput"))
                .StrictConnectFrom({{"InnerData", 0}, {"IdentityAddr", 0}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(
                                ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2), "InnerNetOutput"))
                .InChecker()
                .DataFromByType("IdentityAddr")
                .DataFromByType("Foo1")
                .Result(),
            "success");
}
TEST_F(ConditionUT, If_ThenPlusAndElseMoveOut_OnlyElseHaveMultipleGuarders) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  std::string foo1_name, foo2_name;
  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateDataOutput("Foo2Out", {data0, data1}, 2);
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1Out", {data1, data2});
        return {foo1[0], foo2, foo1[0], foo1[1]};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data0, data1});
        ValueHolder::CreateVoidGuarder("FreeMemory", bar1, {});
        auto bar2 = ValueHolder::CreateSingleDataOutput("Bar", {data1, data2});
        ValueHolder::CreateVoidGuarder("FreeMemory", bar2, {});
        auto bar3 = ValueHolder::CreateSingleDataOutput("Bar", {bar1, bar2});
        ValueHolder::CreateVoidGuarder("FreeMemory", bar3, {});
        auto bar4 = ValueHolder::CreateSingleDataOutput("Bar", {data0, bar3});
        ValueHolder::CreateVoidGuarder("FreeMemory", bar4, {});
        return {bar1, bar2, bar3, bar4};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);
  NodeOutHasGuarder(if_node, 0);
  NodeOutHasGuarder(if_node, 1);
  NodeOutHasGuarder(if_node, 2);
  NodeOutHasGuarder(if_node, 3);

  // then graph
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(then_graph)
                .StrictAllNodeTypes(
                    {{"InnerData", 3}, {"Foo2Out", 1}, {"Foo1Out", 1}, {"IdentityAddr", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(
      FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput"))
          .StrictConnectFrom({{"IdentityAddr", 0}, {"IdentityAddr", 1}, {"IdentityAddr", 2}, {"IdentityAddr", 3}}),
      "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "IdentityAddr"))
                .StrictConnectFrom({{"Foo2Out", 0}, {"Foo1Out", 0}, {"Foo2Out", 0}, {"Foo2Out", 1}}),
            "success");

  // else graph
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph).StrictAllNodeTypes({{"InnerData", 3}, {"Bar", 4}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .StrictConnectFrom({{"Bar", 0}, {"Bar", 0}, {"Bar", 0}, {"Bar", 0}}),
            "success");
}

// Guarder构造需要正确构造具体的guarder 类型
TEST_F(ConditionUT, If_MoveGuarderOut_With_Special_Guarders) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  ge::FastNode *foo2_node, *foo3_node, *foo4_node;
  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {data0, data1});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
        foo2_node = foo2->GetFastNode();
        ValueHolder::CreateVoidGuarder("FreeMemory", foo2, {});
        auto foo3 = ValueHolder::CreateSingleDataOutput("Foo1", {bar1, foo2});
        ValueHolder::CreateVoidGuarder("FreeFftsMem", foo3, {});
        foo3_node = foo3->GetFastNode();
        return {foo3, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo4 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data2});
        ValueHolder::CreateVoidGuarder("FreeFftsMem", foo4, {});
        foo4_node = foo4->GetFastNode();
        return {foo4, data1};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);
  NodeOutHasGuarder(if_node, 0, true);
  NodeOutHasGuarder(if_node, 1, false);

  // then graph
  ASSERT_EQ(
      ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1))
          .StrictAllNodeTypes({{"InnerData", 3}, {"Foo1", 2}, {"Bar1", 1}, {"FreeMemory", 1}, {"InnerNetOutput", 1}}),
      "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1)->FindNode(foo2_node->GetNodeToken()))
                .StrictConnectTo(0, {{"FreeMemory"}, {"Foo1"}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1)->FindNode(foo3_node->GetNodeToken()))
                .StrictConnectTo(0, {{"InnerNetOutput"}}),
            "success");

  // else graph
  ASSERT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2))
                .StrictAllNodeTypes({{"InnerData", 3}, {"Foo1", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2)->FindNode(foo4_node->GetNodeToken()))
                .StrictConnectTo(0, {{"InnerNetOutput"}}),
            "success");

  // root graph
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "FreeFftsMem"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "FreeMemory"), nullptr);
}

TEST_F(ConditionUT, If_InnerDataWithGuarderOutside) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  ValueHolder::CreateVoidGuarder("FreeMemory", data1, {});

  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data1});
        return {foo1, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data1});
        return {bar1, data1};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);

  // else graph
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph)
                .StrictAllNodeTypes({{"InnerData", 1}, {"Bar", 1}, {"IdentityAddr", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Bar")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("IdentityAddr")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  auto inner_data = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerData");
  std::string guarder_type_outside;
  (void) ge::AttrUtils::GetStr(inner_data->GetOpDescBarePtr(), kNodeWithGuarderOutside, guarder_type_outside);
  EXPECT_EQ(guarder_type_outside, "FreeMemory");
}

TEST_F(ConditionUT, If_InnerDataWithGuarderOutside_FreeFixedFeatureMemory) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  ValueHolder::CreateVoidGuarder("FreeFixedFeatureMemory", data1, {});

  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data1});
        return {foo1, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data1});
        return {bar1, data1};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);

  // else graph
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph)
                .StrictAllNodeTypes({{"InnerData", 1}, {"Bar", 1}, {"IdentityAddr", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Bar")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("IdentityAddr")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  auto inner_data = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerData");
  std::string guarder_type_outside;
  (void) ge::AttrUtils::GetStr(inner_data->GetOpDescBarePtr(), kNodeWithGuarderOutside, guarder_type_outside);
  EXPECT_EQ(guarder_type_outside, "FreeFixedFeatureMemory");
}

TEST_F(ConditionUT, If_InnerDataWithGuarderOutside_With_Subgraph_Nesting) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  ValueHolder::CreateVoidGuarder("FreeMemory", data1, {});

  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto data2 = ValueHolder::CreateFeed(1);
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data2});
        auto sub_holder = bg::If<ValueHolder>(
            foo1,
            [&]() -> std::vector<ValueHolderPtr> {
              auto bar1 = ValueHolder::CreateSingleDataOutput("SubSubFoo", {data1});
              return {bar1, data1};
            },
            [&]() -> std::vector<ValueHolderPtr> {
              auto bar1 = ValueHolder::CreateSingleDataOutput("SubSubBar", {data1});
              return {bar1, data1};
            });
        return {foo1, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data1});
        return {bar1, data1};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);

  // then graph
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  auto sub_if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "If");
  ASSERT_NE(sub_if_node, nullptr);

  auto sub_then_graph = ge::FastNodeUtils::GetSubgraphFromNode(sub_if_node, 1);

  ASSERT_EQ(ExeGraphSummaryChecker(sub_then_graph)
                .StrictAllNodeTypes({{"InnerData", 1}, {"SubSubFoo", 1}, {"IdentityAddr", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_then_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("SubSubFoo")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_then_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("IdentityAddr")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  auto inner_data = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_then_graph, "InnerData");
  std::string guarder_type_outside;
  (void) ge::AttrUtils::GetStr(inner_data->GetOpDescBarePtr(), kNodeWithGuarderOutside, guarder_type_outside);
  EXPECT_EQ(guarder_type_outside, "FreeMemory");
}

/*
 * 嵌套子图场景下，如果存在特定的Guarder类型，需要正确将生命周期延长，同时在父图上生成正确的guader type
 * */
TEST_F(ConditionUT, If_InnerDataWithSpecialGuarderOutside_With_Subgraph_Nesting) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  ValueHolder::CreateVoidGuarder("FreeBatchFftsMems", data1, {});

  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto data2 = ValueHolder::CreateFeed(1);
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {data2});
        auto sub_holder = bg::If<ValueHolder>(
            foo1,
            [&]() -> std::vector<ValueHolderPtr> {
              auto bar1 = ValueHolder::CreateSingleDataOutput("SubSubFoo", {data1});
              return {bar1, data1};
            },
            [&]() -> std::vector<ValueHolderPtr> {
              auto bar1 = ValueHolder::CreateSingleDataOutput("SubSubBar", {data1});
              return {bar1, data1};
            });
        return {foo1, data1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data1});
        return {bar1, data1};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "FreeBatchFftsMems"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "FreeMemory"), nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);

  // then graph
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  auto sub_if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "If");
  ASSERT_NE(sub_if_node, nullptr);

  auto sub_free_batch_ffts_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "FreeBatchFftsMems");
  ASSERT_NE(sub_free_batch_ffts_node, nullptr);
  auto sub_free_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "FreeMemory");
  ASSERT_EQ(sub_free_node, nullptr);
}

TEST_F(ConditionUT, If_FreeTwoOutPlusOneInside_OneGuarderConnectToOut) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  std::string foo1_name, foo2_name;
  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo2Out", {data0, data1});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1Out", {data1, data2});
        return {foo1, foo2};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {data0, data1});
        ValueHolder::CreateVoidGuarder("FreeMemory", bar1, {});
        return {bar1, bar1};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);
  NodeOutHasGuarder(if_node, 0);
  NodeOutHasGuarder(if_node, 1);

  // else graph
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph)
                .StrictAllNodeTypes({{"InnerData", 2}, {"Bar", 1}, {"IdentityAddr", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Bar")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("IdentityAddr")
                .DataFromByType("Bar")
                .DataFromByType("InnerData")
                .Result(),
            "success");
}

TEST_F(ConditionUT, If_AddPointFrom_DiffDataOut) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  std::string foo1_name, foo2_name;
  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
        return {data1, foo1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
        return {data0, foo2};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);

  // then graph
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(then_graph)
                .StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"PointFromInputs", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("PointFromInputs")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Foo1")
                .DataFromByType("InnerData")
                .Result(),
            "success");

  // else graph
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph)
                .StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"PointFromInputs", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("PointFromInputs")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Foo1")
                .DataFromByType("InnerData")
                .Result(),
            "success");
}
TEST_F(ConditionUT, If_DoNothing_SameDataOut) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  std::string foo1_name, foo2_name;
  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
        return {data1, foo1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
        return {data1, foo2};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);

  // then graph
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(then_graph).StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput"))
                .StrictConnectFrom({{"InnerData"}, {"Foo1"}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Foo1")
                .DataFromByType("InnerData")
                .Result(),
            "success");

  // else graph
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph).StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .StrictConnectFrom({{"InnerData"}, {"Foo1"}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Foo1")
                .DataFromByType("InnerData")
                .Result(),
            "success");
}
TEST_F(ConditionUT, Case_AddPointFrom_OneDataOut) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  std::string foo1_name, foo2_name;
  auto holder = Case<ValueHolder>(data0, {[&]() -> std::vector<ValueHolderPtr> {
                               auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
                               return {foo1, foo1};
                             },
                             [&]() -> std::vector<ValueHolderPtr> {
                               auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
                               return {data1, foo2};
                             },
                             [&]() -> std::vector<ValueHolderPtr> {
                               auto foo2 = ValueHolder::CreateSingleDataOutput("Bar1", {data0, data1});
                               return {foo2, foo2};
                             }});

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto case_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "Case");
  ASSERT_NE(case_node, nullptr);

  // branch 1 graph
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(case_node, 1);
  ASSERT_NE(then_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(then_graph)
                .StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"PointFromInputs", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput"))
                .StrictConnectFrom({{"PointFromInputs"}, {"Foo1"}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Foo1")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("PointFromInputs")
                .DataFromByType("Foo1")
                .DataFromByType("InnerData")
                .Result(),
            "success");

  // branch 2 graph
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(case_node, 2);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph)
                .StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"PointFromInputs", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .StrictConnectFrom({{"PointFromInputs"}, {"Foo1"}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Foo1")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("PointFromInputs")
                .DataFromByType("InnerData")
                .Result(),
            "success");

  // branch 3 graph
  auto branch3_graph = ge::FastNodeUtils::GetSubgraphFromNode(case_node, 3);
  ASSERT_NE(branch3_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(branch3_graph)
                .StrictAllNodeTypes({{"InnerData", 2}, {"Bar1", 1}, {"PointFromInputs", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(branch3_graph, "InnerNetOutput"))
                .StrictConnectFrom({{"PointFromInputs"}, {"Bar1"}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(branch3_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("Bar1")
                .DataFromByType("InnerData")
                .Result(),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(branch3_graph, "InnerNetOutput"))
                .InChecker()
                .DataFromByType("PointFromInputs")
                .DataFromByType("Bar1")
                .DataFromByType("InnerData")
                .Result(),
            "success");
}
TEST_F(ConditionUT, If_CreateOnePointFrom_MultipleConflicts) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  std::string foo1_name, foo2_name;
  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
        auto foo2 = ValueHolder::CreateDataOutput("Foo2", {foo1, data1}, 2);
        return {data1, foo1, foo2[0], foo2[1]};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto foo3 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
        return {data1, data2, foo3, data0};
      });

  auto frame = ValueHolder::PopGraphFrame(holder, {});
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph().get();
  ASSERT_NE(graph, nullptr);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, "If");
  ASSERT_NE(if_node, nullptr);

  // then graph
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(then_graph)
                .StrictAllNodeTypes(
                    {{"InnerData", 2}, {"Foo1", 1}, {"Foo2", 1}, {"PointFromInputs", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput"))
                .StrictConnectFrom({{"InnerData"}, {"PointFromInputs", 0}, {"Foo2", 0}, {"PointFromInputs", 1}}),
            "success");
  auto pfi = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "PointFromInputs");
  ASSERT_NE(pfi, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(pfi).StrictConnectFrom({{"Foo1", 0}, {"Foo2", 1}}), "success");

  // else graph
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(else_graph)
                .StrictAllNodeTypes({{"InnerData", 3}, {"Foo1", 1}, {"PointFromInputs", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "InnerNetOutput"))
                .StrictConnectFrom({{"InnerData"}, {"PointFromInputs", 0}, {"Foo1"}, {"PointFromInputs", 1}}),
            "success");
  pfi = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "PointFromInputs");
  ASSERT_NE(pfi, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(pfi).StrictConnectFrom({{"InnerData", 0}, {"InnerData", 0}}), "success");
}
TEST_F(ConditionUT, Case_OutputPlacementsSameWithSubgraphs) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  data1->SetPlacement(kOnHost);
  data2->SetPlacement(kOnDeviceHbm);
  auto holder = If<ValueHolder>(
      data0,
      [&]() -> std::vector<ValueHolderPtr> {
        return {data1, data2};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        return {data1, data2};
      });

  ASSERT_EQ(holder.size(), 2);
  EXPECT_EQ(holder.at(0)->GetPlacement(), kOnHost);
  EXPECT_EQ(holder.at(1)->GetPlacement(), kOnDeviceHbm);
}

TEST_F(ConditionUT, Case_OutputPlacementsUnknown_SubgraphsDiff) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto data2 = ValueHolder::CreateFeed(2);

  data0->SetPlacement(kOnHost);
  data1->SetPlacement(kOnDeviceHbm);
  data2->SetPlacement(kFollowing);

  std::string foo1_name, foo2_name;
  auto holder = Case<ValueHolder>(data0, {[&]() -> std::vector<ValueHolderPtr> {
                               auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data1, data2});
                               foo1->SetPlacement(kOnDeviceHbm);
                               return {data1, foo1, data2};
                             },
                             [&]() -> std::vector<ValueHolderPtr> {
                               auto foo2 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
                               foo2->SetPlacement(kOnDeviceHbm);
                               return {data1, foo2, data2};
                             },
                             [&]() -> std::vector<ValueHolderPtr> {
                               auto foo2 = ValueHolder::CreateSingleDataOutput("Bar1", {data0, data1});
                               foo2->SetPlacement(kOnHost);
                               return {foo2, foo2, foo2};
                             }});

  ASSERT_EQ(holder.size(), 3);
  EXPECT_EQ(holder.at(0)->GetPlacement(), kTensorPlacementEnd);
  EXPECT_EQ(holder.at(1)->GetPlacement(), kTensorPlacementEnd);
  EXPECT_EQ(holder.at(2)->GetPlacement(), kOnHost);
}
}  // namespace bg
}  // namespace gert
