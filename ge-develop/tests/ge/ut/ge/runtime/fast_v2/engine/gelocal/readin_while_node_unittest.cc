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

#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"

#include "framework/common/types.h"
#include "core/builder/graph_node.h"
#include "lowering/graph_converter.h"
#include "op_impl/less_important_op_impl.h"

using namespace ge;
namespace gert {
namespace {

const auto kIsZeroInDegree = [](const ge::FastNode *const n) {
  auto op_type = n->GetTypePtr();
  return (IsInnerDataType(op_type) || IsConstType(op_type));
};
// anonymous function to determine weather branch node should be active by branch pivot
const auto kPivotGuard = [](const ge::FastNode *const n) {
  auto op_type = n->GetType();
  if (kIsZeroInDegree(n)) {
    return false;
  }
  const auto &in_nodes = n->GetAllInNodes();
  return std::all_of(in_nodes.begin(), in_nodes.end(), kIsZeroInDegree);
};

const auto is_no_need_update = [](const char *node_type) {
  return IsGraphInputNode(node_type) || IsGraphOutputNode(node_type) || IsUsrOutputNode(node_type) ||
         IsMemTransferNode(node_type) || IsStroreConstDataNode(node_type);
};

#define EXPECT_GRAPH_GUARDED_BY(G, P, D, GN)                                   \
  do {                                                                         \
    for (auto &node : (G)->GetDirectNode()) {                                  \
      if (!kPivotGuard(node) || is_no_need_update(node->GetTypePtr())) {       \
        continue;                                                              \
      }                                                                        \
      auto &nodes = (GN).additional_add_info[P];                                 \
      EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), node) != nodes.end()); \
      EXPECT_GT((GN).additional_indegree_info[node], 0);                         \
    }                                                                          \
                                                                               \
    auto net_output = ge::ExecuteGraphUtils::FindFirstNodeMatchType((G), "InnerNetOutput");           \
    for (auto &node : net_output->GetAllInNodes()) {                           \
      if (is_no_need_update(node->GetTypePtr())) {                             \
        continue;                                                              \
      }                                                                        \
      auto &nodes = (GN).additional_add_info[node];                              \
      EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), (D)) != nodes.end());  \
      EXPECT_GT((GN).additional_indegree_info[(D)], 0);                          \
    }                                                                          \
    for (const auto node : (G)->GetDirectNode()) {                             \
      if ((node->GetAllOutNodes().empty()) && (node->GetType() != "InnerNetOutput")) {                \
        if (is_no_need_update(node->GetTypePtr())) {                           \
          continue;                                                            \
        }                                                                      \
        auto &nodes = (GN).additional_add_info[node];                          \
        EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), (D)) != nodes.end()); \
        EXPECT_GT((GN).additional_indegree_info[(D)], 0);                      \
      } \
    } \
  } while (false)
}  // namespace

class ReadInWhileNodeUT : public bg::BgTest {};
TEST_F(ReadInWhileNodeUT, ReadInWhileNode) {
  auto main_graph = ShareGraph::WhileGraph();
  ASSERT_NE(main_graph, nullptr);
  auto while_node = main_graph->FindFirstNodeMatchType("While");
  ASSERT_NE(while_node, nullptr);

  MockLessImportantNodeConverter(main_graph, {ge::WHILE, ge::DATA, ge::NETOUTPUT});
  auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  main_graph->TopologicalSorting();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto lower_3stage_graph = GraphConverter()
                                .SetModelDescHolder(&model_desc_holder)
                                .ConvertComputeGraphToExecuteGraph(main_graph, global_data);
  ASSERT_NE(lower_3stage_graph, nullptr);
  auto lower_graph = ge::FastNodeUtils::GetSubgraphFromNode(
      ExecuteGraphUtils::FindFirstNodeMatchType(lower_3stage_graph.get(), "Main"), 0);
  ASSERT_NE(lower_graph, nullptr);
  auto lower_while_node = ExecuteGraphUtils::FindFirstNodeMatchType(lower_graph, "While");
  ASSERT_NE(lower_while_node, nullptr);

  auto subgraph_names = lower_while_node->GetOpDescBarePtr()->GetSubgraphInstanceNames();
  EXPECT_EQ(subgraph_names.size(), 2);

  GraphNode graph_node;
  EXPECT_EQ(graph_node.ReadInNodeHasSubgraph(lower_while_node), ge::GRAPH_SUCCESS);

  auto control_frame_graph = lower_3stage_graph->GetSubGraph(subgraph_names[0]);
  auto body_graph = lower_3stage_graph->GetSubGraph(subgraph_names[1]);
  ASSERT_NE(control_frame_graph, nullptr);
  ASSERT_NE(body_graph, nullptr);

  int64_t incre_indegree = 0;
  auto enter = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_frame_graph, "Enter");
  auto exit = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_frame_graph, "Exit");
  for (auto in_node : lower_while_node->GetAllInNodes()) {
    if (is_no_need_update(in_node->GetTypePtr())) {
      continue;
    }
    ++incre_indegree;
    auto &nodes = graph_node.additional_add_info[in_node];
    EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), enter) != nodes.end());
  }
  EXPECT_EQ(incre_indegree, graph_node.additional_indegree_info[enter]);

  for (auto out_node : lower_while_node->GetAllOutNodes()) {
    if (is_no_need_update(out_node->GetTypePtr())) {
      continue;
    }
    auto &nodes = graph_node.additional_add_info[exit];
    EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), out_node) != nodes.end());
    EXPECT_GE(graph_node.additional_indegree_info[out_node], 0);
  }

  auto pivot = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_frame_graph, "BranchPivot");
  auto done = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_frame_graph, "BranchDone");
  ASSERT_NE(pivot, nullptr);
  ASSERT_NE(done, nullptr);
  EXPECT_GRAPH_GUARDED_BY(body_graph, pivot, done, graph_node);

  auto wait_anyone = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_frame_graph, "WaitAnyone");
  auto &done_watch_nodes = graph_node.additional_add_info[done];
  EXPECT_TRUE(std::find(done_watch_nodes.begin(), done_watch_nodes.end(), wait_anyone) != done_watch_nodes.end());

  auto cond_call = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_frame_graph, "SubgraphCall");
  ASSERT_NE(cond_call, nullptr);

  EXPECT_EQ(graph_node.ReadInNodeHasSubgraph(cond_call), ge::GRAPH_SUCCESS);
  auto cond_frame = lower_3stage_graph->GetSubGraph(cond_call->GetOpDescBarePtr()->GetSubgraphInstanceName(0));
  auto cond_graph = lower_3stage_graph->GetSubGraph(cond_call->GetOpDescBarePtr()->GetSubgraphInstanceName(1));
  ASSERT_NE(cond_frame, nullptr);
  ASSERT_NE(cond_graph, nullptr);
  pivot = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_frame, "BranchPivot");
  done = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_frame, "BranchDone");
  ASSERT_NE(pivot, nullptr);
  ASSERT_NE(done, nullptr);
  EXPECT_GRAPH_GUARDED_BY(cond_graph, pivot, done, graph_node);
}

TEST_F(ReadInWhileNodeUT, ReadInCascadeWhileNode) {
  auto main_graph = ShareGraph::WhileGraphCascade();
  ASSERT_NE(main_graph, nullptr);

  MockLessImportantNodeConverter(main_graph, {ge::WHILE, ge::DATA, ge::NETOUTPUT}, {}, true);
  auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  main_graph->TopologicalSorting();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto lower_3stage_graph = GraphConverter()
                                .SetModelDescHolder(&model_desc_holder)
                                .ConvertComputeGraphToExecuteGraph(main_graph, global_data);
  ASSERT_NE(lower_3stage_graph, nullptr);
  auto lower_graph = ge::FastNodeUtils::GetSubgraphFromNode(
      ExecuteGraphUtils::FindFirstNodeMatchType(lower_3stage_graph.get(), "Main"), 0);
  ASSERT_NE(lower_graph, nullptr);

  ge::FastNode *while1_node = nullptr;
  ge::FastNode *while2_node = nullptr;

  for (auto node : lower_graph->GetDirectNode()) {
    if (node->GetType() == "While" && strncmp(node->GetNamePtr(), "While_while1", strlen("While_while1")) == 0) {
      while1_node = node;
    }
    if (node->GetType() == "While" && strncmp(node->GetNamePtr(), "While_while2", strlen("While_while2")) == 0) {
      while2_node = node;
    }
  }
  ASSERT_NE(while1_node, nullptr);
  ASSERT_NE(while2_node, nullptr);

  GraphNode graph_node;
  EXPECT_EQ(graph_node.ReadInNodeHasSubgraph(while1_node), ge::GRAPH_SUCCESS);
  EXPECT_EQ(graph_node.ReadInNodeHasSubgraph(while2_node), ge::GRAPH_SUCCESS);
  auto control_graph1 = lower_3stage_graph->GetSubGraph(while1_node->GetOpDescBarePtr()->GetSubgraphInstanceName(0U));
  ASSERT_NE(control_graph1, nullptr);

  const auto control_graph2 = lower_3stage_graph->GetSubGraph(while2_node->GetOpDescBarePtr()->GetSubgraphInstanceName(0U));
  ASSERT_NE(control_graph2, nullptr);
  const auto control_graph1_end = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph1, "Exit");
  ASSERT_NE(control_graph1_end, nullptr);
  const auto control_graph2_start = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_graph2, "Enter");
  ASSERT_NE(control_graph2_start, nullptr);
  auto &nodes = graph_node.additional_add_info[control_graph1_end];
  EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), control_graph2_start) != nodes.end());
  EXPECT_GT(graph_node.additional_indegree_info[control_graph2_start], 0);
}
}  // namespace gert