/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/gelocal/if_case_converter.h"
#include "engine/gelocal/subgraph_io_converter.h"

#include <gtest/gtest.h>

#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph_builder/value_holder_generator.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"

#include "common/share_graph.h"
#include "register/op_impl_registry.h"
#include "framework/common/types.h"
#include "core/builder/graph_node.h"
#include "common/ge_inner_attrs.h"
#include "graph/utils/execute_graph_adapter.h"

using namespace ge;
namespace gert {
namespace {
const std::string kStubLowerFunc = "_stub_lower_func";
}
class ReadInIfOrCaseNodeUT : public bg::BgTestAutoCreate3StageFrame {
 public:
  // 获取BranchPivot或者BranchDone对应的subgraph
  static ge::ExecuteGraph* GetSubgraph(const ge::FastNode *const node) {
    int64_t index;
    GE_ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::kRelativeBranch, index));
    auto frame_graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(frame_graph);
    auto cond_node = frame_graph->GetParentNodeBarePtr();
    GE_ASSERT_NOTNULL(cond_node);
    return ge::FastNodeUtils::GetSubgraphFromNode(cond_node, index);
  }

  void LoadIfAndCheck(const ge::FastNode *lower_if_node) {
    GraphNode graph_node;
    EXPECT_EQ(graph_node.ReadInNodeHasSubgraph(lower_if_node), ge::GRAPH_SUCCESS);

    auto cond_graph = ge::FastNodeUtils::GetSubgraphFromNode(lower_if_node, 0);
    auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(lower_if_node, 1);
    auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(lower_if_node, 2);
    EXPECT_NE(cond_graph, nullptr);
    EXPECT_NE(then_graph, nullptr);
    EXPECT_NE(else_graph, nullptr);

    const auto is_no_need_update = [](const char *node_type) {
      return IsGraphInputNode(node_type) || IsGraphOutputNode(node_type) || IsUsrOutputNode(node_type) ||
             IsMemTransferNode(node_type) || IsStroreConstDataNode(node_type);
    };
    auto switch_notify = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "SwitchNotify");
    auto wait_anyone = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "WaitAnyone");
    int64_t incre_indegree = 0;
    for (auto in_node : lower_if_node->GetAllInNodes()) {
      if (is_no_need_update(in_node->GetTypePtr())) {
        continue;
      }
      ++incre_indegree;
      auto &nodes = graph_node.additional_add_info[in_node];
      EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), switch_notify) != nodes.end());
    }
    EXPECT_EQ(incre_indegree, graph_node.additional_indegree_info[switch_notify]);

    for (auto out_node : lower_if_node->GetAllOutNodes()) {
      if (is_no_need_update(out_node->GetTypePtr())) {
        continue;
      }
      auto &nodes = graph_node.additional_add_info[wait_anyone];
      EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), out_node) != nodes.end());
      EXPECT_GE(graph_node.additional_indegree_info[out_node], 0);
    }

    const static auto kIsZeroInDegree = [](const ge::FastNode *const n) {
      auto op_type = n->GetTypePtr();
      return (IsInnerDataType(op_type) || IsConstType(op_type));
    };
    // anonymous function to determine weather branch node should be active by branch pivot
    const static auto kPivotGuard = [](const ge::FastNode *const n) {
      auto op_type = n->GetType();
      if (kIsZeroInDegree(n)) {
        return false;
      }
      const auto &in_nodes = n->GetInDataNodes();
      return std::all_of(in_nodes.begin(), in_nodes.end(), kIsZeroInDegree);
    };
    for (auto &n : cond_graph->GetDirectNode()) {
      if (n->GetType() != "BranchPivot" && n->GetType() != "BranchDone") {
        continue;
      }
      auto branch = GetSubgraph(n);
      ASSERT_NE(branch, nullptr);
      if (n->GetType() == "BranchPivot") {
        for (auto &node : branch->GetDirectNode()) {
          if (!kPivotGuard(node) || is_no_need_update(node->GetTypePtr())) {
            continue;
          }
          auto &nodes = graph_node.additional_add_info[n];
          EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), node) != nodes.end());
          EXPECT_GT(graph_node.additional_indegree_info[node], 0);
        }

      } else if (n->GetType() == "BranchDone") {
        auto net_output = ge::ExecuteGraphUtils::FindFirstNodeMatchType(branch, "InnerNetOutput");
        for (auto &node : net_output->GetInDataNodes()) {
          if (is_no_need_update(node->GetTypePtr())) {
            continue;
          }
          auto &nodes = graph_node.additional_add_info[node];
          EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), n) != nodes.end());
          EXPECT_GT(graph_node.additional_indegree_info[n], 0);
        }
        for (const auto node : branch->GetDirectNode()) {
          // 除InnerNetOutput之外的所有叶子节点
          if ((node->GetAllOutNodes().empty()) && (node->GetType() != "InnerNetOutput")) {
            if (is_no_need_update(node->GetTypePtr())) {
              continue;
            }
            auto &nodes = graph_node.additional_add_info[node];
            EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), n) != nodes.end());
            EXPECT_GT(graph_node.additional_indegree_info[n], 0);
          }
        }
      }
    }
  }
};

TEST_F(ReadInIfOrCaseNodeUT, ReadInIf_Success_TwoGraphs) {
  auto main_graph = ShareGraph::IfGraph();

  auto subgraphs = main_graph->GetAllSubgraphs();
  for (auto &subgraph : subgraphs) {
    ge::AttrUtils::SetStr(subgraph->FindFirstNodeMatchType("Shape")->GetOpDesc(), "_ge_attr_lowering_func",
                          kStubLowerFunc);
  }

  auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  LowerInput data_input = {{}, {}, &global_data};
  LowerInput if_input = {{}, {}, &global_data};
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == ge::DATA) {
      auto lr = LoweringDataNode(node, data_input);
      EXPECT_TRUE(lr.result.IsSuccess());
      if_input.input_shapes.push_back(lr.out_shapes[0]);
      if_input.input_addrs.push_back(lr.out_addrs[0]);
    }
  }

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  global_data.LoweringAndSplitRtStreams(1);
  auto if_lower_result = LoweringIf(if_node, if_input);
  EXPECT_TRUE(if_lower_result.result.IsSuccess());

  auto execute_graph =
      bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(if_lower_result.out_addrs), if_lower_result.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);

  auto lower_if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(execute_graph.get(), "If");
  EXPECT_NE(lower_if_node, nullptr);

  LoadIfAndCheck(lower_if_node);
}
TEST_F(ReadInIfOrCaseNodeUT, ReadInIf_Success_OneGraphOnlyDataAndNetOutput) {
  auto main_graph = ShareGraph::IfOneBranchGraph();
  for (auto &clean_node : ge::GraphUtils::FindNodesByTypeFromAllNodes(main_graph, "Clean")) {
    ge::AttrUtils::SetStr(clean_node->GetOpDesc(), "_ge_attr_lowering_func", kStubLowerFunc);
  }

  auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  LowerInput data_input = {{}, {}, &global_data};
  LowerInput if_input = {{}, {}, &global_data};
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == ge::DATA) {
      auto lr = LoweringDataNode(node, data_input);
      EXPECT_TRUE(lr.result.IsSuccess());
      if_input.input_shapes.push_back(lr.out_shapes[0]);
      if_input.input_addrs.push_back(lr.out_addrs[0]);
    }
  }

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  auto if_lower_result = LoweringIf(if_node, if_input);
  EXPECT_TRUE(if_lower_result.result.IsSuccess());

  auto execute_graph =
      bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(if_lower_result.out_addrs), if_lower_result.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  auto lower_if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(execute_graph.get(), "If");
  EXPECT_NE(lower_if_node, nullptr);

  LoadIfAndCheck(lower_if_node);
}
TEST_F(ReadInIfOrCaseNodeUT, ReadInIf_Success_OneEmptyGraph) {
  auto main_graph = ShareGraph::IfOneBranchGraph2();
  for (auto &clean_node : ge::GraphUtils::FindNodesByTypeFromAllNodes(main_graph, "Clean")) {
    ge::AttrUtils::SetStr(clean_node->GetOpDesc(), "_ge_attr_lowering_func", kStubLowerFunc);
  }

  auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  LowerInput data_input = {{}, {}, &global_data};
  LowerInput if_input = {{}, {}, &global_data};
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == ge::DATA) {
      auto lr = LoweringDataNode(node, data_input);
      EXPECT_TRUE(lr.result.IsSuccess());
      if_input.input_shapes.push_back(lr.out_shapes[0]);
      if_input.input_addrs.push_back(lr.out_addrs[0]);
    }
  }

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  auto if_lower_result = LoweringIf(if_node, if_input);
  EXPECT_TRUE(if_lower_result.result.IsSuccess());

  auto execute_graph =
      bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(if_lower_result.out_addrs), if_lower_result.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  auto lower_if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(execute_graph.get(), "If");
  EXPECT_NE(lower_if_node, nullptr);

  LoadIfAndCheck(lower_if_node);
}
}  // namespace gert
