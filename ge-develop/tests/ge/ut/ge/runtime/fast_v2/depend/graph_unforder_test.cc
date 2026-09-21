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
#include "ge_graph_dsl/graph_dsl.h"
#include "common/types.h"
#include "graph/unfold/graph_unfolder.h"
#include "common/share_graph.h"
#include "graph/utils/node_utils.h"

#include <omg/parser/parser_types.h>

using namespace ge;
namespace gert {
class GraphUnfolderTest : public testing::Test {};

void UpdateUnkownFlag(const ge::ComputeGraphPtr &root_graph){
  for (auto &sub_graph : root_graph->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(true);
  }
  int stage = 0;
  for (const auto &node : root_graph->GetDirectNode()){
    AttrUtils::SetInt(node->GetOpDesc(), ATTR_STAGE_LEVEL, stage);
  }
}

TEST_F(GraphUnfolderTest, test_subgraph_unford_success){
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("data_i", OP_CFG(ge::DATA_TYPE).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0))->NODE("less", ge::LESS)->NODE("netoutput", ge::NETOUTPUT));
    CHAIN(NODE("const_5", ge::CONSTANT)->NODE("less"));
  };

  DEF_GRAPH(sub_2) {
    CHAIN(NODE("data_a", OP_CFG(ge::DATA_TYPE).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1))->NODE("mul", ge::MUL)->NODE("netoutput", ge::NETOUTPUT));
    CHAIN(NODE("const_2", ge::CONSTANT)->NODE("mul"));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_a", OP_CFG(ge::DATA_TYPE).Attr(ATTR_NAME_INDEX, 0))->NODE("partition_call", ge::PARTITIONEDCALL, sub_1, sub_2));
    CHAIN(NODE("data_i", OP_CFG(ge::DATA_TYPE).Attr(ATTR_NAME_INDEX, 1))->NODE("partition_call"));
  };

  auto graph = ToComputeGraph(g1);
  UpdateUnkownFlag(graph);
  ge::ComputeGraphPtr flatten_graph;
  ASSERT_EQ(GraphUnfolder::UnfoldSubgraphs(graph, flatten_graph), ge::GRAPH_SUCCESS);
  ASSERT_EQ(flatten_graph->GetAllNodesSize(), 4);
}

TEST_F(GraphUnfolderTest, test_known_shape_subgraph_unford_success){
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("data_i", OP_CFG(ge::DATA_TYPE).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0))->NODE("less", ge::LESS)->NODE("netoutput", ge::NETOUTPUT));
    CHAIN(NODE("const_5", ge::CONSTANT)->NODE("less"));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_a", OP_CFG(ge::DATA_TYPE).Attr(ATTR_NAME_INDEX, 0))->NODE("partition_call", ge::PARTITIONEDCALL, sub_1));
    CHAIN(NODE("data_i", OP_CFG(ge::DATA_TYPE).Attr(ATTR_NAME_INDEX, 1))->NODE("partition_call"));
  };

  auto graph = ToComputeGraph(g1);
  ge::ComputeGraphPtr flatten_graph;
  ASSERT_EQ(GraphUnfolder::UnfoldSubgraphs(graph, flatten_graph), ge::GRAPH_SUCCESS);
  ASSERT_EQ(flatten_graph->GetAllNodesSize(), 7);
}

ComputeGraphPtr BuildNestedOnceGraph() {
  auto main_graph = std::make_shared<ge::ComputeGraph>("root");
  auto data = NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);

  auto sub_builder = []() {
    auto graph = std::make_shared<ge::ComputeGraph>("sub");
    auto data = NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output()
                    .Build(graph);
    auto output = NodeBuilder("output1", ge::NETOUTPUT).ControlInput(data).Build(graph);
    graph->SetGraphUnknownFlag(true);
    return graph;
  };

  auto nested = NodeBuilder("nested_call", ge::PARTITIONEDCALL).Input(data).Attr("f", sub_builder()).Build(main_graph);
  auto foo = NodeBuilder("foo", "Foo").ControlInput(nested).Build(main_graph);
  auto output = NodeBuilder("output", ge::NETOUTPUT).ControlInput(foo).Build(main_graph);

  GE_DUMP(main_graph, "Graph_nested");
  return main_graph;
}

ComputeGraphPtr BuildNestedTwiceGraph() {
  auto main_graph = std::make_shared<ge::ComputeGraph>("pipeline");
  auto data = NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto op = NodeBuilder("x", "X").Output().Build(main_graph);

  auto sub_builder = []() {
    auto graph = std::make_shared<ge::ComputeGraph>("nest_1");
    auto data = NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output()
                    .Build(graph);
    auto op = NodeBuilder("x1", "X").Input(data).Output().Build(graph);

    auto sub_builder = []() {
      auto graph = std::make_shared<ge::ComputeGraph>("nest_2");
      auto data = NodeBuilder("data2", ge::DATA)
                      .Attr(ge::ATTR_NAME_INDEX, 0)
                      .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                      .Output()
                      .Build(graph);
      auto output = NodeBuilder("output2", ge::NETOUTPUT).ControlInput(data).Build(graph);
      graph->SetGraphUnknownFlag(true);
      return graph;
    };

    auto nested_1 = NodeBuilder("nested_call1", ge::PARTITIONEDCALL).Input(op).Attr("f", sub_builder()).Build(graph);
    auto output = NodeBuilder("output1", ge::NETOUTPUT).ControlInput(nested_1).Build(graph);
    graph->SetGraphUnknownFlag(true);
    return graph;
  };

  auto nested = NodeBuilder("nested_call", ge::PARTITIONEDCALL).Input(op).Attr("f", sub_builder()).Build(main_graph);
  auto output = NodeBuilder("output", ge::NETOUTPUT).ControlInput(nested).Build(main_graph);

  GE_DUMP(main_graph, "Graph_nested");
  return main_graph;
}

ComputeGraphPtr BuildCaseSubGraph() {
  auto main_graph = std::make_shared<ge::ComputeGraph>("multi_batch_graph");
  auto data = NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data2 = NodeBuilder("data2", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);
  auto add1 = NodeBuilder("add", ADD).Input(data).Input(data2).Output().Build(main_graph);
  auto sub_builder = []() {
    auto sub_graph_builder = [](bool unknown_flag, std::string graph_name) {
      auto graph = std::make_shared<ge::ComputeGraph>(graph_name);
      auto data = NodeBuilder("data1", ge::DATA)
                      .Attr(ge::ATTR_NAME_INDEX, 0)
                      .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                      .Output()
                      .Build(graph);
      auto cast1 = NodeBuilder("cast1", CAST).Input(data).Output().Build(graph);
      auto output = NodeBuilder("output2", ge::NETOUTPUT).Input(cast1).Build(graph);
      graph->SetGraphUnknownFlag(unknown_flag);
      return graph;
    };

    auto sub_builder_case = [&sub_graph_builder](bool unknown_flag) {
      std::string graph_name = "case" + std::to_string(static_cast<int32_t>(unknown_flag)); 
      auto graph = std::make_shared<ge::ComputeGraph>(graph_name);
      auto data = NodeBuilder("data1", ge::DATA)
                      .Attr(ge::ATTR_NAME_INDEX, 0)
                      .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1)
                      .Output()
                      .Build(graph);
      auto data2 = NodeBuilder("data2", ge::DATA)
                      .Attr(ge::ATTR_NAME_INDEX, 1)
                      .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 2)
                      .Output()
                      .Build(graph);
      auto add2 = NodeBuilder("add2", ADD).Input(data).Input(data2).Output().Build(graph);
      std::string known_node_name = graph_name + "known";
      auto nested_known = NodeBuilder("nested_known", ge::PARTITIONEDCALL)
          .Input(add2).Output().Attr("subgraph_known", sub_graph_builder(false, known_node_name)).Build(graph);
      auto output = NodeBuilder("output2", ge::NETOUTPUT).Input(nested_known).Build(graph);
      graph->SetGraphUnknownFlag(unknown_flag);
      return graph;
    };

    auto graph = std::make_shared<ge::ComputeGraph>("nest_1");
    auto data = NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 0)
                    .Output()
                    .Build(graph);
    auto data2 = NodeBuilder("data2", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 1)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1)
                    .Output()
                    .Build(graph);
    auto data3 = NodeBuilder("data3", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 2)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 2)
                    .Output()
                    .Build(graph);
    auto op = NodeBuilder("case", "Case")
        .Input(data).Input(data2).Input(data3).Output()
        .Output()
        .Attr("batch1", sub_builder_case(false))
        .Attr("batch2", sub_builder_case(true))
        .Build(graph);
    auto nested_1 = NodeBuilder("nested_call1_known", ge::PARTITIONEDCALL)
        .Input(op).Output().Attr("known", sub_graph_builder(false, "unknown_graph_new")).Build(graph);
    auto output = NodeBuilder("output1", ge::NETOUTPUT).Input(nested_1).Build(graph);
    graph->SetGraphUnknownFlag(true);
    return graph;
  };
  auto nested = NodeBuilder("nested_call_unknown", ge::PARTITIONEDCALL)
      .Input(add1).Input(data).Input(data2).Output()
      .Attr("subgrah", sub_builder()).Build(main_graph);
  auto output = NodeBuilder("output", ge::NETOUTPUT).Input(nested).Build(main_graph);
  return main_graph;
}

ComputeGraphPtr BuildIfSubGraph() {
  auto main_graph = std::make_shared<ge::ComputeGraph>("multi_batch_graph");
  auto data = NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data2 = NodeBuilder("data2", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);

  auto sub_builder_known = []() {
    auto graph = std::make_shared<ge::ComputeGraph>("nest_2_known");
    auto data = NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1)
                    .Output()
                    .Build(graph);
    auto cast1 = NodeBuilder("cast1", CAST).Input(data).Output().Build(graph);
    auto output = NodeBuilder("output2", ge::NETOUTPUT).Input(cast1).Build(graph);
    graph->SetGraphUnknownFlag(false);
    return graph;
  };
  auto sub_builder_unknown = []() {
    auto graph = std::make_shared<ge::ComputeGraph>("nest_2_unknown");
    auto data = NodeBuilder("data1", ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1)
                    .Output()
                    .Build(graph);
    auto cast1 = NodeBuilder("cast1", CAST).Input(data).Output().Build(graph);
    auto output = NodeBuilder("output2", ge::NETOUTPUT).Input(cast1).Build(graph);
    graph->SetGraphUnknownFlag(true);
    return graph;
  };

  auto nested_1 = NodeBuilder("If", ge::IF)
      .Input(data).Input(data2).Output().Attr("unknown", sub_builder_unknown()).Attr("unknown", sub_builder_known()).Build(main_graph);
  auto output = NodeBuilder("output", ge::NETOUTPUT).Input(nested_1).Build(main_graph);
  return main_graph;
}

TEST_F(GraphUnfolderTest, test_nested_once_graph) {
  auto graph = BuildNestedOnceGraph();
  ge::ComputeGraphPtr flatten_graph;
  ASSERT_EQ(GraphUnfolder::UnfoldSubgraphs(graph, flatten_graph), ge::GRAPH_SUCCESS);
  ASSERT_EQ(flatten_graph->GetDirectNodesSize(), 3U);
  auto data = flatten_graph->FindNode("data");
  ASSERT_NE(data, nullptr);
  auto foo = flatten_graph->FindNode("foo");
  ASSERT_NE(foo, nullptr);
  ASSERT_EQ(flatten_graph->FindFirstNodeMatchType(ge::PARTITIONEDCALL), nullptr);  // Inlined
  ASSERT_EQ(data->GetOutControlAnchor()->GetFirstPeerAnchor(), foo->GetInControlAnchor());
}

TEST_F(GraphUnfolderTest, test_nested_twice_graph) {
  auto graph = BuildNestedTwiceGraph();
  ge::ComputeGraphPtr flatten_graph;
  ASSERT_EQ(GraphUnfolder::UnfoldSubgraphs(graph, flatten_graph), ge::GRAPH_SUCCESS);
  ASSERT_EQ(flatten_graph->GetDirectNodesSize(), 4U);  // data->x1->x2->net-output
  auto data = flatten_graph->FindNode("data");
  ASSERT_NE(data, nullptr);
  auto x = flatten_graph->FindNode("x");
  ASSERT_NE(x, nullptr);
  auto x1 = flatten_graph->FindNode("x1");
  ASSERT_NE(x1, nullptr);
  ASSERT_EQ(flatten_graph->FindFirstNodeMatchType(ge::PARTITIONEDCALL), nullptr);  // Inlined
  ASSERT_EQ(x->GetOutDataNodesSize(), 1U);
  ASSERT_EQ(*x->GetOutDataNodes().begin(), x1);
  auto output = flatten_graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(x1->GetOutControlAnchor()->GetFirstPeerAnchor(), output->GetInControlAnchor());
}

TEST_F(GraphUnfolderTest, test_case_sub_graph) {
  auto graph = BuildCaseSubGraph();
  ge::ComputeGraphPtr flatten_graph;
  ASSERT_EQ(GraphUnfolder::UnfoldSubgraphs(graph, flatten_graph), ge::GRAPH_SUCCESS);
  ASSERT_EQ(flatten_graph->GetDirectNodesSize(), 6U);
  ASSERT_EQ(flatten_graph->GetAllSubgraphs().size(), 5U);
  for (const auto &node : flatten_graph->GetDirectNode()) {
    if (node->GetType() == "Case") {
      std::vector<ge::ComputeGraphPtr> subgraphs;
      ge::NodeUtils::GetDirectSubgraphs(node, subgraphs);
      ASSERT_EQ(subgraphs.size(), 2UL);
      for (const auto &subgraph : subgraphs) {
        ASSERT_EQ(subgraph->GetDirectNodesSize(), 5U);
      }
    }
  }
}

TEST_F(GraphUnfolderTest, test_if_sub_graph) {
  auto graph = BuildIfSubGraph();
  ge::ComputeGraphPtr flatten_graph;
  ASSERT_EQ(GraphUnfolder::UnfoldSubgraphs(graph, flatten_graph), ge::GRAPH_SUCCESS);
  ASSERT_EQ(flatten_graph->GetDirectNodesSize(), 4U);
  ASSERT_EQ(flatten_graph->GetAllSubgraphs().size(), 2U);
}

/*
 *1、图上只有一个partitioncall
 *2、图上有多个partitioncall
 *3、图上有partitioncall嵌套
 *4、图上有if/case/while节点
 *5、partitioncall嵌套if/case/while
 *7、if/case/while嵌套partitioncall
 *8、
 */

// 在原图上展开partitioncall子图
TEST_F(GraphUnfolderTest, test_inplace_nested_once_graph) {


  auto graph = BuildNestedOnceGraph();
  ASSERT_EQ(GraphUnfolder::UnfoldAllPartitioncallInPlace(graph), ge::GRAPH_SUCCESS);
  ASSERT_EQ(graph->GetDirectNodesSize(), 3U);
  auto data = graph->FindNode("data");
  ASSERT_NE(data, nullptr);
  auto foo = graph->FindNode("foo");
  ASSERT_NE(foo, nullptr);
  ASSERT_EQ(graph->FindFirstNodeMatchType(ge::PARTITIONEDCALL), nullptr);  // Inlined
  ASSERT_EQ(data->GetOutControlAnchor()->GetFirstPeerAnchor(), foo->GetInControlAnchor());
}

// 在原图上展开嵌套的partitioncall子图
TEST_F(GraphUnfolderTest, test_inplace_nested_twice_graph) {
  auto graph = BuildNestedTwiceGraph();
  ASSERT_EQ(GraphUnfolder::UnfoldAllPartitioncallInPlace(graph), ge::GRAPH_SUCCESS);
  ASSERT_EQ(graph->GetDirectNodesSize(), 4U);  // data->x1->x2->net-output
  auto data = graph->FindNode("data");
  ASSERT_NE(data, nullptr);
  auto x = graph->FindNode("x");
  ASSERT_NE(x, nullptr);
  auto x1 = graph->FindNode("x1");
  ASSERT_NE(x1, nullptr);
  ASSERT_EQ(graph->FindFirstNodeMatchType(ge::PARTITIONEDCALL), nullptr);  // Inlined
  ASSERT_EQ(x->GetOutDataNodesSize(), 1U);
  ASSERT_EQ(*x->GetOutDataNodes().begin(), x1);
  auto output = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(x1->GetOutControlAnchor()->GetFirstPeerAnchor(), output->GetInControlAnchor());
}

// 在原图上展开case子图
TEST_F(GraphUnfolderTest, test_inplace_case_sub_graph) {
  auto graph =  ShareGraph::BuildCaseWithNestedPartitionedCall();
  ASSERT_EQ(GraphUnfolder::UnfoldAllPartitioncallInPlace(graph), ge::GRAPH_SUCCESS);
  auto case_node =  graph->FindFirstNodeMatchType(CASE);
  ASSERT_NE(case_node, nullptr);
  std::vector<ge::ComputeGraphPtr> subgraphs;
  ge::NodeUtils::GetDirectSubgraphs(case_node, subgraphs);
  ASSERT_EQ(subgraphs.size(), 2UL);
  // case子图上的partitioncall节点展开
  for (const auto &subgraph : subgraphs) {
    ASSERT_EQ(subgraph->FindFirstNodeMatchType(PARTITIONEDCALL), nullptr);
    ASSERT_NE(subgraph->FindFirstNodeMatchType(SQRT), nullptr);
  }
}

// 在原图上展开if子图
TEST_F(GraphUnfolderTest, test_inplace_if_sub_graph) {
  auto graph = ShareGraph::BuildIfWithNestedPartitionedCall();
  ASSERT_EQ(GraphUnfolder::UnfoldAllPartitioncallInPlace(graph), ge::GRAPH_SUCCESS);
  auto if_node = graph->FindFirstNodeMatchType(IF);
  ASSERT_NE(if_node, nullptr);
  std::vector<ge::ComputeGraphPtr> subgraphs;
  ge::NodeUtils::GetDirectSubgraphs(if_node, subgraphs);
  ASSERT_EQ(subgraphs.size(), 2UL);
  // if子图上的partitioncall节点展开
  for (const auto &subgraph : subgraphs) {
    ASSERT_EQ(subgraph->FindFirstNodeMatchType(PARTITIONEDCALL), nullptr);
    ASSERT_NE(subgraph->FindFirstNodeMatchType(SQRT), nullptr);
  }
}
} // namespace gert

