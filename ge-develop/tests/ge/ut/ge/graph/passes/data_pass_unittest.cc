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
#include "graph/utils/graph_utils.h"
#include "graph/passes/control_flow_and_stream/data_pass.h"
#include "graph/compute_graph.h"
#include "graph_builder_utils.h"

namespace ge{
  class UtestDataPass : public testing::Test {
   protected:
    void SetUp() {}
    void TearDown() {}
};

ComputeGraphPtr MakeGraph_PartitionedCall() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data = builder.AddNode("Data1", "Data", 1, 1);
  auto partcall1 = builder.AddNode("partcall1", "PartitionedCall", 1, 1);
  auto partcall2 = builder.AddNode("partcall2", "PartitionedCall", 1, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, partcall2, 0);
  builder.AddDataEdge(partcall2, 0, partcall1, 0);
  builder.AddDataEdge(partcall1, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph1");
  auto sub_data1 = sub_builder1.AddNode("sub_data1", "Data", 1, 1);
  auto data1_desc = sub_data1->GetOpDesc();
  AttrUtils::SetInt(data1_desc, "index", 0);
  auto sub_relu1 = sub_builder1.AddNode("sub_relu1", "Relu", 1, 1);
  auto sub_output1 = sub_builder1.AddNode("sub_output1", "NetOutput", 1, 0);
  sub_builder1.AddDataEdge(sub_data1, 0, sub_relu1, 0);
  sub_builder1.AddDataEdge(sub_relu1, 0, sub_output1, 0);
  auto subgraph1 = sub_builder1.GetGraph();

  auto part_node1 = root_graph->FindNode("partcall1");
  auto part_desc1 = part_node1->GetOpDesc();
  part_desc1->AddSubgraphName("sub_graph1");
  part_desc1->SetSubgraphInstanceName(0, "sub_graph1");

  subgraph1->SetParentNode(part_node1);
  subgraph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph1", subgraph1);

  return root_graph;
}

ComputeGraphPtr MakeGraph_CASE() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data = builder.AddNode("Data1", "Data", 1, 1);
  auto case1 = builder.AddNode("case1", "Case", 1, 1);
  auto case2 = builder.AddNode("case2", "Case", 1, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, case2, 0);
  builder.AddDataEdge(case2, 0, case1, 0);
  builder.AddDataEdge(case1, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph1_case");
  auto sub_data1 = sub_builder1.AddNode("Case", "Data", 1, 1);
  auto data1_desc = sub_data1->GetOpDesc();
  AttrUtils::SetInt(data1_desc, "index", 0);
  auto sub_relu1 = sub_builder1.AddNode("sub_relu1", "Relu", 1, 1);
  auto sub_output1 = sub_builder1.AddNode("sub_output1", "NetOutput", 1, 0);
  sub_builder1.AddDataEdge(sub_data1, 0, sub_relu1, 0);
  sub_builder1.AddDataEdge(sub_relu1, 0, sub_output1, 0);
  auto subgraph1 = sub_builder1.GetGraph();

  ut::GraphBuilder sub_builder2 = ut::GraphBuilder("sub_graph2_case");
  auto sub_data2 = sub_builder2.AddNode("sub_data2", "Data", 1, 1);
  auto case3 = sub_builder2.AddNode("case3", "Case", 1, 1);
  auto sub_output2 = sub_builder2.AddNode("sub_output2", "NetOutput", 1, 0);
  auto output2_desc = sub_output2->GetOpDesc();
  auto output2_desc_in = output2_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output2_desc_in, "_parent_node_index", 0);
  sub_builder2.AddDataEdge(sub_data2, 0, case3, 0);
  sub_builder2.AddDataEdge(case3, 0, sub_output2, 0);
  auto subgraph2 = sub_builder2.GetGraph();

  auto part_node1 = root_graph->FindNode("case1");
  auto part_desc1 = part_node1->GetOpDesc();
  part_desc1->AddSubgraphName("sub_graph1_case");
  part_desc1->SetSubgraphInstanceName(0, "sub_graph1_case");

  subgraph1->SetParentNode(part_node1);
  subgraph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph1_case", subgraph1);

  auto part_node2 = root_graph->FindNode("case2");
  auto part_desc2 = part_node2->GetOpDesc();
  part_desc2->AddSubgraphName("sub_graph2_case");
  part_desc2->SetSubgraphInstanceName(0, "sub_graph2_case");

  subgraph2->SetParentNode(part_node2);
  subgraph2->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph2_case", subgraph2);

  return root_graph;
}

ComputeGraphPtr MakeGraph_If() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data = builder.AddNode("Data1", "Data", 1, 1);
  auto if1 = builder.AddNode("if1", "If", 1, 1);
  auto if2 = builder.AddNode("if2", "If", 1, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, if2, 0);
  builder.AddDataEdge(if2, 0, if1, 0);
  builder.AddDataEdge(if1, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph1_if");
  auto sub_data1 = sub_builder1.AddNode("If", "Data", 1, 1);
  auto data1_desc = sub_data1->GetOpDesc();
  AttrUtils::SetInt(data1_desc, "index", 0);
  auto sub_relu1 = sub_builder1.AddNode("sub_relu1", "Relu", 1, 1);
  auto sub_output1 = sub_builder1.AddNode("sub_output1", "NetOutput", 1, 0);
  sub_builder1.AddDataEdge(sub_data1, 0, sub_relu1, 0);
  sub_builder1.AddDataEdge(sub_relu1, 0, sub_output1, 0);
  auto subgraph1 = sub_builder1.GetGraph();

  ut::GraphBuilder sub_builder2 = ut::GraphBuilder("sub_graph2_if");
  auto sub_data2 = sub_builder2.AddNode("sub_data2", "Data", 1, 1);
  auto if3 = sub_builder2.AddNode("if3", "If", 1, 1);
  auto sub_output2 = sub_builder2.AddNode("sub_output2", "NetOutput", 1, 0);
  auto output2_desc = sub_output2->GetOpDesc();
  auto output2_desc_in = output2_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output2_desc_in, "_parent_node_index", 0);
  sub_builder2.AddDataEdge(sub_data2, 0, if3, 0);
  sub_builder2.AddDataEdge(if3, 0, sub_output2, 0);
  auto subgraph2 = sub_builder2.GetGraph();

  auto part_node1 = root_graph->FindNode("if1");
  auto part_desc1 = part_node1->GetOpDesc();
  part_desc1->AddSubgraphName("sub_graph1_if");
  part_desc1->SetSubgraphInstanceName(0, "sub_graph1_if");

  subgraph1->SetParentNode(part_node1);
  subgraph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph1_if", subgraph1);

  auto part_node2 = root_graph->FindNode("if2");
  auto part_desc2 = part_node2->GetOpDesc();
  part_desc2->AddSubgraphName("sub_graph2_if");
  part_desc2->SetSubgraphInstanceName(0, "sub_graph2_if");

  subgraph2->SetParentNode(part_node2);
  subgraph2->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph2_if", subgraph2);

  return root_graph;
}

ComputeGraphPtr MakeGraph_While() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data = builder.AddNode("Data1", "Data", 1, 1);
  auto While1 = builder.AddNode("While1", "While", 1, 1);
  auto While2 = builder.AddNode("While2", "While", 1, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, While2, 0);
  builder.AddDataEdge(While2, 0, While1, 0);
  builder.AddDataEdge(While1, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph1_While");
  auto sub_data1 = sub_builder1.AddNode("While", "Data", 1, 1);
  auto data1_desc = sub_data1->GetOpDesc();
  AttrUtils::SetInt(data1_desc, "index", 0);
  auto sub_relu1 = sub_builder1.AddNode("sub_relu1", "Relu", 1, 1);
  auto sub_output1 = sub_builder1.AddNode("sub_output1", "NetOutput", 1, 0);
  sub_builder1.AddDataEdge(sub_data1, 0, sub_relu1, 0);
  sub_builder1.AddDataEdge(sub_relu1, 0, sub_output1, 0);
  auto subgraph1 = sub_builder1.GetGraph();

  ut::GraphBuilder sub_builder2 = ut::GraphBuilder("sub_graph2_While");
  auto sub_data2 = sub_builder2.AddNode("sub_data2", "Data", 1, 1);
  auto While3 = sub_builder2.AddNode("While3", "While", 1, 1);
  auto sub_output2 = sub_builder2.AddNode("sub_output2", "NetOutput", 1, 0);
  auto output2_desc = sub_output2->GetOpDesc();
  auto output2_desc_in = output2_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output2_desc_in, "_parent_node_index", 0);
  sub_builder2.AddDataEdge(sub_data2, 0, While3, 0);
  sub_builder2.AddDataEdge(While3, 0, sub_output2, 0);
  auto subgraph2 = sub_builder2.GetGraph();

  auto part_node1 = root_graph->FindNode("While1");
  auto part_desc1 = part_node1->GetOpDesc();
  part_desc1->AddSubgraphName("sub_graph1_While");
  part_desc1->SetSubgraphInstanceName(0, "sub_graph1_While");

  subgraph1->SetParentNode(part_node1);
  subgraph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph1_While", subgraph1);

  auto part_node2 = root_graph->FindNode("While2");
  auto part_desc2 = part_node2->GetOpDesc();
  part_desc2->AddSubgraphName("sub_graph2_While");
  part_desc2->SetSubgraphInstanceName(0, "sub_graph2_While");

  subgraph2->SetParentNode(part_node2);
  subgraph2->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph2_While", subgraph2);

  return root_graph;
}

ComputeGraphPtr MakeGraph_For() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data = builder.AddNode("Data1", "Data", 1, 1);
  auto For1 = builder.AddNode("For1", "For", 1, 1);
  auto For2 = builder.AddNode("For2", "For", 1, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, For2, 0);
  builder.AddDataEdge(For2, 0, For1, 0);
  builder.AddDataEdge(For1, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph1_For");
  auto sub_data1 = sub_builder1.AddNode("For", "Data", 1, 1);
  auto data1_desc = sub_data1->GetOpDesc();
  AttrUtils::SetInt(data1_desc, "index", 0);
  auto sub_relu1 = sub_builder1.AddNode("sub_relu1", "Relu", 1, 1);
  auto sub_output1 = sub_builder1.AddNode("sub_output1", "NetOutput", 1, 0);
  sub_builder1.AddDataEdge(sub_data1, 0, sub_relu1, 0);
  sub_builder1.AddDataEdge(sub_relu1, 0, sub_output1, 0);
  auto subgraph1 = sub_builder1.GetGraph();

  ut::GraphBuilder sub_builder2 = ut::GraphBuilder("sub_graph2_For");
  auto sub_data2 = sub_builder2.AddNode("sub_data2", "Data", 1, 1);
  auto For3 = sub_builder2.AddNode("For3", "For", 1, 1);
  auto sub_output2 = sub_builder2.AddNode("sub_output2", "NetOutput", 1, 0);
  auto output2_desc = sub_output2->GetOpDesc();
  auto output2_desc_in = output2_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output2_desc_in, "_parent_node_index", 0);
  sub_builder2.AddDataEdge(sub_data2, 0, For3, 0);
  sub_builder2.AddDataEdge(For3, 0, sub_output2, 0);
  auto subgraph2 = sub_builder2.GetGraph();

  auto part_node1 = root_graph->FindNode("For1");
  auto part_desc1 = part_node1->GetOpDesc();
  part_desc1->AddSubgraphName("sub_graph1_For");
  part_desc1->SetSubgraphInstanceName(0, "sub_graph1_For");

  subgraph1->SetParentNode(part_node1);
  subgraph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph1_For", subgraph1);

  auto part_node2 = root_graph->FindNode("For2");
  auto part_desc2 = part_node2->GetOpDesc();
  part_desc2->AddSubgraphName("sub_graph2_For");
  part_desc2->SetSubgraphInstanceName(0, "sub_graph2_For");

  subgraph2->SetParentNode(part_node2);
  subgraph2->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph2_For", subgraph2);

  return root_graph;
}

ComputeGraphPtr MakeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data = builder.AddNode("Data1", "Data", 1, 1);
  auto partcall1 = builder.AddNode("partcall1", "Cond", 1, 1);
  auto partcall2 = builder.AddNode("partcall2", "Cond", 1, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, partcall2, 0);
  builder.AddDataEdge(partcall2, 0, partcall1, 0);
  builder.AddDataEdge(partcall1, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph");
  auto sub_data1 = sub_builder1.AddNode("sub_data1", "Data", 1, 1);
  auto data1_desc = sub_data1->GetOpDesc();
  AttrUtils::SetInt(data1_desc, "index", 0);
  auto sub_relu1 = sub_builder1.AddNode("sub_relu1", "Relu", 1, 1);
  auto sub_output1 = sub_builder1.AddNode("sub_output1", "NetOutput", 1, 0);
  sub_builder1.AddDataEdge(sub_data1, 0, sub_relu1, 0);
  sub_builder1.AddDataEdge(sub_relu1, 0, sub_output1, 0);
  auto subgraph1 = sub_builder1.GetGraph();

  auto part_node1 = root_graph->FindNode("partcall1");
  auto part_desc1 = part_node1->GetOpDesc();
  part_desc1->AddSubgraphName("sub_graph");
  part_desc1->SetSubgraphInstanceName(0, "sub_graph");

  subgraph1->SetParentNode(part_node1);
  subgraph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph", subgraph1);

  return root_graph;
}

TEST_F(UtestDataPass, datapassRun) {

  auto graph1 = MakeGraph_PartitionedCall();
  auto sub_graph1 = graph1->GetSubgraph("sub_graph1");
  auto sub_graph2 = graph1->GetSubgraph("sub_graph2");
  auto sub_graph3 = graph1->GetSubgraph("sub_graph3");

  auto graph2 = MakeGraph_CASE();
  auto sub_graph1_case = graph2->GetSubgraph("sub_graph1_case");
  auto sub_graph2_case = graph2->GetSubgraph("sub_graph2_case");

  auto graph3 = MakeGraph_If();
  auto sub_graph1_if = graph3->GetSubgraph("sub_graph1_if");

  auto graph4 = MakeGraph_While();
  auto sub_graph1_While = graph4->GetSubgraph("sub_graph1_While");

  auto graph5 = MakeGraph_For();
  auto sub_graph1_For = graph5->GetSubgraph("sub_graph1_For");

  auto graph = MakeGraph();
  auto sub = graph->GetSubgraph("sub_graph");
  DataPass pass;

  EXPECT_EQ(pass.Run(graph1), SUCCESS);
  EXPECT_EQ(pass.Run(sub_graph1), SUCCESS);
  EXPECT_EQ(pass.Run(sub_graph1_case), SUCCESS);
  EXPECT_EQ(pass.Run(sub_graph1_if), SUCCESS);
  EXPECT_EQ(pass.Run(sub_graph1_While), SUCCESS);
  EXPECT_EQ(pass.Run(sub_graph1_For), SUCCESS);
  EXPECT_EQ(pass.Run(sub), FAILED);
}
}
