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
#include <string>
#include <map>
#include <gmock/gmock.h>

#include "graph/node.h"
#include "graph/passes/variable_optimize/split_variable_into_subgraph_pass.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"

#include "common/share_graph.h"

using namespace ge;

class UtestNodePassSplitVariableIntoSubgraphPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};


/*
      variable               +--------------------+
         |                   |         data       |
    partitioncall            |         /   \      |
         |                   |       cast1  cast2 |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
*/
ComputeGraphPtr BuildGraphVarPartitionedCall() {
  std::vector<int64_t> shape = {8, 3, 16, 16};  // HWCN
  auto variable = OP_CFG(VARIABLE)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
                      .InCnt(1)
                      .OutCnt(1)
                      .InNames({"x"})
                      .OutNames({"y"})
                      .Build("variable");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE(variable)->NODE("partitioncall", PARTITIONEDCALL)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");

  auto p_node = main_graph->FindFirstNodeMatchType(PARTITIONEDCALL);

  auto cast1 = OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Build("cast1");
  auto cast2 = OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Build("cast2");

  auto sub_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", "Data")->NODE(cast1)->NODE("NetOutput1", "NetOutput"));
      CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE(cast2)->NODE("NetOutput1", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  sub_graph->SetName("sub");
  auto data_node = sub_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto netoutput_node = sub_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_graph->SetParentGraph(main_graph);
  sub_graph->SetParentNode(p_node);

  main_graph->AddSubgraph(sub_graph);
  p_node->GetOpDesc()->AddSubgraphName("sub");
  p_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  sub_graph->SetGraphUnknownFlag(false);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"partitioncall"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}
/*
      variable               +--------------------+
         |                   |         data       |
    partitioncall            |         /   \      |
         |                   |       cast1  cast2 |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
*/
ComputeGraphPtr BuildGraphVarPartitionedCallWithWrongSubgraph() {
  std::vector<int64_t> shape = {8, 3, 16, 16};  // HWCN
  auto variable = OP_CFG(VARIABLE)
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .InNames({"x"})
      .OutNames({"y"})
      .Build("variable");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE(variable)->NODE("partitioncall", PARTITIONEDCALL)->NODE("NetOutput", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");

  auto p_node = main_graph->FindFirstNodeMatchType(PARTITIONEDCALL);

  auto cast1 = OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Build("cast1");
  auto cast2 = OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Build("cast2");

  auto sub_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "Data")->NODE(cast1)->NODE("NetOutput1", "NetOutput"));
                   CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE(cast2)->NODE("NetOutput1", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  sub_graph->SetName("sub");
  auto data_node = sub_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto netoutput_node = sub_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_graph->SetParentGraph(main_graph);
  sub_graph->SetParentNode(p_node);

  main_graph->AddSubgraph(sub_graph);
  p_node->GetOpDesc()->AddSubgraphName("sub_wrong");
  p_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub_wrong");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  sub_graph->SetGraphUnknownFlag(false);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"partitioncall"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
      variable               +--------------------+
         |                   |         data       |
    partitioncall            |         /   \      |
         |                   |       cast1  cast2 |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
*/
ComputeGraphPtr BuildGraphVarPartitionedCallWithUnknownShapeSubgraph() {
  std::vector<int64_t> shape = {8, -1, 16, 16};  // HWCN
  auto variable = OP_CFG(VARIABLE)
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .InNames({"x"})
      .OutNames({"y"})
      .Build("variable");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE(variable)->NODE("partitioncall", PARTITIONEDCALL)->NODE("NetOutput", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");

  auto p_node = main_graph->FindFirstNodeMatchType(PARTITIONEDCALL);

  auto cast1 = OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("cast1");
  auto cast2 = OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("cast2");

  auto sub_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "Data")->NODE(cast1)->NODE("NetOutput1", "NetOutput"));
                   CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE(cast2)->NODE("NetOutput1", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  sub_graph->SetName("sub");
  auto data_node = sub_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto netoutput_node = sub_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_graph->SetParentGraph(main_graph);
  sub_graph->SetParentNode(p_node);

  main_graph->AddSubgraph(sub_graph);
  p_node->GetOpDesc()->AddSubgraphName("sub");
  p_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  sub_graph->SetGraphUnknownFlag(true);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"partitioncall"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
 *   variable  data          +---------------------------+
         \     /             |    data    data1          |
    partitioncall            |       \       |           |
         |                   |       cast1  cast2  const |
      netoutput              |         |     /c   /c     |
                             |         netoutput         |
                             +---------------------------+
*/
ComputeGraphPtr BuildGraphVarPartitionedCallWithMultiInnerDataSubgraph() {
  std::vector<int64_t> shape = {8, 3, 16, 16};  // HWCN
  auto variable = OP_CFG(VARIABLE)
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .InNames({"x"})
      .OutNames({"y"})
      .Build("variable");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data", DATA)->NODE("partitioncall", PARTITIONEDCALL)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(variable)->EDGE(0, 1)->NODE("partitioncall", PARTITIONEDCALL)->NODE("NetOutput", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");

  auto p_node = main_graph->FindFirstNodeMatchType(PARTITIONEDCALL);

  auto cast1 = OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Build("cast1");
  auto cast2 = OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Build("cast2");

  auto sub_graph = [&]() {
    DEF_GRAPH(g) {
      CHAIN(NODE("data0", DATA)->NODE(cast1)->CTRL_EDGE()->NODE("NetOutput1", "NetOutput"));
      CHAIN(NODE("data1", DATA)->NODE(cast2)->NODE("NetOutput1", "NetOutput"));
      CHAIN(NODE("const", CONSTANT)->CTRL_EDGE()->NODE("NetOutput1", "NetOutput"));
    };
    return ToComputeGraph(g);
  }();
  sub_graph->SetName("sub");
  auto data_node = sub_graph->FindNode("data0");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto data1_node = sub_graph->FindNode("data1");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 1);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto netoutput_node = sub_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_graph->SetParentGraph(main_graph);
  sub_graph->SetParentNode(p_node);

  main_graph->AddSubgraph(sub_graph);
  p_node->GetOpDesc()->AddSubgraphName("sub");
  p_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  sub_graph->SetGraphUnknownFlag(false);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"partitioncall"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
      variable               +--------------------+
         |                   |  data    variable  |
    partitioncall            |     \    /         |
         |                   |       add          |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
*/
ComputeGraphPtr BuildGraphPartitionedCallWithVarInside() {
  std::vector<int64_t> shape = {8, 3, 16, 16};  // HWCN
  auto variable = OP_CFG(VARIABLE)
      .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
      .InCnt(1)
      .OutCnt(1)
      .InNames({"x"})
      .OutNames({"y"})
      .Build("variable");
  auto main_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE(variable)->NODE("partitioncall", PARTITIONEDCALL)->NODE("NetOutput", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");

  auto p_node = main_graph->FindFirstNodeMatchType(PARTITIONEDCALL);

  auto add = OP_CFG(ADD).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add");

  auto sub_graph = [&]() {
    DEF_GRAPH(g) {
                   CHAIN(NODE("data", "Data")->NODE(add)->NODE("NetOutput1", "NetOutput"));
                   CHAIN(NODE(variable)->EDGE(0, 1)->NODE(add)->NODE("NetOutput1", "NetOutput"));
                 };
    return ToComputeGraph(g);
  }();
  sub_graph->SetName("sub");
  auto data_node = sub_graph->FindFirstNodeMatchType("Data");
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(data_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto netoutput_node = sub_graph->FindFirstNodeMatchType("NetOutput");
  ge::AttrUtils::SetInt(netoutput_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_graph->SetParentGraph(main_graph);
  sub_graph->SetParentNode(p_node);

  main_graph->AddSubgraph(sub_graph);
  p_node->GetOpDesc()->AddSubgraphName("sub");
  p_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");
  main_graph->TopologicalSorting();

  main_graph->SetGraphUnknownFlag(false);
  sub_graph->SetGraphUnknownFlag(false);

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"partitioncall"});
  net_output->GetOpDesc()->SetSrcIndex({0});

  return main_graph;
}

/*
      variable               +--------------------+
         |                   |         data       |
    partitioncall            |         /   \      |
         |                   |       cast1  cast2 |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
*/
TEST_F(UtestNodePassSplitVariableIntoSubgraphPass, split_variable_connect_to_partitioncall) {
  auto graph = BuildGraphVarPartitionedCall();
  auto partitioncall_node = graph->FindNode("partitioncall");
  auto var_of_root = graph->FindNode("variable");
  EXPECT_EQ(partitioncall_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(partitioncall_node->GetInDataNodesSize(), 1);

  SplitVariableIntoSubgraphPass split_var_into_subgraph_pass;
  NamesToPass names_to_passes;
  names_to_passes.emplace_back("SplitVariableIntoSubgraphPass", &split_var_into_subgraph_pass);
  GEPass ge_passes(graph);
  auto ret = ge_passes.Run(names_to_passes);
  EXPECT_EQ(ret, SUCCESS);

  // check var_ref in root graph
  EXPECT_EQ(partitioncall_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(partitioncall_node->GetInDataNodesSize(), 1);

  // check var_ref in sub graph
  auto subgraph_names = partitioncall_node->GetOpDescBarePtr()->GetSubgraphInstanceNames();
  for (const auto &subgraph_name : subgraph_names) {
    const auto subgraph = graph->GetSubgraph(subgraph_name);
    const auto var_of_sub = subgraph->FindNode(var_of_root->GetName());
    EXPECT_NE(var_of_sub, nullptr);
    std::string ref_var_src_var_name;
    AttrUtils::GetStr(var_of_sub->GetOpDesc()->GetOutputDescPtr(0U), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);
    EXPECT_TRUE(ref_var_src_var_name.empty());

    EXPECT_EQ(var_of_sub->GetInDataNodesSize(), 0);
    EXPECT_EQ(var_of_sub->GetInControlNodesSize(), 1);
    EXPECT_EQ(var_of_sub->GetOutDataNodesSize(), 2);
    EXPECT_STREQ(var_of_sub->GetOutDataNodes().at(0)->GetType().c_str(), CAST);
    EXPECT_STREQ(var_of_sub->GetOutDataNodes().at(1)->GetType().c_str(), CAST);
  }
}

/*
      variable               +--------------------+
         |                   |  data    variable  |
    partitioncall            |     \    /         |
         |                   |       add          |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
*/
TEST_F(UtestNodePassSplitVariableIntoSubgraphPass, split_variable_connect_to_partitioncall_already_has_var_insubgraph) {
  auto graph = BuildGraphPartitionedCallWithVarInside();
  auto partitioncall_node = graph->FindNode("partitioncall");
  auto var_of_root = graph->FindNode("variable");
  EXPECT_EQ(partitioncall_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(partitioncall_node->GetInDataNodesSize(), 1);

  SplitVariableIntoSubgraphPass split_var_into_subgraph_pass;
  NamesToPass names_to_passes;
  names_to_passes.emplace_back("SplitVariableIntoSubgraphPass", &split_var_into_subgraph_pass);
  GEPass ge_passes(graph);
  auto ret = ge_passes.Run(names_to_passes);
  EXPECT_EQ(ret, SUCCESS);

  // check var_ref in root graph
  EXPECT_EQ(partitioncall_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(partitioncall_node->GetInDataNodesSize(), 1);

  // check how many var_ref in sub graph
  auto subgraph_names = partitioncall_node->GetOpDescBarePtr()->GetSubgraphInstanceNames();
  EXPECT_EQ(subgraph_names.size(), 1);
  const auto subgraph = graph->GetSubgraph(subgraph_names[0]);
  const auto var_of_sub = subgraph->FindNode(var_of_root->GetName());
  EXPECT_NE(var_of_sub, nullptr);
  std::string ref_var_src_var_name;
  AttrUtils::GetStr(var_of_sub->GetOpDesc()->GetOutputDescPtr(0U), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);
  EXPECT_TRUE(ref_var_src_var_name.empty());

  EXPECT_EQ(var_of_sub->GetInDataNodesSize(), 0);
  EXPECT_EQ(var_of_sub->GetInControlNodesSize(), 1);
  EXPECT_EQ(var_of_sub->GetOutDataNodesSize(), 2);
  EXPECT_EQ(var_of_sub->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "add");
  EXPECT_EQ(var_of_sub->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(1)->GetOwnerNode()->GetName(), "add");
}

/*
 *                             +-------------+  +-----------+
 *                             |Then Graph   |  |Else Graph |
 *       NetOutput             |             |  |           |
 *           |                 | NetOutput   |  | NetOutput |
 *          if <----------->   |   |         |  |   |       |
 *           |                 | assign      |  |  Cast     |
 *           |                 |   |  \      |  |   |       |
 *        /    \               | Data const  |  | Data      |
 * pred(Data)  input(RefData)  +-------------+  +-----------+
 */
TEST_F(UtestNodePassSplitVariableIntoSubgraphPass, split_refdata_connect_to_if) {
  auto graph = gert::ShareGraph::IfGraphWithRefData();
  auto if_node = graph->FindNode("if");
  auto ref_data_of_root = graph->FindNode("input");
  EXPECT_EQ(if_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(if_node->GetInDataNodesSize(), 2);

  SplitVariableIntoSubgraphPass split_var_into_subgraph_pass;
  NamesToPass names_to_passes;
  names_to_passes.emplace_back("SplitVariableIntoSubgraphPass", &split_var_into_subgraph_pass);
  GEPass ge_passes(graph);
  auto ret = ge_passes.Run(names_to_passes);
  EXPECT_EQ(ret, SUCCESS);

  // check var_ref in root graph
  EXPECT_EQ(if_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(if_node->GetInDataNodesSize(), 2);

  // check var_ref in sub graph
  auto subgraph_names = if_node->GetOpDescBarePtr()->GetSubgraphInstanceNames();
  for (const auto &subgraph_name : subgraph_names) {
    const auto subgraph = graph->GetSubgraph(subgraph_name);
    const auto ref_data_of_sub = subgraph->FindNode(ref_data_of_root->GetName());
    EXPECT_NE(ref_data_of_sub, nullptr);

    EXPECT_EQ(ref_data_of_sub->GetInDataNodesSize(), 0);
    EXPECT_EQ(ref_data_of_sub->GetInControlNodesSize(), 1);
  }
}

/*
 *        NetOutput            +-------------+  +-----------+
 *            |                |   branch0   |  |  branch1  |
 *          cast               |             |  |           |
 *           |                 | NetOutput   |  | NetOutput |
 *          Case <-----------> |    |        |  |   |       |
 *           |                 | Cast        |  |  Cast     |
 *           |                 |   |         |  |   |       |
 *        /    \               | Data        |  | Data      |
 * pred(Data)  input(RefData)  +-------------+  +-----------+
 */
TEST_F(UtestNodePassSplitVariableIntoSubgraphPass, split_refdata_connect_to_case_multi_batch) {
  auto graph = gert::ShareGraph::CaseGraphWithRefData();
  auto case_node = graph->FindNode("case");
  auto ref_data_of_root = graph->FindNode("input");
  EXPECT_EQ(case_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(case_node->GetInDataNodesSize(), 2);

  SplitVariableIntoSubgraphPass split_var_into_subgraph_pass;
  NamesToPass names_to_passes;
  names_to_passes.emplace_back("SplitVariableIntoSubgraphPass", &split_var_into_subgraph_pass);
  GEPass ge_passes(graph);
  auto ret = ge_passes.Run(names_to_passes);
  EXPECT_EQ(ret, SUCCESS);

  // check var_ref in root graph
  EXPECT_EQ(case_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(case_node->GetInDataNodesSize(), 2);

  // check var_ref in sub graph
  auto subgraph_names = case_node->GetOpDescBarePtr()->GetSubgraphInstanceNames();
  for (const auto &subgraph_name : subgraph_names) {
    const auto subgraph = graph->GetSubgraph(subgraph_name);
    const auto ref_data_of_sub = subgraph->FindNode(ref_data_of_root->GetName());
    EXPECT_NE(ref_data_of_sub, nullptr);

    EXPECT_EQ(ref_data_of_sub->GetInDataNodesSize(), 0);
    EXPECT_EQ(ref_data_of_sub->GetInControlNodesSize(), 1);
    auto data_node = subgraph->FindFirstNodeMatchType("Data");
    auto data_op_desc = data_node->GetOpDesc();
    ASSERT_NE(data_op_desc, nullptr);
    auto ref_data_op_desc = ref_data_of_sub->GetOpDesc();
    ASSERT_NE(ref_data_op_desc, nullptr);
    EXPECT_EQ(ref_data_op_desc->GetOutputDesc(0U).GetShape(), data_op_desc->GetOutputDesc(0U).GetShape());
    EXPECT_EQ(ref_data_op_desc->GetOutputDesc(0U).GetOriginShape(), data_op_desc->GetOutputDesc(0U).GetOriginShape());
    EXPECT_EQ(ref_data_op_desc->GetInputDesc(0U).GetShape(), data_op_desc->GetInputDesc(0U).GetShape());
    EXPECT_EQ(ref_data_op_desc->GetInputDesc(0U).GetOriginShape(), data_op_desc->GetInputDesc(0U).GetOriginShape());
  }
}

/*                           body                       cond
 input(refdata)  data            +--------------------+      +---------------------+
       \         /                | data(ref)   data1  |     |      data(0)        |
           while                  |      \      /      |     |       |             |
              |                   |        assign      |     |      cast           |
           netoutput              |         |(0)       |     |       |(0)          |
                                  |      netoutput     |     |      netoutput      |
                                  +--------------------+     +---------------------+
 */
TEST_F(UtestNodePassSplitVariableIntoSubgraphPass, split_refdata_connect_to_while) {
  auto graph = gert::ShareGraph::BuildGraphRefdataWhile();
  auto while_node = graph->FindNode("while");
  auto ref_data_of_root = graph->FindNode("input");
  EXPECT_EQ(while_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(while_node->GetInDataNodesSize(), 2);

  SplitVariableIntoSubgraphPass split_var_into_subgraph_pass;
  NamesToPass names_to_passes;
  names_to_passes.emplace_back("SplitVariableIntoSubgraphPass", &split_var_into_subgraph_pass);
  GEPass ge_passes(graph);
  auto ret = ge_passes.Run(names_to_passes);
  EXPECT_EQ(ret, SUCCESS);

  // check var_ref in root graph
  EXPECT_EQ(while_node->GetInControlNodesSize(), 0);
  EXPECT_EQ(while_node->GetInDataNodesSize(), 2);
  EXPECT_EQ(while_node->GetOutDataAnchor(0)->GetPeerInDataNodesSize(), 0);
  EXPECT_EQ(while_node->GetOutControlNodesSize(), 1);
  EXPECT_EQ(while_node->GetOutControlNodes().at(0)->GetType(), CAST);
  EXPECT_EQ(ref_data_of_root->GetOutDataNodesSize(), 2);
  EXPECT_EQ(ref_data_of_root->GetOutDataNodes().at(1)->GetType(), CAST);

  // check var_ref in sub graph
  auto subgraph_names = while_node->GetOpDescBarePtr()->GetSubgraphInstanceNames();
  for (const auto &subgraph_name : subgraph_names) {
    const auto subgraph = graph->GetSubgraph(subgraph_name);
    const auto ref_data_of_sub = subgraph->FindNode(ref_data_of_root->GetName());
    EXPECT_NE(ref_data_of_sub, nullptr);

    EXPECT_EQ(ref_data_of_sub->GetInDataNodesSize(), 0);
    EXPECT_EQ(ref_data_of_sub->GetInControlNodesSize(), 1);
  }
}

/*
      variable               +--------------------+
         |                   |         data       |
    partitioncall            |         /   \      |
         |                   |       cast1  cast2 |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
*/
TEST_F(UtestNodePassSplitVariableIntoSubgraphPass, split_variable_connect_to_partitioncall_subgraph_is_null) {
  auto graph = BuildGraphVarPartitionedCallWithWrongSubgraph();
  SplitVariableIntoSubgraphPass split_var_into_subgraph_pass;
  NamesToPass names_to_passes;
  names_to_passes.emplace_back("SplitVariableIntoSubgraphPass", &split_var_into_subgraph_pass);
  GEPass ge_passes(graph);
  auto ret = ge_passes.Run(names_to_passes);
  EXPECT_NE(ret, SUCCESS);
}

/*
      variable               +--------------------+
         |                   |         data       |
    partitioncall            |         /   \      |
         |                   |       cast1  cast2 |
      netoutput              |         |          |
                             |      netoutput     |
                             +--------------------+
*/
TEST_F(UtestNodePassSplitVariableIntoSubgraphPass, split_variable_connect_to_partitioncall_subgraph_is_unknown_shape) {
  auto graph = BuildGraphVarPartitionedCallWithUnknownShapeSubgraph();
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  GE_DUMP(graph, "splitvar");
  SplitVariableIntoSubgraphPass split_var_into_subgraph_pass;
  NamesToPass names_to_passes;
  names_to_passes.emplace_back("SplitVariableIntoSubgraphPass", &split_var_into_subgraph_pass);
  GEPass ge_passes(graph);
  auto ret = ge_passes.Run(names_to_passes);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
}

/*
 *   variable  data          +---------------------------+
         \     /             |    data    data1          |
    partitioncall            |       \       |           |
         |                   |       cast1  cast2  const |
      netoutput              |         |     /c   /c     |
                             |         netoutput         |
                             +---------------------------+
*/
TEST_F(UtestNodePassSplitVariableIntoSubgraphPass, split_variable_connect_to_partitioncall_find_right_inner_data) {
  auto graph = BuildGraphVarPartitionedCallWithMultiInnerDataSubgraph();
  SplitVariableIntoSubgraphPass split_var_into_subgraph_pass;
  NamesToPass names_to_passes;
  names_to_passes.emplace_back("SplitVariableIntoSubgraphPass", &split_var_into_subgraph_pass);
  GEPass ge_passes(graph);
  auto ret = ge_passes.Run(names_to_passes);
  EXPECT_EQ(ret, SUCCESS);

  auto partitioncall_node = graph->FindNode("partitioncall");
  auto var_of_root = graph->FindNode("variable");
  // check var_ref in sub graph
  auto subgraph_names = partitioncall_node->GetOpDescBarePtr()->GetSubgraphInstanceNames();
  for (const auto &subgraph_name : subgraph_names) {
    const auto subgraph = graph->GetSubgraph(subgraph_name);
    const auto var_of_sub = subgraph->FindNode(var_of_root->GetName());
    EXPECT_NE(var_of_sub, nullptr);
    std::string ref_var_src_var_name;
    AttrUtils::GetStr(var_of_sub->GetOpDesc()->GetOutputDescPtr(0U), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);
    EXPECT_TRUE(ref_var_src_var_name.empty());

    EXPECT_EQ(var_of_sub->GetInDataNodesSize(), 0);
    EXPECT_EQ(var_of_sub->GetInControlNodesSize(), 1);
    EXPECT_EQ(var_of_sub->GetOutDataNodesSize(), 1);
    EXPECT_STREQ(var_of_sub->GetOutDataNodes().at(0)->GetType().c_str(), CAST);
  }
}
