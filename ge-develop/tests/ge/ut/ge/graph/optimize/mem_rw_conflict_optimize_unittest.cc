/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "graph/optimize/graph_optimize.h"
#include "macro_utils/dt_public_unscope.h"
#include "../passes/graph_builder_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/share_graph.h"

namespace ge {
class UTest_Graph_Mem_RW_Conflict_Optimize : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
namespace {
/*
 * Data -cast - netoutput
 */
ComputeGraphPtr BuildGraph_Readonly_Subgraph(const string subraph_name){
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("data1", DATA, 0,1);
  auto cast = sub_builder.AddNode("cast", CAST, 1,1);
  auto netoutput = sub_builder.AddNode("netoutput",NETOUTPUT, 1,1);
  AttrUtils::SetInt(data1->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX,0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0),ATTR_NAME_PARENT_NODE_INDEX,0);

  sub_builder.AddDataEdge(data1,0,cast,0);
  sub_builder.AddDataEdge(cast,0,netoutput,0);
  return sub_builder.GetGraph();
}

/*
   var   data1
     \    /
      assign
        |
    netoutput
*/
ComputeGraphPtr BuildGraph_Writeable_Subgraph(const string subraph_name) {
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("data1", DATA, 0, 1);
  auto var = sub_builder.AddNode("var1", VARIABLE, 0, 1);
  auto ref_node = sub_builder.AddNode("assign", ASSIGN, 2, 1);
  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0),ATTR_NAME_PARENT_NODE_INDEX,0);

  sub_builder.AddDataEdge(var, 0, ref_node, 0);
  sub_builder.AddDataEdge(data1, 0, ref_node, 1);
  sub_builder.AddDataEdge(ref_node, 0, netoutput, 0);
  return sub_builder.GetGraph();
}

/*
 * Data -cast - netoutput
 */
ComputeGraphPtr BuildGraph_With_Output_Readonly_Subgraph(const string subraph_name){
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("data1", DATA, 0,1);
  auto const1 = sub_builder.AddNode("const1", CONSTANT, 0,1);
  auto netoutput = sub_builder.AddNode("netoutput",NETOUTPUT, 1,1);
  AttrUtils::SetInt(data1->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0),ATTR_NAME_PARENT_NODE_INDEX,0);

  sub_builder.AddDataEdge(const1,0, netoutput,0);
  return sub_builder.GetGraph();
}

ComputeGraphPtr BuildGraph_With_Output_Writeable_Subgraph(const string subraph_name) {
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("data1", DATA, 0, 1);
  auto var = sub_builder.AddNode("var1", VARIABLE, 0, 1);
  auto ref_node = sub_builder.AddNode("assign", ASSIGN, 2, 1);
  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_builder.AddDataEdge(var, 0, ref_node, 0);
  sub_builder.AddDataEdge(data1, 0, ref_node, 1);
  sub_builder.AddDataEdge(ref_node, 0, netoutput, 0);
  return sub_builder.GetGraph();
}

/*
  data0   data1
     \    /
      assign
        |
    netoutput
*/
ComputeGraphPtr BuildGraph_Writeable_Subgraph2(const string subraph_name) {
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data0 = sub_builder.AddNode("data0", DATA, 0, 1);
  auto data1 = sub_builder.AddNode("data1", DATA, 0, 1);
  auto ref_node = sub_builder.AddNode("assign", ASSIGN, 2, 1);
  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_builder.AddDataEdge(data0, 0, ref_node, 0);
  sub_builder.AddDataEdge(data1, 0, ref_node, 1);
  sub_builder.AddDataEdge(ref_node, 0, netoutput, 0);
  return sub_builder.GetGraph();
}
/*
 *      const - allreduce
 *            \ 
 *              if
 *         insert identity
 */
ComputeGraphPtr BuildGraph_Readonly_ScopeWrite() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto ctrl_const = builder.AddNode("ctrl_const", CONSTANT, 0, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto if_node = builder.AddNode("if", IF, 1,0);

  builder.AddDataEdge(const1, 0, allreduce, 0);
  builder.AddDataEdge(const1, 0, if_node, 0);
  builder.AddControlEdge(ctrl_const, allreduce);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0,subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}
/*       const1---allreduce  const1--identity - allreduce
 *               /                 /
 *  var-identity--cast1   ==>   var-----cast1
 *              \                 \
 *               if                if
 */
ComputeGraphPtr BuildGraph_Identiyt_Split(){
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var", VARIABLE, 0, 1);
  auto identity = builder.AddNode("identity", IDENTITY, 1, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
  auto if_node = builder.AddNode("if", IF, 1,0);

  builder.AddDataEdge(var, 0 , identity, 0);
  builder.AddDataEdge(identity, 0 , allreduce, 0);
  builder.AddDataEdge(identity, 0 , cast1, 0);
  builder.AddDataEdge(identity, 0 , if_node, 0);
  builder.AddControlEdge(const1, allreduce);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0,subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}
/*
 * mul == allreduce
 * need insert identity
 */
ComputeGraphPtr BuildGraph_mul_1To2_ScopeWrite() {
  auto builder = ut::GraphBuilder("test");
  auto mul = builder.AddNode("mul", MUL, 2,1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 2,0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  builder.AddDataEdge(mul,0,allreduce,0);
  builder.AddDataEdge(mul,0,allreduce,1);
  return builder.GetGraph();
}
/*                                             foo1
 *                                              /
 *         foo1                            identity
 *          /                                 /
 * const ---------           ===>    const -----------
 *                \                               \
 *               foo2                          identity
 *                                                  \
 *                                                 foo2
 */
 ComputeGraphPtr BuildGraph_fifo_without_subgraph() {
   auto builder = ut::GraphBuilder("test");
   auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
   auto foo1 = builder.AddNode("foo1", RELU, 1,1);
   auto foo2 = builder.AddNode("foo2", RELU, 1,1);
   AttrUtils::SetInt(foo1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, 1);
   AttrUtils::SetInt(foo2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, 1);
   builder.AddDataEdge(const1, 0, foo1, 0);
   builder.AddDataEdge(const1, 0, foo2, 0);
   return builder.GetGraph();
}
/*                                             foo1
 *                                              /
 *         foo1                            identity
 *          /                                 /
 * const ---------           ===>    const -----------
 *                \                               \
 *                if                              if
 */
ComputeGraphPtr BuildGraph_fifo_with_subgraph() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto foo1 = builder.AddNode("foo1", RELU, 1, 1);
  auto if_node = builder.AddNode("if", IF, 1, 0);

  AttrUtils::SetInt(foo1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, 1);

  builder.AddDataEdge(const1, 0, foo1, 0);
  builder.AddDataEdge(const1, 0, if_node, 0);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}
/**  
 *         partitioncall
 *        +--------------------------+
 *        |                          |
 *   var->| data1                    |
 *        |       \                  |
 *        |        assign->netoutput |->netoutput
 *        |       /                  |
 * const->| data2                    |
 *        +--------------------------+
*/
ComputeGraphPtr BuildGraph_writable_subgraph_with_write() {
  auto builder = ut::GraphBuilder("test");
  auto var = builder.AddNode("var", VARIABLE, 0, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto partitioncall = builder.AddNode("partitioncall", PARTITIONEDCALL, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);

  builder.AddDataEdge(var, 0, partitioncall, 0);
  builder.AddDataEdge(const1, 0, partitioncall, 1);
  builder.AddDataEdge(partitioncall, 0, netoutput, 0);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "sub_branch";
  ComputeGraphPtr sub_branch = BuildGraph_Writeable_Subgraph2(subgraph_name);
  sub_branch->SetParentNode(partitioncall);
  sub_branch->SetParentGraph(root_graph);
  partitioncall->GetOpDesc()->AddSubgraphName(subgraph_name);
  partitioncall->GetOpDesc()->SetSubgraphInstanceName(0,subgraph_name);
  root_graph->AddSubgraph(subgraph_name, sub_branch);
  return root_graph;
}

ComputeGraphPtr BuildGraphWithSubgraph() {
  auto builder = ut::GraphBuilder("test");
  // id1 should be removed
  auto id1 = builder.AddNode("id1", IDENTITY, 1, 1);
  auto data0 = builder.AddNode("data0", DATA, 1, 1);
  auto data1 = builder.AddNode("data1", DATA, 1, 1);
  auto var0 = builder.AddNode("var0", VARIABLE, 1, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto ref_node = builder.AddNode("ref_node", ASSIGN, 2, 1);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto if_node = builder.AddNode("if", IF, 2, 2);
  auto netoutput_node = builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  auto out_anch = id1->GetOutControlAnchor();

  builder.AddDataEdge(data1, 0, id1, 0);
  builder.AddDataEdge(id1, 0, allreduce, 0);
  builder.AddDataEdge(allreduce, 0, if_node, 0);
  builder.AddDataEdge(data0, 0, if_node, 1);
  builder.AddDataEdge(var0, 0, ref_node, 1);
  builder.AddDataEdge(if_node, 0, ref_node, 0);
  builder.AddDataEdge(ref_node, 0, netoutput_node, 0);
  builder.AddDataEdge(if_node, 1, netoutput_node, 1);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  string subgraph_name1 = "else_branch";
  // else_branch_graph should insert indentity
  ComputeGraphPtr else_branch_graph = BuildGraph_Writeable_Subgraph(subgraph_name1);
  else_branch_graph->SetParentNode(if_node);
  else_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name1);
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, subgraph_name1);
  root_graph->AddSubgraph(subgraph_name1, else_branch_graph);
  return root_graph;
}

ComputeGraphPtr BuildGraphWithIfSubgraph() {
  auto builder = ut::GraphBuilder("test");
  // id1 should be removed
  auto id1 = builder.AddNode("id1", IDENTITY, 1, 1);
  auto data0 = builder.AddNode("data0", DATA, 0, 1);
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto var0 = builder.AddNode("var0", VARIABLE, 1, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto ref_node = builder.AddNode("ref_node", ASSIGN, 2, 1);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto if_node = builder.AddNode("if", IF, 2, 1);
  auto netoutput_node = builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  auto out_anch = id1->GetOutControlAnchor();

  builder.AddDataEdge(data1, 0, id1, 0);
  builder.AddDataEdge(id1, 0, allreduce, 0);
  builder.AddDataEdge(allreduce, 0, if_node, 0);
  builder.AddDataEdge(data0, 0, if_node, 1);
  builder.AddDataEdge(var0, 0, ref_node, 1);
  builder.AddDataEdge(if_node, 0, ref_node, 0);
  builder.AddDataEdge(ref_node, 0, netoutput_node, 0);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_With_Output_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  string subgraph_name1 = "else_branch";
  // else_branch_graph should insert indentity
  ComputeGraphPtr else_branch_graph = BuildGraph_With_Output_Writeable_Subgraph(subgraph_name1);
  else_branch_graph->SetParentNode(if_node);
  else_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name1);
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, subgraph_name1);
  root_graph->AddSubgraph(subgraph_name1, else_branch_graph);
  return root_graph;
}

}  // namespace
// const -> allreduce
// const -> Identity -> allreduce
TEST(UtestGraphPassesHcclMemcpyPass, testReadonlyScopeWriteConflict) {
  ComputeGraphPtr graph = BuildGraph_Readonly_ScopeWrite();
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto allreduce = graph->FindNode("allreduce");
  EXPECT_EQ(allreduce->GetInDataNodes().at(0)->GetType(), IDENTITY);
}
TEST(UtestGraphPassesHcclMemcpyPass, testIdentiytSplit) {
  ComputeGraphPtr graph = BuildGraph_Identiyt_Split();
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto allreduce = graph->FindNode("allreduce");
  auto allreduce_in_node = allreduce->GetInDataNodes().at(0);
  EXPECT_EQ(allreduce_in_node->GetType(), IDENTITY);
  EXPECT_EQ(allreduce_in_node->GetInControlNodes().at(0)->GetType(), CONSTANT);
}

/*
 * mul == allreduce
 * need insert identity
 */
TEST(UtestGraphPassesHcclMemcpyPass, testMul_1To2_ScopeWrite) {
  ComputeGraphPtr graph = BuildGraph_mul_1To2_ScopeWrite();
  EXPECT_EQ(graph->GetDirectNodesSize(), 2);
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
}

TEST(UtestGraphPassesHcclMemcpyPass, testRWConflictOfFIFOWithoutSubgraph) {
  ComputeGraphPtr graph = BuildGraph_fifo_without_subgraph();
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  auto assign1 = graph->FindNode("foo1");
  EXPECT_EQ(assign1->GetInDataNodes().at(0)->GetType(), IDENTITY);
  auto assign2 = graph->FindNode("foo2");
  EXPECT_EQ(assign2->GetInDataNodes().at(0)->GetType(), IDENTITY);
}

TEST(UtestGraphPassesHcclMemcpyPass, testRWConflictOfFIFOWithSubgraph) {
  ComputeGraphPtr graph = BuildGraph_fifo_with_subgraph();
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  auto assign1 = graph->FindNode("foo1");
  EXPECT_EQ(assign1->GetInDataNodes().at(0)->GetType(), IDENTITY);
}

/*
 *      const - allreduce
 *            \ if
 *         insert identity
 */
TEST(UtestGraphPassesHcclMemcpyPass, CheckRWConflict) {
  ComputeGraphPtr graph = BuildGraph_Readonly_ScopeWrite();
  bool has_conflict = true;
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.CheckRWConflict(graph, has_conflict), SUCCESS);
}

/**  
 *         partitioncall
 *        +--------------------------+
 *        |                          |
 *   var->| data1                    |
 *        |       \                  |
 *        |        assign->netoutput |
 *        |       /                  |
 * const->| data2                    |
 *        +--------------------------+
*/
TEST(UtestGraphPassesHcclMemcpyPass, CheckRWConflict_VarDirectConnectToPartitioncall) {
  ComputeGraphPtr graph = BuildGraph_writable_subgraph_with_write();
  bool has_conflict = true;
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.CheckRWConflict(graph, has_conflict), SUCCESS);
  EXPECT_EQ(has_conflict, false);

  // if parent node index on subgraph data invalid, consider it has conflict
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "data0" && AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX)) {
      // 给子图内data设置一个错误的parentnode index
      AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, -1);
    }
  }
  GraphOptimize graph_optimizer1;
  EXPECT_EQ(graph_optimizer1.CheckRWConflict(graph, has_conflict), SUCCESS);
  EXPECT_TRUE(has_conflict == true);
}

TEST(UtestGraphPassesHcclMemcpyPass, HandleMemoryRWConflictWithSubgraphs) {
  ComputeGraphPtr graph = BuildGraphWithSubgraph();
  GraphOptimize graph_optimizer;
  auto before_node_size = graph->GetDirectNodesSize();
  auto else_graph = graph->GetSubgraph("else_branch");
  auto before_node_size_else = else_graph->GetDirectNodesSize();
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  // data和scopwrite输入链接，不能删除id1
  EXPECT_EQ(before_node_size, graph->GetDirectNodesSize());
  // add identity in else graph before Netoutput, size +1
  EXPECT_EQ(before_node_size_else, else_graph->GetDirectNodesSize() - 1U);
}

TEST(UtestGraphPassesHcclMemcpyPass, HandleMemoryRWConflictWithIfSubgraphs) {
  ComputeGraphPtr graph = BuildGraphWithIfSubgraph();
  GraphOptimize graph_optimizer;
  auto before_node_size = graph->GetDirectNodesSize();
  auto else_graph = graph->GetSubgraph("else_branch");
  auto before_node_size_else = else_graph->GetDirectNodesSize();
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  // data和scopwrite输入链接，不能删除id1
  EXPECT_EQ(before_node_size, graph->GetDirectNodesSize());
  // add identity in else graph before Netoutput, size +1
  EXPECT_EQ(before_node_size_else, else_graph->GetDirectNodesSize() - 1U);
  // insert identity into else graph
  EXPECT_NE(else_graph->FindFirstNodeMatchType("Identity"), nullptr);
}

/**
mul --> allreduce
    \
        allreduce 
need insert identity before allreduce
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestSoftread2MultiScopeWrite) {
  auto builder = ut::GraphBuilder("test");
  auto mul = builder.AddNode("mul", MUL, 2,1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1,0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto allreduce2 = builder.AddNode("allreduce2", HCOMALLREDUCE, 1,0);
  AttrUtils::SetBool(allreduce2->GetOpDesc(), "_input_mutable", true);
  builder.AddDataEdge(mul,0,allreduce,0);
  builder.AddDataEdge(mul,0,allreduce2,0);
  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_EQ(allreduce->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_EQ(allreduce2->GetInDataNodes().at(0)->GetType(), IDENTITY);
}

/*
 cosnt -> assign  ==> const --> identity --> assign
 var   /                          val    /
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestReadOnly2Writable) {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("data1", CONSTANT, 0, 1);
  auto var = builder.AddNode("var1", VARIABLE, 0, 1);
  auto ref_node = builder.AddNode("assign", ASSIGN, 2, 1);
  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  builder.AddDataEdge(data1, 0, ref_node, 0);
  builder.AddDataEdge(var, 0, ref_node, 1);
  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_EQ(ref_node->GetInDataNodes().at(0)->GetType(), IDENTITY);
}
/*
        data               except: no identity insert
        /\
       /  \
    assign assign
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestRootGraphData2Writable) {
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data1", DATA, 0, 1);
  auto ref_node1 = builder.AddNode("assign1", ASSIGN, 1, 1);
  auto ref_node2 = builder.AddNode("assign2", ASSIGN, 1, 1);

  ref_node1->GetOpDesc()->UpdateInputName({{"ref", 0}});
  ref_node1->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  ref_node2->GetOpDesc()->UpdateInputName({{"ref", 0}});
  ref_node2->GetOpDesc()->UpdateOutputName({{"ref", 0}});

  builder.AddDataEdge(data, 0, ref_node1, 0);
  builder.AddDataEdge(data, 0, ref_node2, 0);

  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_NE(ref_node1->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_NE(ref_node2->GetInDataNodes().at(0)->GetType(), IDENTITY);
}
/*
        data1          data2
         |    \          /
         |     \        /
         |      \      /
  allreduce1    allreduce2  
data2-->allreduce2: 不插入identity
data1-->allreduce1: 插入identity
data1-->allreduce2: 插入identity
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestData2ScopeWriteNode) {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto allreduce1 = builder.AddNode("allreduce1", HCOMALLREDUCE, 1,0);
  AttrUtils::SetBool(allreduce1->GetOpDesc(), "_input_mutable", true);
  auto allreduce2 = builder.AddNode("allreduce2", HCOMALLREDUCE, 2,0);
  AttrUtils::SetBool(allreduce2->GetOpDesc(), "_input_mutable", true);

  builder.AddDataEdge(data1, 0, allreduce1, 0);
  builder.AddDataEdge(data1, 0, allreduce2, 0);
  builder.AddDataEdge(data2, 0, allreduce2, 1);

  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_EQ(allreduce1->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_EQ(allreduce2->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_NE(allreduce2->GetInDataNodes().at(1)->GetType(), IDENTITY);
}


/*
              data1      data2      data3
                |       /  |          /
                |      /   |         / 
            mul(ref 0)     |        /
              \            |       /
               \           |      /
              mul(ref 0)  mul(ref 0)  
                   \        /
                    \      /
                    allreduce
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestMulRefNode2ScopeWriteNode) {
  auto builder = ut::GraphBuilder("test");
  auto mul_ref1 = builder.AddNode("mul_ref1", MUL, 2, 1);
  mul_ref1->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  mul_ref1->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto mul_ref2 = builder.AddNode("mul_ref2", MUL, 2, 1);
  mul_ref2->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  mul_ref2->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto mul_ref3 = builder.AddNode("mul_ref3", MUL, 2, 1);
  mul_ref3->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  mul_ref3->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 2, 0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto data3 = builder.AddNode("data3", DATA, 0, 1);

  builder.AddDataEdge(data1, 0, mul_ref1, 0);
  builder.AddDataEdge(data2, 0, mul_ref1, 1);
  builder.AddDataEdge(mul_ref1, 0, mul_ref2, 0);
  builder.AddDataEdge(data2, 0, mul_ref2, 1);
  builder.AddDataEdge(data2, 0, mul_ref3, 0);
  builder.AddDataEdge(data3, 0, mul_ref3, 1);
  builder.AddDataEdge(mul_ref2, 0, allreduce, 0);
  builder.AddDataEdge(mul_ref3, 0, allreduce, 1);

  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_NE(allreduce->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_EQ(allreduce->GetInDataNodes().at(1)->GetType(), IDENTITY);
}

TEST(UtestGraphPassesHcclMemcpyPass, TestMulRefNode2ScopeWriteNode2) {
  auto builder = ut::GraphBuilder("test");
  auto mul_ref = builder.AddNode("mul_ref", MUL, 2, 1);
  mul_ref->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  mul_ref->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  AttrUtils::SetStr(mul_ref->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, "ref"); 
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);

  builder.AddDataEdge(mul_ref, 0, allreduce, 0);
  
  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto temp_node = allreduce->GetInDataNodes().at(0);
  EXPECT_EQ(temp_node->GetType(), IDENTITY);
  EXPECT_TRUE(!AttrUtils::HasAttr(temp_node->GetOpDesc()->GetInputDesc(0), REF_VAR_SRC_VAR_NAME));
}

TEST(UtestGraphPassesHcclMemcpyPass, TestConst2ScopeWriteNode) {
  auto builder = ut::GraphBuilder("test");
  auto constant = builder.AddNode("const", CONSTANT, 0, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);

  builder.AddDataEdge(constant, 0, allreduce, 0);
  
  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_EQ(allreduce->GetInDataNodes().at(0)->GetType(), IDENTITY);
}


}  // namespace ge
