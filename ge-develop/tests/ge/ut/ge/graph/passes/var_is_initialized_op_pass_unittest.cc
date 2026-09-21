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

#include "common/types.h"
#include "graph/graph.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/passes/control_flow_and_stream/merge_pass.h"
#include "graph/passes/control_flow_and_stream/switch_dead_branch_elimination.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/passes/variable_optimize/var_is_initialized_op_pass.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph_builder_utils.h"
#include "graph/passes/pass_manager.h"
#include "omg/omg.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"

using namespace domi;

namespace ge {

class UTEST_graph_passes_var_is_initialized_op_pass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

/*
 *           netoutput
 *              |
 *            merge
 *           /    \
 *  switch_ref     \
 *      |           \
 * var_is_init--->switch
 *     |            |
 *    var          const
 */
static void MakeGraphCorrect(ComputeGraphPtr graph) {
  GeTensorDesc ge_tensor_desc;
  GeTensorDesc ge_tensor_desc_var_is;
  ge_tensor_desc_var_is.SetDataType(DT_BOOL);

  auto variable_op = std::make_shared<OpDesc>("Variable", VARIABLE);
  variable_op->AddOutputDesc(ge_tensor_desc);
  auto variable_node = graph->AddNode(variable_op);

  auto const_op = std::make_shared<OpDesc>("const", CONSTANT);
  const_op->AddOutputDesc(ge_tensor_desc);
  auto const_node = graph->AddNode(const_op);

  auto var_is_initialized_op = std::make_shared<OpDesc>("VARISINITIALIZEDOP", VARISINITIALIZEDOP);
  var_is_initialized_op->AddInputDesc(ge_tensor_desc);
  var_is_initialized_op->AddOutputDesc(ge_tensor_desc_var_is);
  auto var_is_initialized_node = graph->AddNode(var_is_initialized_op);

  auto switch_op = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddOutputDesc(ge_tensor_desc);
  switch_op->AddOutputDesc(ge_tensor_desc);
  auto switch_node = graph->AddNode(switch_op);

  auto ref_switch_op = std::make_shared<OpDesc>("RefSwitch", REFSWITCH);
  ref_switch_op->AddInputDesc(ge_tensor_desc);
  ref_switch_op->AddInputDesc(ge_tensor_desc);
  ref_switch_op->AddOutputDesc(ge_tensor_desc);
  ref_switch_op->AddOutputDesc(ge_tensor_desc);
  auto ref_switch_node = graph->AddNode(ref_switch_op);

  auto merge_op = std::make_shared<OpDesc>("Merge", MERGE);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddOutputDesc(ge_tensor_desc);
  merge_op->AddOutputDesc(ge_tensor_desc);
  auto merge_node = graph->AddNode(merge_op);

  auto net_output_op = std::make_shared<OpDesc>("NetOutput", NETOUTPUT);
  net_output_op->AddInputDesc(ge_tensor_desc);
  net_output_op->AddOutputDesc(ge_tensor_desc);
  auto net_output_node = graph->AddNode(net_output_op);

  GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), var_is_initialized_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), ref_switch_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), ref_switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_switch_node->GetOutDataAnchor(1), merge_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

  GraphUtils::AddEdge(variable_node->GetOutControlAnchor(), var_is_initialized_node->GetInControlAnchor());
  GraphUtils::AddEdge(var_is_initialized_node->GetOutControlAnchor(), switch_node->GetInControlAnchor());
  GraphUtils::AddEdge(var_is_initialized_node->GetOutControlAnchor(), ref_switch_node->GetInControlAnchor());
}

/*
 *           netoutput
 *              |
 *            merge
 *           /    \
 *  switch_ref     \
 *      |           \
 * var_is_init--->switch
 *         \    /
 *         const
 */
static void MakeGraphError(ComputeGraphPtr graph) {
  GeTensorDesc ge_tensor_desc;
  GeTensorDesc ge_tensor_desc_var_is;
  ge_tensor_desc_var_is.SetDataType(DT_BOOL);

  auto const_op = std::make_shared<OpDesc>("Const", CONSTANTOP);
  const_op->AddOutputDesc(ge_tensor_desc);
  auto const_node = graph->AddNode(const_op);

  auto var_is_initialized_op = std::make_shared<OpDesc>("VARISINITIALIZEDOP", VARISINITIALIZEDOP);
  var_is_initialized_op->AddInputDesc(ge_tensor_desc);
  var_is_initialized_op->AddOutputDesc(ge_tensor_desc_var_is);
  auto var_is_initialized_node = graph->AddNode(var_is_initialized_op);

  auto switch_op = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddOutputDesc(ge_tensor_desc);
  switch_op->AddOutputDesc(ge_tensor_desc);
  auto switch_node = graph->AddNode(switch_op);

  auto ref_switch_op = std::make_shared<OpDesc>("RefSwitch", REFSWITCH);
  ref_switch_op->AddInputDesc(ge_tensor_desc);
  ref_switch_op->AddOutputDesc(ge_tensor_desc);
  auto ref_switch_node = graph->AddNode(ref_switch_op);

  auto merge_op = std::make_shared<OpDesc>("Merge", MERGE);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddOutputDesc(ge_tensor_desc);
  auto merge_node = graph->AddNode(merge_op);

  auto net_output_op = std::make_shared<OpDesc>("NetOutput", NETOUTPUT);
  net_output_op->AddInputDesc(ge_tensor_desc);
  net_output_op->AddOutputDesc(ge_tensor_desc);
  auto net_output_node = graph->AddNode(net_output_op);

  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), var_is_initialized_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), ref_switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_node->GetOutDataAnchor(1), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_switch_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));
}

/*
 *           netoutput
 *              |
 *            merge
 *           /    \
 *  switch_ref     \
 *      |           \
 * var_is_init--->switch
 *    |     \    /
 *   var     const
 */
static void MakeGraphErrorTwo(ComputeGraphPtr graph) {
  GeTensorDesc ge_tensor_desc;
  GeTensorDesc ge_tensor_desc_var_is;
  ge_tensor_desc_var_is.SetDataType(DT_BOOL);

  auto const_op = std::make_shared<OpDesc>("Const", CONSTANTOP);
  const_op->AddOutputDesc(ge_tensor_desc);
  auto const_node = graph->AddNode(const_op);

  auto variable_op = std::make_shared<OpDesc>("Variable", VARIABLE);
  variable_op->AddOutputDesc(ge_tensor_desc);
  auto variable_node = graph->AddNode(variable_op);

  auto var_is_initialized_op = std::make_shared<OpDesc>("VARISINITIALIZEDOP", VARISINITIALIZEDOP);
  var_is_initialized_op->AddInputDesc(ge_tensor_desc);
  var_is_initialized_op->AddInputDesc(ge_tensor_desc);
  var_is_initialized_op->AddOutputDesc(ge_tensor_desc_var_is);
  auto var_is_initialized_node = graph->AddNode(var_is_initialized_op);

  auto switch_op = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddOutputDesc(ge_tensor_desc);
  auto switch_node = graph->AddNode(switch_op);

  auto ref_switch_op = std::make_shared<OpDesc>("RefSwitch", REFSWITCH);
  ref_switch_op->AddInputDesc(ge_tensor_desc);
  ref_switch_op->AddOutputDesc(ge_tensor_desc);
  auto ref_switch_node = graph->AddNode(ref_switch_op);

  auto merge_op = std::make_shared<OpDesc>("Merge", MERGE);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddOutputDesc(ge_tensor_desc);
  auto merge_node = graph->AddNode(merge_op);

  auto net_output_op = std::make_shared<OpDesc>("NetOutput", NETOUTPUT);
  net_output_op->AddInputDesc(ge_tensor_desc);
  net_output_op->AddOutputDesc(ge_tensor_desc);
  auto net_output_node = graph->AddNode(net_output_op);

  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), var_is_initialized_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), var_is_initialized_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), ref_switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_switch_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));
}

/*
 *           netoutput
 *              |
 *            merge
 *           /    \
 *  switch_ref     \
 *      |           \
 * var_is_init--->switch
 *         \    /
 *          var
 */
static void MakeGraphErrorThree(ComputeGraphPtr graph) {
  GeTensorDesc ge_tensor_desc;
  GeTensorDesc ge_tensor_desc_var_is;
  ge_tensor_desc_var_is.SetDataType(DT_BOOL);

  auto variable_op = std::make_shared<OpDesc>("Variable", VARIABLE);
  variable_op->AddOutputDesc(ge_tensor_desc);
  auto variable_node = graph->AddNode(variable_op);

  auto var_is_initialized_op = std::make_shared<OpDesc>("VARISINITIALIZEDOP", VARISINITIALIZEDOP);
  var_is_initialized_op->AddInputDesc(ge_tensor_desc);
  var_is_initialized_op->AddOutputDesc(ge_tensor_desc_var_is);
  var_is_initialized_op->AddOutputDesc(ge_tensor_desc_var_is);
  auto var_is_initialized_node = graph->AddNode(var_is_initialized_op);

  auto switch_op = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddOutputDesc(ge_tensor_desc);
  auto switch_node = graph->AddNode(switch_op);

  auto ref_switch_op = std::make_shared<OpDesc>("RefSwitch", REFSWITCH);
  ref_switch_op->AddInputDesc(ge_tensor_desc);
  ref_switch_op->AddOutputDesc(ge_tensor_desc);
  auto ref_switch_node = graph->AddNode(ref_switch_op);

  auto merge_op = std::make_shared<OpDesc>("Merge", MERGE);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddOutputDesc(ge_tensor_desc);
  auto merge_node = graph->AddNode(merge_op);

  auto net_output_op = std::make_shared<OpDesc>("NetOutput", NETOUTPUT);
  net_output_op->AddInputDesc(ge_tensor_desc);
  net_output_op->AddOutputDesc(ge_tensor_desc);
  auto net_output_node = graph->AddNode(net_output_op);

  GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), var_is_initialized_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), ref_switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_switch_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));
}

/*
 *
 *             netoutput1
 *              /      \
 *            /         \
 *          /          assign2
 *        /             |     \
 *       |             |      var2
 *       |           merge1
 *       |         /F      \
 *       |    swtich1       \T
 *       |    |     \        \
 *       | const2    \       refswitch1
 *       |            \      /     |
 *    assign1    var_is_init1     |
 *    /    \    /                |
 * const1   var1------------------
 */
static ComputeGraphPtr BuildGraph1() {
  ge::ut::GraphBuilder builder("g1");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto assign1 = builder.AddNode("assign1", "Assign", 2, 1);
  auto var_is_init1 = builder.AddNode("var_is_init1", "VarIsInitializedOp", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 0, 1);
  auto refswitch1 = builder.AddNode("refswitch1", "RefSwitch", 2, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 2, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto assign2 = builder.AddNode("assign2", "Assign", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 2, 0);

  builder.AddDataEdge(const1, 0, assign1, 1);
  builder.AddDataEdge(var1, 0, assign1, 0);
  builder.AddDataEdge(var1, 0, var_is_init1, 0);
  builder.AddDataEdge(var1, 0, refswitch1, 0);
  builder.AddDataEdge(assign1, 0, netoutput1, 0);
  builder.AddDataEdge(var_is_init1, 0, switch1, 1);
  builder.AddDataEdge(var_is_init1, 0, refswitch1, 1);
  builder.AddDataEdge(const2, 0, switch1, 0);
  builder.AddDataEdge(refswitch1, 1, merge1, 1);
  builder.AddDataEdge(switch1, 0, merge1, 0);
  builder.AddDataEdge(var2, 0, assign2, 0);
  builder.AddDataEdge(merge1, 0, assign2, 1);
  builder.AddDataEdge(assign2, 0, netoutput1, 1);

  auto graph = builder.GetGraph();
  graph->TopologicalSorting();
  return graph;
}

/*
 *             netoutput1
 *                  |
 *                 sub1
 *               /      \
 *      assign2           assign3
 *       /    \           /     \
 *   var2    conv1     var3    const4
 *           |   \
 *        addn1 const3
 *       c/   \
 * assign1    add1
 *   |   \   /   \
 * var1 const1   const2
 */
static ComputeGraphPtr BuildGraph2() {
  auto builder = ge::ut::GraphBuilder("g2");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto const2 = builder.AddNode("const2", "Const", 0, 1);
  auto assign1 = builder.AddNode("assign1", "Assign", 2, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto const3 = builder.AddNode("const3", "Const", 0, 1);
  auto conv1 = builder.AddNode("conv1", "Conv2D", 2, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto var3 = builder.AddNode("var3", "Variable", 0, 1);
  auto const4 = builder.AddNode("const4", "Const", 0, 1);
  auto assign2 = builder.AddNode("assign2", "Assign", 2, 1);
  auto assign3 = builder.AddNode("assign3", "Assign", 2, 1);
  auto sub1 = builder.AddNode("sub1", "Sub", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, assign1, 0);
  builder.AddDataEdge(const1, 0, assign1, 1);
  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(const2, 0, add1, 1);
  builder.AddControlEdge(assign1, addn1);
  builder.AddDataEdge(add1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, conv1, 0);
  builder.AddDataEdge(const3, 0, conv1, 1);
  builder.AddDataEdge(var2, 0, assign2, 0);
  builder.AddDataEdge(conv1, 0, assign2, 1);
  builder.AddDataEdge(var3, 0, assign3, 0);
  builder.AddDataEdge(const4, 0, assign3, 1);
  builder.AddDataEdge(assign2, 0, sub1, 0);
  builder.AddDataEdge(assign3, 0, sub1, 1);
  builder.AddDataEdge(sub1, 0, netoutput1, 0);

  return builder.GetGraph();
}

/*
 *
 *             netoutput1
 *              /      \
 *            /         \
 *          /          assign2
 *        /             |     \
 *       |             |      var2
 *       |           merge1
 *       |         /F      \
 *       |    swtich1       \T
 *       |    |     \        \
 *       | const2    \       refswitch1
 *       |            \      /     |
 *     addn1----->var_is_init1     |
 *       |    c     /              |
 *    assign1     /               |
 *    /    \    /                |
 * const1   var1------------------
 */
static ComputeGraphPtr BuildGraph3() {
  ge::ut::GraphBuilder builder("g3");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto assign1 = builder.AddNode("assign1", "Assign", 2, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto var_is_init1 = builder.AddNode("var_is_init1", "VarIsInitializedOp", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 0, 1);
  auto refswitch1 = builder.AddNode("refswitch1", "RefSwitch", 2, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 2, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto assign2 = builder.AddNode("assign2", "Assign", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 2, 0);

  builder.AddDataEdge(const1, 0, assign1, 1);
  builder.AddDataEdge(var1, 0, assign1, 0);
  builder.AddDataEdge(var1, 0, var_is_init1, 0);
  builder.AddDataEdge(var1, 0, refswitch1, 0);
  builder.AddDataEdge(assign1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, netoutput1, 0);
  builder.AddControlEdge(addn1, var_is_init1);
  builder.AddDataEdge(var_is_init1, 0, switch1, 1);
  builder.AddDataEdge(var_is_init1, 0, refswitch1, 1);
  builder.AddDataEdge(const2, 0, switch1, 0);
  builder.AddDataEdge(refswitch1, 1, merge1, 1);
  builder.AddDataEdge(switch1, 0, merge1, 0);
  builder.AddDataEdge(var2, 0, assign2, 0);
  builder.AddDataEdge(merge1, 0, assign2, 1);
  builder.AddDataEdge(assign2, 0, netoutput1, 1);

  auto graph = builder.GetGraph();
  graph->TopologicalSorting();
  return graph;
}

/*
 *
 *             netoutput1
 *              /      \
 *            /         \
 *          /          assign2
 *        /             |     \
 *       |             |      var2
 *       |           merge1
 *       |         /F      \
 *       |    swtich1       \T
 *       |    |     \        \
 *       | const2    \       refswitch1
 *       |      c     \      /     |
 *    assign1 <---var_is_init1     |
 *    /    \    /                |
 * const1   var1------------------
 */
static ComputeGraphPtr BuildGraph4() {
  ge::ut::GraphBuilder builder("g4");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto assign1 = builder.AddNode("assign1", "Assign", 2, 1);
  auto var_is_init1 = builder.AddNode("var_is_init1", "VarIsInitializedOp", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 0, 1);
  auto refswitch1 = builder.AddNode("refswitch1", "RefSwitch", 2, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 2, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto assign2 = builder.AddNode("assign2", "Assign", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 2, 0);

  builder.AddDataEdge(const1, 0, assign1, 1);
  builder.AddDataEdge(var1, 0, assign1, 0);
  builder.AddDataEdge(var1, 0, var_is_init1, 0);
  builder.AddDataEdge(var1, 0, refswitch1, 0);
  builder.AddDataEdge(assign1, 0, netoutput1, 0);
  builder.AddDataEdge(var_is_init1, 0, switch1, 1);
  builder.AddDataEdge(var_is_init1, 0, refswitch1, 1);
  builder.AddControlEdge(var_is_init1, assign1);
  builder.AddDataEdge(const2, 0, switch1, 0);
  builder.AddDataEdge(refswitch1, 1, merge1, 1);
  builder.AddDataEdge(switch1, 0, merge1, 0);
  builder.AddDataEdge(var2, 0, assign2, 0);
  builder.AddDataEdge(merge1, 0, assign2, 1);
  builder.AddDataEdge(assign2, 0, netoutput1, 1);

  auto graph = builder.GetGraph();
  graph->TopologicalSorting();
  return graph;
}

/*
 *           netoutput
 *              |
 *            merge
 *           /    \
 *  switch_ref     \
 *      |           \
 * is_var_init--->switch
 *     |            |
 *    var          const
 */
static void MakeVarInitGraph(ComputeGraphPtr graph) {
  GeTensorDesc ge_tensor_desc;
  GeTensorDesc ge_tensor_desc_var_is;
  ge_tensor_desc_var_is.SetDataType(DT_BOOL);

  auto variable_op = std::make_shared<OpDesc>("Variable", VARIABLE);
  variable_op->AddOutputDesc(ge_tensor_desc);
  auto variable_node = graph->AddNode(variable_op);

  auto const_op = std::make_shared<OpDesc>("const", CONSTANT);
  const_op->AddOutputDesc(ge_tensor_desc);
  auto const_node = graph->AddNode(const_op);

  auto var_is_initialized_op = std::make_shared<OpDesc>("ISVARIABLEINITIALIZED", ISVARIABLEINITIALIZED);
  var_is_initialized_op->AddInputDesc(ge_tensor_desc);
  var_is_initialized_op->AddOutputDesc(ge_tensor_desc_var_is);
  auto var_is_initialized_node = graph->AddNode(var_is_initialized_op);

  auto switch_op = std::make_shared<OpDesc>("Switch", SWITCH);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddInputDesc(ge_tensor_desc);
  switch_op->AddOutputDesc(ge_tensor_desc);
  switch_op->AddOutputDesc(ge_tensor_desc);
  auto switch_node = graph->AddNode(switch_op);

  auto ref_switch_op = std::make_shared<OpDesc>("RefSwitch", REFSWITCH);
  ref_switch_op->AddInputDesc(ge_tensor_desc);
  ref_switch_op->AddInputDesc(ge_tensor_desc);
  ref_switch_op->AddOutputDesc(ge_tensor_desc);
  ref_switch_op->AddOutputDesc(ge_tensor_desc);
  auto ref_switch_node = graph->AddNode(ref_switch_op);

  auto merge_op = std::make_shared<OpDesc>("Merge", MERGE);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddInputDesc(ge_tensor_desc);
  merge_op->AddOutputDesc(ge_tensor_desc);
  merge_op->AddOutputDesc(ge_tensor_desc);
  auto merge_node = graph->AddNode(merge_op);

  auto net_output_op = std::make_shared<OpDesc>("NetOutput", NETOUTPUT);
  net_output_op->AddInputDesc(ge_tensor_desc);
  net_output_op->AddOutputDesc(ge_tensor_desc);
  auto net_output_node = graph->AddNode(net_output_op);

  GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), var_is_initialized_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(var_is_initialized_node->GetOutDataAnchor(0), ref_switch_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), ref_switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ref_switch_node->GetOutDataAnchor(1), merge_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

  GraphUtils::AddEdge(variable_node->GetOutControlAnchor(), var_is_initialized_node->GetInControlAnchor());
  GraphUtils::AddEdge(var_is_initialized_node->GetOutControlAnchor(), switch_node->GetInControlAnchor());
  GraphUtils::AddEdge(var_is_initialized_node->GetOutControlAnchor(), ref_switch_node->GetInControlAnchor());
}

TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, run_success) {
  uint64_t session_id = 0;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  MakeGraphCorrect(graph);
  graph->SetSessionID(session_id);

  ge::NodePtr node_var_is = nullptr;
  
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == VARISINITIALIZEDOP) {
      graphStatus graph_ret = node->SetOwnerComputeGraph(graph);
      EXPECT_EQ(GRAPH_SUCCESS, graph_ret);

      node_var_is = node;
    }
  }
  NodePtr var_node = graph->FindNode("Variable");
  EXPECT_NE(var_node, nullptr);
  VarManager::Instance(session_id)->AssignVarMem(var_node->GetName(), nullptr, var_node->GetOpDesc()->GetOutputDesc(0), 0);

  VarIsInitializedOpPass var_pass;
  Status ret = var_pass.Run(node_var_is);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);

  ge::NodePtr node_switch = nullptr;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == SWITCH) {
      graphStatus graph_ret = node->SetOwnerComputeGraph(graph);
      EXPECT_EQ(GRAPH_SUCCESS, graph_ret);

      node_switch = node;
    }
  }

  SwitchDeadBranchElimination switch_pass;
  ret = switch_pass.Run(node_switch);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);

  ge::NodePtr node_refswitch = nullptr;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == REFSWITCH) {
      graphStatus graph_ret = node->SetOwnerComputeGraph(graph);
      EXPECT_EQ(GRAPH_SUCCESS, graph_ret);

      node_refswitch = node;
    }
  }

  ret = switch_pass.Run(node_refswitch);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  GraphUtils::DumpGEGraph(graph, "004");

  ge::NodePtr node_merge = nullptr;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == MERGE) {
      graphStatus graph_ret = node->SetOwnerComputeGraph(graph);
      EXPECT_EQ(GRAPH_SUCCESS, graph_ret);

      node_merge = node;
    }
  }
  MergePass merge_pass;
  ret = merge_pass.Run(node_merge);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  VarManagerPool::Instance().Destory();
}

TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, change_to_false) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  MakeGraphCorrect(graph);

  ge::NodePtr node_var_is = nullptr;
  ge::NodePtr node_other = nullptr;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == VARISINITIALIZEDOP) {
      graphStatus graph_ret = node->SetOwnerComputeGraph(graph);
      EXPECT_EQ(GRAPH_SUCCESS, graph_ret);

      node_var_is = node;
    } else {
      node_other = node;
    }
  }

  VarIsInitializedOpPass varPass;
  Status ret = varPass.Run(node_var_is);
  EXPECT_EQ(SUCCESS, ret);
  auto const_node = graph->FindNode(node_var_is->GetName());
  EXPECT_NE(const_node, nullptr);
  std::shared_ptr<const GeTensor> tensor = nullptr;
  EXPECT_TRUE(AttrUtils::GetTensor(const_node->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor));
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(*(tensor->GetData().GetData()), 0);

  ret = varPass.Run(node_other);
  EXPECT_EQ(SUCCESS, ret);
  VarManagerPool::Instance().Destory();
}


TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, run_fail_2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  MakeGraphError(graph);

  ge::NodePtr node_var_is = nullptr;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == VARISINITIALIZEDOP) {
      graphStatus graph_ret = node->SetOwnerComputeGraph(graph);
      EXPECT_EQ(GRAPH_SUCCESS, graph_ret);

      node_var_is = node;
    }
  }

  VarIsInitializedOpPass varPass;
  Status ret = varPass.Run(node_var_is);
  EXPECT_EQ(FAILED, ret);
  VarManagerPool::Instance().Destory();
}


TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, run_fail_3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  MakeGraphErrorTwo(graph);

  ge::NodePtr node_var_is = nullptr;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == VARISINITIALIZEDOP) {
      graphStatus graph_ret = node->SetOwnerComputeGraph(graph);
      EXPECT_EQ(GRAPH_SUCCESS, graph_ret);

      node_var_is = node;
    }
  }

  VarIsInitializedOpPass varPass;
  Status ret = varPass.Run(node_var_is);
  EXPECT_EQ(FAILED, ret);
  VarManagerPool::Instance().Destory();
}

TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, run_fail_4) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  MakeGraphErrorThree(graph);

  ge::NodePtr node_var_is = nullptr;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == VARISINITIALIZEDOP) {
      graphStatus graph_ret = node->SetOwnerComputeGraph(graph);
      EXPECT_EQ(GRAPH_SUCCESS, graph_ret);

      node_var_is = node;
    }
  }

  NodePtr var_node = graph->FindNode("Variable");
  EXPECT_NE(var_node, nullptr);
  VarManager::Instance(0)->AssignVarMem(var_node->GetName(), nullptr, var_node->GetOpDesc()->GetOutputDesc(0), 0);

  VarIsInitializedOpPass varPass;
  Status ret = varPass.Run(node_var_is);
  EXPECT_EQ(FAILED, ret);
  VarManagerPool::Instance().Destory();
}


TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, init_before_VarIsInitializedOp) {
  auto graph = BuildGraph3();
  ge::GEPass ge_pass(graph);
  ge::NamesToPass names_to_pass;
  VarIsInitializedOpPass var_pass;
  names_to_pass.emplace_back("VarIsInitializedOpPass", &var_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);

  auto var_is_init1 = graph->FindNode("var_is_init1");
  EXPECT_NE(var_is_init1, nullptr);
  EXPECT_EQ(var_is_init1->GetType(), "Const");
  EXPECT_EQ(var_is_init1->GetInNodes().size(), 2);
  std::set<std::string> var_is_init1_in_ctrl_nodes;
  for (auto &node : var_is_init1->GetInControlNodes()) {
    var_is_init1_in_ctrl_nodes.insert(node->GetName());
  }
  EXPECT_EQ(var_is_init1_in_ctrl_nodes, std::set<std::string>({"addn1", "var1"}));
  std::shared_ptr<const GeTensor> tensor = nullptr;
  EXPECT_TRUE(AttrUtils::GetTensor(var_is_init1->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor));
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(*(tensor->GetData().GetData()), 1);
}

TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, init_after_VarIsInitializedOp) {
  auto graph = BuildGraph4();
  ge::GEPass ge_pass(graph);
  ge::NamesToPass names_to_pass;
  VarIsInitializedOpPass var_pass;
  names_to_pass.emplace_back("VarIsInitializedOpPass", &var_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);

  auto var_is_init1 = graph->FindNode("var_is_init1");
  EXPECT_NE(var_is_init1, nullptr);
  EXPECT_EQ(var_is_init1->GetType(), "Const");
  EXPECT_EQ(var_is_init1->GetInNodes().size(), 1);
  std::set<std::string> nodes;
  for (auto &node : var_is_init1->GetOutDataNodes()) {
    nodes.insert(node->GetName());
  }
  EXPECT_EQ(nodes, std::set<std::string>({"switch1", "refswitch1"}));
  EXPECT_EQ(var_is_init1->GetOutNodes().size(), 3);
  EXPECT_EQ(var_is_init1->GetOutControlNodes().at(0)->GetName(), "assign1");
  EXPECT_EQ(var_is_init1->GetInControlNodes().at(0)->GetName(), "var1");

  std::shared_ptr<const GeTensor> tensor = nullptr;
  EXPECT_TRUE(AttrUtils::GetTensor(var_is_init1->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor));
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(*(tensor->GetData().GetData()), 0);
}

TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, init_parallel_VarIsInitializedOp) {
  auto graph = BuildGraph1();
  ge::GEPass ge_pass(graph);
  ge::NamesToPass names_to_pass;
  VarIsInitializedOpPass var_pass;
  names_to_pass.emplace_back("VarIsInitializedOpPass", &var_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);

  auto var_is_init1 = graph->FindNode("var_is_init1");
  EXPECT_NE(var_is_init1, nullptr);
  EXPECT_EQ(var_is_init1->GetType(), "Const");
  std::shared_ptr<const GeTensor> tensor = nullptr;
  EXPECT_TRUE(AttrUtils::GetTensor(var_is_init1->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor));
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(*(tensor->GetData().GetData()), 0);
  auto assign1 = graph->FindNode("assign1");
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(*var_pass.nodes_to_inited_vars_[assign1->GetOpDesc()->GetId()],
            std::set<int64_t>({var1->GetOpDesc()->GetId()}));
  EXPECT_EQ(var_pass.nodes_to_inited_vars_.count(var1->GetOpDesc()->GetId()), 0);
  auto merge1 = graph->FindNode("merge1");
  EXPECT_EQ(var_pass.nodes_to_inited_vars_.count(merge1->GetOpDesc()->GetId()), 0);
  auto assign2 = graph->FindNode("assign2");
  auto var2 = graph->FindNode("var2");
  EXPECT_EQ(*var_pass.nodes_to_inited_vars_[assign2->GetOpDesc()->GetId()],
            std::set<int64_t>({var2->GetOpDesc()->GetId()}));
  auto netoutput1 = graph->FindNode("netoutput1");
  EXPECT_EQ(*var_pass.nodes_to_inited_vars_[netoutput1->GetOpDesc()->GetId()],
            std::set<int64_t>({var1->GetOpDesc()->GetId(), var2->GetOpDesc()->GetId()}));
}

TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, init_delivering) {
  auto graph = BuildGraph2();
  graph->TopologicalSorting();
  ge::GEPass ge_pass(graph);
  ge::NamesToPass names_to_pass;
  VarIsInitializedOpPass var_pass;
  names_to_pass.emplace_back("VarIsInitializedOpPass", &var_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  std::map<std::string, std::set<std::string>> nodes_to_inited_vars({{"var1", {}},
                                                                     {"const1", {}},
                                                                     {"const2", {}},
                                                                     {"assign1", {"var1"}},
                                                                     {"add1", {}},
                                                                     {"addn1", {"var1"}},
                                                                     {"const3", {}},
                                                                     {"conv1", {"var1"}},
                                                                     {"var2", {}},
                                                                     {"assign2", {"var1", "var2"}},
                                                                     {"var3", {}},
                                                                     {"const4", {}},
                                                                     {"assign3", {"var3"}},
                                                                     {"sub1", {"var1", "var2", "var3"}},
                                                                     {"netoutput1", {"var1", "var2", "var3"}}});

  for (auto &node_to_inited_vars : nodes_to_inited_vars) {
    auto node = graph->FindNode(node_to_inited_vars.first);
    if (node_to_inited_vars.second.empty()) {
      EXPECT_EQ(var_pass.nodes_to_inited_vars_.count(node->GetOpDesc()->GetId()), 0);
      continue;
    }
    std::set<int64_t> expected;
    for (auto &inited_var : node_to_inited_vars.second) {
      expected.insert(graph->FindNode(inited_var)->GetOpDesc()->GetId());
    }
    EXPECT_EQ(*var_pass.nodes_to_inited_vars_[node->GetOpDesc()->GetId()], expected);
  }
}

TEST_F(UTEST_graph_passes_var_is_initialized_op_pass, is_variable_initialized_success) {
  uint64_t session_id = 0;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_is_var_init_graph");
  MakeVarInitGraph(graph);
  graph->SetSessionID(session_id);

  ge::NodePtr node_is_var_init = nullptr;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == ISVARIABLEINITIALIZED) {
      graphStatus graph_ret = node->SetOwnerComputeGraph(graph);
      EXPECT_EQ(GRAPH_SUCCESS, graph_ret);
      node_is_var_init = node;
      break;
    }
  }
  EXPECT_NE(node_is_var_init, nullptr);

  VarIsInitializedOpPass var_pass;
  Status ret = var_pass.Run(node_is_var_init);
  EXPECT_EQ(SUCCESS, ret);

  auto const_node = graph->FindNode(node_is_var_init->GetName());
  EXPECT_NE(const_node, nullptr);
  EXPECT_EQ(const_node->GetType(), "Const");

  std::shared_ptr<const GeTensor> tensor = nullptr;
  EXPECT_TRUE(AttrUtils::GetTensor(const_node->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor));
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(*(tensor->GetData().GetData()), 0);

  VarManagerPool::Instance().Destory();
}
}  // namespace ge
