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

#include "macro_utils/dt_public_scope.h"
#include "graph/node.h"
#include "graph/passes/variable_optimize/variable_prepare_op_pass.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;

class UtestGraphPassesVariablePreparePass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

/*
   variable    const
       \      /
        assign
*/
static ComputeGraphPtr BuildGraphVariablePreparePass() {
  DEF_GRAPH(variablePreparePassGraph) {
    CHAIN(NODE("variable", VARIABLE)->EDGE(0U, 0U)->NODE("assign", ASSIGN));
    CHAIN(NODE("const", CONSTANT)->EDGE(0U, 1U)->NODE("assign", ASSIGN));
    CHAIN(NODE("variable", CONSTANT)->EDGE(0U, 0U)->NODE("framworkop", FRAMEWORKOP));
    CHAIN(NODE("variable", VARIABLE)->EDGE(0U, 0U)->NODE("variable1", VARIABLE));
    CHAIN(NODE("variable1", VARIABLE)->EDGE(0U, 0U)->NODE("netoutput", NETOUTPUT));
  };

  const auto graph = ToGeGraph(variablePreparePassGraph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();

  // set the same name for in/output name for variable
  const map<string, uint32_t> name_index = {{"var", 0}};
  auto var = compute_graph->FindNode("variable");
  auto var_op_desc = var->GetOpDesc();
  var_op_desc->UpdateInputName(name_index);
  var_op_desc->UpdateOutputName(name_index);

  // set the same name for in/output name for variable1
  var = compute_graph->FindNode("variable1");
  var_op_desc = var->GetOpDesc();
  var_op_desc->UpdateInputName(name_index);
  var_op_desc->UpdateOutputName(name_index);

  // set the same name for in/output name for netoutput
  var = compute_graph->FindNode("netoutput");
  var_op_desc = var->GetOpDesc();
  var_op_desc->UpdateInputName(name_index);

  // set the same name for in/output name for assign
  var = compute_graph->FindNode("assign");
  var_op_desc = var->GetOpDesc();
  const map<string, uint32_t> name_index0 = {{"var", 0}};
  var_op_desc->UpdateInputName(name_index0);
  const map<string, uint32_t> name_index1 = {{"var", 0}};
  var_op_desc->UpdateInputName(name_index1);
  var_op_desc->UpdateOutputName(name_index0);

  auto fmk_op = compute_graph->FindNode("framworkop");
  (void)AttrUtils::SetStr(fmk_op->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, ASSIGN);

  return compute_graph;
}

/*
                    data
*     refdata         |
        |             |
           \         /
              assign
                 |
                 |
              netoutput

*/
static ComputeGraphPtr BuildRefDataGraph() {
  auto data_op = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("data");
  auto refdata_op = OP_CFG("RefData")
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("refdata");
  DEF_GRAPH(RefDataGraph) {
    CHAIN(NODE(refdata_op)->EDGE(0U, 0U)->NODE("assign", ASSIGN));
    CHAIN(NODE(data_op)->EDGE(0U, 1U)->NODE("assign"));
    CHAIN(NODE("assign")->NODE("netoutput", NETOUTPUT));
  };

  const auto graph = ToGeGraph(RefDataGraph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();

  // set the same name for in/output name for refdata
  const map<string, uint32_t> name_index = {{"x", 0}};
  auto refdata = compute_graph->FindNode("refdata");
  auto refdata_op_desc = refdata->GetOpDesc();
  EXPECT_TRUE(refdata_op_desc->UpdateInputName(name_index));
  EXPECT_TRUE(refdata_op_desc->UpdateOutputName(name_index));

  // set the same name for in/output name for netoutput
  auto netoutput = compute_graph->FindNode("netoutput");
  auto netoutput_op_desc = netoutput->GetOpDesc();
  EXPECT_TRUE(netoutput_op_desc->UpdateInputName(name_index));

  // set the same name for in/output name for assign
  auto assign = compute_graph->FindNode("assign");
  auto assign_op_desc = assign->GetOpDesc();
  const map<string, uint32_t> name_index0 = {{"ref", 0}, {"value", 1}};
  EXPECT_TRUE(assign_op_desc->UpdateInputName(name_index0));
  const map<string, uint32_t> name_index1 = {{"ref", 0}};
  EXPECT_TRUE(assign_op_desc->UpdateOutputName(name_index1));
  return compute_graph; 
}


static ComputeGraphPtr BuildControlOpGraph(bool is_set = true) {
  DEF_GRAPH(ControlOpGraph) {
    CHAIN(NODE("variable", VARIABLE)->EDGE(0U, 0U)->NODE("refswitch", FRAMEWORKOP));
    CHAIN(NODE("const0", CONSTANT)->EDGE(0U, 1U)->NODE("refswitch", FRAMEWORKOP));
    CHAIN(NODE("refswitch", FRAMEWORKOP)->EDGE(0U, 0U)->NODE("assign", ASSIGN));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0U, 1U)->NODE("assign", ASSIGN));
    CHAIN(NODE("assign", ASSIGN)->EDGE(0U, 0U)->NODE("refmerge", FRAMEWORKOP));
    CHAIN(NODE("refmerge", FRAMEWORKOP)->EDGE(0U, 0U)->NODE("netoutput", NETOUTPUT));
  };

  const auto graph = ToGeGraph(ControlOpGraph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();

  // set the same name for in/output name for variable
  const map<string, uint32_t> name_index = {{"var", 0}};
  auto var = compute_graph->FindNode("variable");
  auto var_op_desc = var->GetOpDesc();
  var_op_desc->UpdateInputName(name_index);
  var_op_desc->UpdateOutputName(name_index);

  // set the same name for in/output name for assign
  auto assign = compute_graph->FindNode("assign");
  auto assign_op_desc = assign->GetOpDesc();
  const map<string, uint32_t> name_index1 = {{"ref", 0}, {"value", 1}};
  assign_op_desc->UpdateInputName(name_index1);
  assign_op_desc->UpdateOutputName(name_index1);

  auto refswitch = compute_graph->FindNode("refswitch");
  (void)AttrUtils::SetStr(refswitch->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, REFSWITCH);

  if (is_set) {
    auto refmerge = compute_graph->FindNode("refmerge");
    (void)AttrUtils::SetStr(refmerge->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, REFMERGE);
  }

  return compute_graph;
}

TEST_F(UtestGraphPassesVariablePreparePass, variable_prepare_pass_succ1) {
  auto graph = BuildGraphVariablePreparePass();
  ge::VariablePrepareOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(SUCCESS, status);

  // InsertVariableRef
  auto cur_node = graph->FindNode("variable1");
  auto var_node = graph->FindNode("netoutput");
  status = pass_.InsertVariableRef(cur_node, 0U, var_node);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesVariablePreparePass, get_peer_node_failed) {
  auto graph = BuildGraphVariablePreparePass();
  VariablePrepareOpPass pass_;

  // invalid input failed
  auto cur_node = graph->FindNode("variable1");
  std::stack<std::pair<NodePtr, std::pair<int32_t, int32_t>>> nodes;
  ge::Status status = pass_.GetPeerNodeOfRefOutput(cur_node, -1, nodes);
  EXPECT_EQ(PARAM_INVALID, status);

  status = pass_.GetPeerNodeOfRefOutput(cur_node, 1U, nodes);
  EXPECT_EQ(SUCCESS, status);

  status = pass_.GetPeerNodeOfRefOutput(cur_node, 2U, nodes);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesVariablePreparePass, add_variable_ref) {
  auto graph = BuildGraphVariablePreparePass();
  VariablePrepareOpPass pass_;

  // invalid input failed
  auto cur_node = graph->FindNode("variable");
  auto var_node = graph->FindNode("variable1");
  ge::Status status = pass_.AddVariableRef(cur_node, var_node, 3U);
  EXPECT_EQ(SUCCESS, status);

  OpDescPtr pdesc = var_node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(pdesc, REF_VAR_SRC_VAR_NAME, "src_var_name");
  status = pass_.AddVariableRef(cur_node, var_node, 0U);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesVariablePreparePass, create_ref_identity_failed) {
  VariablePrepareOpPass pass_;
  // invalid input failed
  const NodePtr node = shared_ptr<ge::Node>(new (std::nothrow) ge::Node());
  auto ptr = pass_.CreateRefIdentity("src_var_name", node, 0U);
  EXPECT_EQ(nullptr, ptr);
}

TEST_F(UtestGraphPassesVariablePreparePass, create_var_ref_failed) {
  VariablePrepareOpPass pass_;
  // invalid input failed
  const NodePtr node = shared_ptr<ge::Node>(new (std::nothrow) ge::Node());
  auto ptr = pass_.CreateVariableRef("src_var_name", node);
  EXPECT_EQ(nullptr, ptr);
}

TEST_F(UtestGraphPassesVariablePreparePass, GetWritableNodeOutIndex) {
  VariablePrepareOpPass pass_;
  // invalid input failed
  std::vector<int32_t> vec;
  EXPECT_NO_THROW(pass_.GetWritableNodeOutIndex(nullptr, 0U, vec));
}

TEST_F(UtestGraphPassesVariablePreparePass, generate_ref_type_and_inputoutput_map_failed) {
  VariablePrepareOpPass pass_;
  // invalid input failed
  const NodePtr node = shared_ptr<ge::Node>(new (std::nothrow) ge::Node());
  EXPECT_NO_THROW(pass_.GenerateRefTypeAndInputOutputMap(node));
}

TEST_F(UtestGraphPassesVariablePreparePass, variable_prepare_pass_control_op) {
  auto graph_f = BuildControlOpGraph(false);
  ge::VariablePrepareOpPass pass_f_;
  ge::Status status = pass_f_.Run(graph_f);
  EXPECT_EQ(FAILED, status);

  auto graph = BuildControlOpGraph();
  ge::VariablePrepareOpPass pass_;
  status = pass_.Run(graph);
  EXPECT_EQ(SUCCESS, status);

  auto identity_node = graph->FindNode("Identity_assign_TO_refmerge_0");
  EXPECT_NE(identity_node, nullptr);
}

//                    RefDataGraph

// ┌──────┐  (0,1)   ┌─────────┐  (0,0)   ┌───────────┐
// │ data │ ───────> │ assign  │ ───────> │ netoutput │
// └──────┘          └─────────┘          └───────────┘
//                     ∧
//                     │ (0,0)
//                     │
//                   ┌─────────┐
//                   │ refdata │
//                   └─────────┘

TEST_F(UtestGraphPassesVariablePreparePass, add_variable_ref_for_refdata) {
  auto graph = BuildRefDataGraph();  
  VariablePrepareOpPass pass;
  auto status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);

  // check ref attr on var_ref
  auto refdata = graph->FindNode("refdata");
  auto ref_of_refdata = graph->FindNode("refdata_TO_assign_REF_0");
  std::string src_var_name;
  (void)ge::AttrUtils::GetStr(ref_of_refdata->GetOpDesc(), REF_VAR_SRC_VAR_NAME, src_var_name);
  EXPECT_STREQ(src_var_name.c_str(), refdata->GetName().c_str());
}
