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
#include <iostream>
#include "graph/debug/ge_attr_define.h"
#include "register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"
#include "register/scope/scope_pass_impl.h"
#include "register/scope/scope_pass_registry_impl.h"

using namespace ge;
class UtestScopeGraph : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/*    --- sub0, sub1 only for UT test ---
*        placeholder0  placeholder1
*          |       /\  /\       |
*          |      /  \/  \      |
*          |     /   /\   \     |
*          |     |  /  \  |     |
*          |     add0   mul0    |
*          |      /\    c/|\    |
*          |     / sub0 / | \   |
*            mul1 ---- /  |   add1
*              \          |     |\
*               \ ----  add2    | sub1
*                         |     |
*                    retval0 retval1
*/

void CreateGraphDef(domi::tensorflow::GraphDef &graph_def) {
  // 1. add node
  auto placeholder0 = graph_def.add_node();
  auto placeholder1 = graph_def.add_node();
  auto add0 = graph_def.add_node();
  auto add1 = graph_def.add_node();
  auto sub0 = graph_def.add_node();
  auto mul0 = graph_def.add_node();
  auto mul1 = graph_def.add_node();
  auto add2 = graph_def.add_node();
  auto retval0 = graph_def.add_node();
  auto retval1 = graph_def.add_node();

  // 2. set info
  placeholder0->set_name("placeholder0");
  placeholder0->set_op("PlaceHolder");
  placeholder1->set_name("placeholder1");
  placeholder1->set_op("PlaceHolder");

  add0->set_name("add0");
  add0->set_op("Add");
  add1->set_name("add1");
  add1->set_op("Add");
  add2->set_name("add2");
  add2->set_op("Add");
  sub0->set_name("add0/sub0");
  sub0->set_op("Sub");

  mul0->set_name("mul0");
  mul0->set_op("Mul");
  mul1->set_name("mul1");
  mul1->set_op("Mul");

  retval0->set_name("retval0");
  retval0->set_op("_RetVal");
  retval1->set_name("retval1");
  retval1->set_op("_RetVal");

  // 3. add edges
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");
  sub0->add_input("placeholder0");
  sub0->add_input("placeholder1");

  mul0->add_input("placeholder0");
  mul0->add_input("placeholder1");

  mul1->add_input("placeholder0");
  mul1->add_input("add0");
  mul1->add_input("^mul0");

  add1->add_input("mul0");
  add1->add_input("placeholder1");

  add2->add_input("mul1");
  add2->add_input("mul0");

  retval0->add_input("add2:0");
  retval1->add_input("add1:0");
}

TEST_F(UtestScopeGraph, test_build_scope_graph_succ) {
  domi::tensorflow::GraphDef graph_def;

  CreateGraphDef(graph_def);
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  Status ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);
  auto nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 10);

  // checkpoint 1
  auto mul0_iter = nodes_map.find("mul0");
  ASSERT_NE(mul0_iter, nodes_map.end());
  std::vector<std::string> mul0_inputs;
  std::vector<std::string> mul0_outputs;
  mul0_iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_INPUTS, mul0_inputs);
  mul0_iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_OUTPUTS, mul0_outputs);
  ASSERT_EQ(mul0_inputs.size(), 2);
  EXPECT_EQ(mul0_inputs.at(0), "0:placeholder0:0");
  EXPECT_EQ(mul0_inputs.at(1), "1:placeholder1:0");
  ASSERT_EQ(mul0_outputs.size(), 3);
  EXPECT_EQ(mul0_outputs.at(0), "-1:mul1:-1");
  EXPECT_EQ(mul0_outputs.at(1), "0:add1:0");
  EXPECT_EQ(mul0_outputs.at(2), "0:add2:1");

  // checkpoint 2
  auto mul1_iter = nodes_map.find("mul1");
  ASSERT_NE(mul1_iter, nodes_map.end());
  std::vector<std::string> mul1_inputs;
  std::vector<std::string> mul1_outputs;
  mul1_iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_INPUTS, mul1_inputs);
  mul1_iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_OUTPUTS, mul1_outputs);
  ASSERT_EQ(mul1_inputs.size(), 3);
  EXPECT_EQ(mul1_inputs.at(0), "-1:mul0:-1");
  EXPECT_EQ(mul1_inputs.at(1), "0:placeholder0:0");
  EXPECT_EQ(mul1_inputs.at(2), "1:add0:0");
  ASSERT_EQ(mul1_outputs.size(), 1);
  EXPECT_EQ(mul1_outputs.at(0), "0:add2:0");
}

TEST_F(UtestScopeGraph, test_build_scope_graph_node_without_inout) {
  domi::tensorflow::GraphDef graph_def;
  auto no_op = graph_def.add_node();
  no_op->set_name("test_scope/no_op");
  no_op->set_op("NoOp");

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  Status ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  auto nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 1);
  auto iter = nodes_map.find("test_scope/no_op");
  ASSERT_NE(iter, nodes_map.end());
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  graphStatus get_input_attr = iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_INPUTS, inputs);
  graphStatus get_output_attr = iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_OUTPUTS, outputs);
  ASSERT_EQ(get_input_attr, GRAPH_SUCCESS);
  ASSERT_EQ(get_output_attr, GRAPH_SUCCESS);
  EXPECT_EQ(inputs.size(), 0);
  EXPECT_EQ(outputs.size(), 0);
}

TEST_F(UtestScopeGraph, test_build_scope_graph_failed) {
  domi::tensorflow::GraphDef graph_def;
  auto placeholder0 = graph_def.add_node();
  auto placeholder1 = graph_def.add_node();
  auto add0 = graph_def.add_node();

  placeholder0->set_name("placeholder0");
  placeholder0->set_op("PlaceHolder");
  placeholder1->set_name("placeholder1");
  placeholder1->set_op("PlaceHolder");

  add0->set_name("add0");
  add0->set_op("Add");
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  Status ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;

  // 1. input name is invalied
  add0->set_input(0, "placeholder0:invalid:input");
  impl->BuildScopeGraph(&graph_def);
  auto nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 0);

  // 2. index is invalid
  add0->set_input(0, "placeholder0:s1");
  impl->BuildScopeGraph(&graph_def);
  nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 0);

  // 3. index is out of range
  add0->set_input(0, "placeholder0:12356890666666");
  impl->BuildScopeGraph(&graph_def);
  nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 0);

  // index is negative
  add0->set_input(0, "placeholder0:-1");
  impl->BuildScopeGraph(&graph_def);
  nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 0);
}

TEST_F(UtestScopeGraph, IsFusionOpTest) {
  bool retBool;
  Status stat;
  FusionScopesResult *retFusnScopRst;

  domi::tensorflow::GraphDef graph_def;
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  Status ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  // AddFusionScopesResult
  FusionScopesResult *fusionResult = new (std::nothrow) FusionScopesResult();
  ASSERT_NE(fusionResult, nullptr);
  stat = fusionResult->Init();
  EXPECT_EQ(stat, SUCCESS);
  impl->AddFusionScopesResult(nullptr);
  impl->AddFusionScopesResult(fusionResult);

  // add nodes for check FusionOp
  std::vector<OperatorPtr> fusionRstNodes;
  OperatorPtr op1(new (std::nothrow) ge::Operator("addTest", "Add"));
  fusionRstNodes.push_back(op1);
  OperatorPtr op2(new (std::nothrow) ge::Operator("Sub", "Sub"));
  fusionRstNodes.push_back(op2);
  OperatorPtr op3(new (std::nothrow) ge::Operator("Mul", "Mul"));
  fusionRstNodes.push_back(op3);
  fusionResult->impl_->AddNodes(fusionRstNodes);

  // IsFusionOp
  domi::tensorflow::NodeDef *emptyNode = nullptr;
  retBool = impl->IsFusionOp(emptyNode);
  EXPECT_EQ(retBool, false);
  retFusnScopRst = impl->GetFusionScopesResults(emptyNode);
  EXPECT_EQ(retFusnScopRst, nullptr);

  domi::tensorflow::NodeDef *tmpNode = graph_def.add_node();
  tmpNode->set_name("div");
  tmpNode->set_op("Div");
  tmpNode->add_input("placeholder0");
  tmpNode->add_input("placeholder1");
  retBool = impl->IsFusionOp(tmpNode);
  EXPECT_EQ(retBool, false);
  retFusnScopRst = impl->GetFusionScopesResults(tmpNode);
  EXPECT_EQ(retFusnScopRst, nullptr);
  retFusnScopRst = impl->GetFusionScopesResults(std::string("div"));
  EXPECT_EQ(retFusnScopRst, nullptr);

  // IsFusionOpChild
  std::vector<ScopeFusionOpInfo> info_list;
  retBool = impl->IsFusionOpChild(std::string("nodeName"), info_list);
  EXPECT_EQ(retBool, false);
  retBool = impl->IsFusionOpChild(std::string("addTest"), info_list);
  EXPECT_EQ(retBool, true);
  retBool = impl->IsFusionOpChild(std::string("Sub"), info_list);
  EXPECT_EQ(retBool, true);
  retBool = impl->IsFusionOpChild(std::string("Mul"), info_list);
  EXPECT_EQ(retBool, true);

  // FusionOpChildIgnore
  retBool = impl->FusionOpChildIgnore(info_list.front());
  EXPECT_EQ(retBool, true);

  std::vector<int32_t> index_map = {1, 2};
  fusionResult->InsertInputs("Sub", index_map);
  fusionResult->InsertOutputs("Mul", index_map);
  retBool = impl->FusionOpChildIgnore(info_list.back());
  EXPECT_EQ(retBool, false);

  // GetInputOrOutputIndex
  int32_t old_index = -1;
  int32_t new_index = 2;
  stat = impl->GetInputOrOutputIndex(info_list.front(), old_index, true, new_index);
  EXPECT_EQ(new_index, -1);
  EXPECT_EQ(stat, SUCCESS);

  old_index = 666;
  stat = impl->GetInputOrOutputIndex(info_list.front(), old_index, true, new_index);
  EXPECT_EQ(new_index, kFusionDisableIndex);
  EXPECT_EQ(stat, SUCCESS);

  old_index = 1;
  stat = impl->GetInputOrOutputIndex(info_list.back(), old_index, true, new_index);
  EXPECT_EQ(new_index, kFusionDisableIndex);
  EXPECT_EQ(stat, SUCCESS);
}

TEST_F(UtestScopeGraph, IsFusionOpTest_Fail) {
  // AddFusionScopesResult
  FusionScopesResult *fusionResult = new (std::nothrow) FusionScopesResult();
  ASSERT_NE(fusionResult, nullptr);
  std::string name = "name";
  fusionResult->SetName(name);
  fusionResult->SetName(name.c_str());
  fusionResult->SetType(name);
  fusionResult->SetType(name.c_str());
  fusionResult->SetDescription(name);
  fusionResult->SetDescription(name.c_str());
  EXPECT_EQ(fusionResult->Name(), "");
  AscendString as;
  EXPECT_EQ(fusionResult->Name(as), ge::GRAPH_PARAM_INVALID);
  EXPECT_EQ(fusionResult->Nodes().empty(), true);
  fusionResult->InsertInputs(name, {});
  fusionResult->InsertInputs(name.c_str(), {});
  fusionResult->InsertOutputs(name, {});
  delete fusionResult;
}

TEST_F(UtestScopeGraph, ScopeImplAddNodesTest) {
  Status ret;
  domi::tensorflow::GraphDef graph_def;
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);
  const ScopeTree *scopeTree = scope_graph->GetScopeTree();

  OperatorPtr nodeDef1 = nullptr;
  scopeTree->GetAllScopes().front()->impl_->AddNode(nodeDef1);
  scopeTree->impl_->AddNodeToScope(nodeDef1);

  OperatorPtr nodeDef2(new (std::nothrow) ge::Operator("add0/sub0", "Add"));
  scopeTree->GetAllScopes().front()->impl_->AddNode(nodeDef2);
  scopeTree->impl_->AddNodeToScope(nodeDef2);
  EXPECT_EQ(scopeTree->GetAllScopes().empty(), false);

  std::unordered_map<AscendString, ge::OperatorPtr> node_map;
  scopeTree->GetAllScopes().front()->AllNodesMap();
  ret = scopeTree->GetAllScopes().front()->AllNodesMap(node_map);
  EXPECT_EQ(ret, ge::SUCCESS);

  std::vector<Scope *> scopes = scopeTree->GetAllScopes().front()->impl_->GetAllSubScopes();
  EXPECT_EQ(scopes.empty(), false);
}

TEST_F(UtestScopeGraph, GetScopeLastName) {
  Status ret;
  domi::tensorflow::GraphDef graph_def;
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);
  const ScopeTree *scopeTree = scope_graph->GetScopeTree();
  scopeTree->GetAllScopes().front()->LastName();
  AscendString name;
  ret = scopeTree->GetAllScopes().front()->LastName(name);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestScopeGraph, TrimScopeIndex) {
  domi::tensorflow::GraphDef graph_def;
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  Status ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);
  const ScopeTree *scopeTree = scope_graph->GetScopeTree();

  std::string scope_str = "scope_str_2";
  std::string retStr1 = scopeTree->GetAllScopes().front()->impl_->TrimScopeIndex(scope_str);
  EXPECT_EQ(retStr1 == scope_str, false);

  scope_str = "scope_str_9223372036854775807";
  std::string retStr2 = scopeTree->GetAllScopes().front()->impl_->TrimScopeIndex(scope_str);
  EXPECT_EQ(retStr2 == scope_str, true);

  scope_str = "scope_str_";
  std::string retStr3 = scopeTree->GetAllScopes().front()->impl_->TrimScopeIndex(scope_str);
  EXPECT_EQ(retStr3 == scope_str, true);

  scope_str = "scope_66666";
  std::string retStr4 = scopeTree->GetAllScopes().front()->impl_->TrimScopeIndex(scope_str);
  EXPECT_EQ(retStr4 == scope_str, false);
}

TEST_F(UtestScopeGraph, ScopeImplOpTypeTest) {
  int retInt;
  domi::tensorflow::GraphDef graph_def;
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  Status ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);
  const ScopeTree *scopeTree = scope_graph->GetScopeTree();

  const std::string op_type1 = "Add";
  const std::string op_type2 = "Mul666";
  retInt = scopeTree->GetAllScopes().front()->impl_->GetOpTypeNum(op_type1);
  EXPECT_EQ(retInt, -1);
  retInt = scopeTree->GetAllScopes().front()->impl_->GetOpTypeNum(std::string("type1"));
  EXPECT_EQ(retInt, -1);

  scopeTree->GetAllScopes().front()->impl_->OpsNumInc(op_type1);
  scopeTree->GetAllScopes().front()->impl_->OpsNumInc(op_type1);
  scopeTree->GetAllScopes().front()->impl_->OpsNumInc(op_type2);
  retInt = scopeTree->GetAllScopes().front()->impl_->GetOpTypeNum(op_type1);
  EXPECT_EQ(retInt, 2);
  retInt = scopeTree->GetAllScopes().front()->impl_->GetOpTypeNum(op_type2);
  EXPECT_EQ(retInt, 1);

  scopeTree->GetAllScopes().front()->impl_->ClearTypeAndSubType();
  const std::vector<Scope *> &sub_scopes = scopeTree->GetAllScopes().front()->impl_->GetAllSubScopes();
  for (auto &sub_scope : sub_scopes) {
    std::string type = sub_scope->SubType();
    EXPECT_EQ(type == "", true);
  }
}

TEST_F(UtestScopeGraph, scopeGraphInit) {
  Status ret;
  domi::tensorflow::GraphDef graph_def;
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);
  const ScopeTree *scopeTree = scope_graph->GetScopeTree();

  // init
  const char_t *name = "init_name";
  const char_t *sub_type = "sub_type";
  scopeTree->GetAllScopes().front()->Init(name, sub_type, nullptr);

  // Name
  AscendString a_name;
  ret = scopeTree->GetAllScopes().front()->Name(a_name);
  EXPECT_EQ(ret, ge::SUCCESS);

  // SubType
  scopeTree->GetAllScopes().front()->SubType();
  AscendString type;
  ret = scopeTree->GetAllScopes().front()->SubType(type);
  EXPECT_EQ(ret, ge::SUCCESS);

  // GetScope
  const Scope *scope1 = scopeTree->GetAllScopes().front()->GetSubScope(std::string("Add"));
  EXPECT_EQ(scope1, nullptr);
  // Used to test function overloading
  const Scope *scope2 = scopeTree->GetAllScopes().front()->GetSubScope("Add");
  EXPECT_EQ(scope2, nullptr);

  const Scope *scope3 = scopeTree->GetAllScopes().front()->GetFatherScope();
  EXPECT_EQ(scope3, nullptr);

  std::vector<Scope *> scopes = scopeTree->GetAllScopes().front()->GetAllSubScopes();
  EXPECT_EQ(scopes.empty(), true);

  // GetNodesMap
  std::unordered_map<AscendString, ge::OperatorPtr> nodes_map;
  scope_graph->GetNodesMap();
  ret = scope_graph->GetNodesMap(nodes_map);
  EXPECT_EQ(ret, ge::SUCCESS);
}

class UtestFusionScope : public testing::Test {
 public:
  domi::tensorflow::GraphDef graph_def;
  std::shared_ptr<ScopeGraph> scope_graph;
  FusionScopesResult *fusion_rlt0;
  FusionScopesResult *fusion_rlt;

 protected:
  void SetUp() {
    Status ret;
    CreateGraphDef(graph_def);

    scope_graph = std::make_shared<ScopeGraph>();
    ASSERT_NE(scope_graph, nullptr);
    ret = scope_graph->Init();
    ASSERT_EQ(ret, ge::SUCCESS);

    fusion_rlt0 = new (std::nothrow) FusionScopesResult();
    fusion_rlt = new (std::nothrow) FusionScopesResult();
    ASSERT_NE(fusion_rlt, nullptr);
    ret = fusion_rlt->Init();
    ASSERT_EQ(ret, ge::SUCCESS);
  }
  void TearDown() {
    delete fusion_rlt0;
    delete fusion_rlt;
  }
};

TEST_F(UtestFusionScope, FusionScopesResultSetInfo) {
  fusion_rlt->SetName(std::string("fusionRstName1"));
  EXPECT_EQ(fusion_rlt->Name() == "fusionRstName1", true);

  fusion_rlt->SetName("fusionRstName2");
  const std::string fsnName1 = fusion_rlt->Name();
  AscendString fsnName2;
  fusion_rlt->Name(fsnName2);
  EXPECT_EQ(strncmp(fsnName2.GetString(), "fusionRstName2", strlen("fusionRstName2")), false);

  fusion_rlt->SetType(std::string("fusionRstype1"));
  fusion_rlt->SetType("fusionRstype2");

  fusion_rlt->SetDescription(std::string("fusionRstDesc1"));
  fusion_rlt->SetDescription("fusionRstDesc2");
  EXPECT_EQ(fusion_rlt->impl_->Description() == "fusionRstDesc2", true);
}

TEST_F(UtestFusionScope, FusionScopesResultInnerNodeInfo) {
  fusion_rlt->SetName("fusionRstName");
  fusion_rlt->SetType("fusionRstype");
  fusion_rlt->SetDescription("fusionRstDesc");

  std::vector<int32_t> index_map(6);
  index_map.push_back(1);
  index_map.push_back(2);
  index_map.push_back(5);
  index_map.push_back(6);

  fusion_rlt->InsertInputs(std::string("innerIOpName1"), index_map);
  fusion_rlt->InsertInputs("innerIOpName2", index_map);

  fusion_rlt->InsertOutputs(std::string("innerOOpName1"), index_map);
  fusion_rlt->InsertOutputs("innerOOpName2", index_map);

  FusionScopesResult::InnerNodeInfo *InnerNode1;
  fusion_rlt0->AddInnerNode(std::string("InnerNodeName1"), std::string("InnerNodeType1"));
  InnerNode1 = fusion_rlt->AddInnerNode("InnerNodeName1", "InnerNodeType1");
  (void)InnerNode1;
  FusionScopesResult::InnerNodeInfo *InnerNode2;
  fusion_rlt0->AddInnerNode(std::string("InnerNodeName2"), std::string("InnerNodeType2"));
  InnerNode2 = fusion_rlt->AddInnerNode("InnerNodeName2", "InnerNodeType2");
  InnerNode2->InsertOutput(kOutputToFusionScope, 0);
  FusionScopesResult::InnerNodeInfo *retInnerNodeInfo;
  retInnerNodeInfo = fusion_rlt0->MutableRecentInnerNode();
  EXPECT_EQ(retInnerNodeInfo, nullptr);
  retInnerNodeInfo = fusion_rlt->MutableRecentInnerNode();
  EXPECT_NE(retInnerNodeInfo, nullptr);

  retInnerNodeInfo = fusion_rlt0->MutableInnerNode(1);
  EXPECT_EQ(retInnerNodeInfo, nullptr);
  retInnerNodeInfo = fusion_rlt->MutableInnerNode(1);
  EXPECT_NE(retInnerNodeInfo, nullptr);
  ge::graphStatus retGraphStat;
  retGraphStat = fusion_rlt0->CheckInnerNodesInfo();
  EXPECT_EQ(retGraphStat, ge::GRAPH_PARAM_INVALID);
  retGraphStat = fusion_rlt->CheckInnerNodesInfo();
  EXPECT_EQ(retGraphStat, ge::GRAPH_PARAM_INVALID);

  FusionInnerNodesInfo nodes_info = fusion_rlt->impl_->GetInnerNodesInfo();
  EXPECT_EQ(nodes_info.empty(), false);
}

TEST_F(UtestFusionScope, FusionScopesResultCheckInnerNodesInfo) {
  fusion_rlt->SetName("CheckInnerNodesInfo");
  fusion_rlt->SetType("CheckInnerNodesInfo");
  fusion_rlt->SetDescription("CheckInnerNodesInfo");

  std::vector<int32_t> index_map{0};
  const std::string input_name("inputop");
  const std::string output_name("outputop");
  fusion_rlt->InsertInputs(input_name, index_map);
  fusion_rlt->InsertOutputs(output_name, index_map);
  auto InnerNode = fusion_rlt->AddInnerNode("InnerNodeName", "InnerNodeType");
  InnerNode->InsertInput(kInputFromFusionScope, 0);
  InnerNode->InsertOutput(kOutputToFusionScope, 0);
  // check
  EXPECT_EQ(fusion_rlt->CheckInnerNodesInfo(), ge::GRAPH_SUCCESS);

  FusionInnerNodesInfo nodes_info = fusion_rlt->impl_->GetInnerNodesInfo();
  //check
  EXPECT_EQ(nodes_info.size(), 1U);

  const auto name =  std::get<0>(nodes_info[0U]); 
  const auto type =  std::get<1>(nodes_info[0U]); 
  const auto inputs =  std::get<2>(nodes_info[0U]); 
  const auto outputs =  std::get<3>(nodes_info[0U]); 
  const auto op =  std::get<4>(nodes_info[0U]); 

  EXPECT_EQ(name, "CheckInnerNodesInfo/InnerNodeName");
  EXPECT_EQ(type, "InnerNodeType");
  EXPECT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0U].first, kInputFromFusionScope);
  EXPECT_EQ(inputs[0U].second, 0);
  EXPECT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0U].first, kOutputToFusionScope);
  EXPECT_EQ(outputs[0U].second, 0);
  EXPECT_NE(op, nullptr);
}

TEST_F(UtestFusionScope, InnerNodeInit) {
  FusionScopesResult::InnerNodeInfo InnerNode1(std::string("FusionNode1"));
  InnerNode1.SetName(std::string("InnerAdd"));
  InnerNode1.SetType(std::string("Add"));
  InnerNode1.InsertInput(std::string("Input1"), 1);
  InnerNode1.InsertOutput(std::string("Output1"), 11);

  FusionScopesResult::InnerNodeInfo InnerNode2("FusionNode2");
  InnerNode2.SetName("InnerSub");
  InnerNode2.SetType("Sub");
  InnerNode2.InsertInput("Input2", 2);
  InnerNode2.InsertOutput("Output2", 22);

  FusionScopesResult::InnerNodeInfo InnerNode3("FusionNodeName3", "NodeName3", "NodeType3");

  EXPECT_NE(InnerNode1.BuildInnerNode(), ge::GRAPH_PARAM_INVALID);
  EXPECT_NE(InnerNode1.MutableOperator(), nullptr);

  std::string InnerNodeInfoStr;
  AscendString InnerNodeInfoAscendStr;
  //name
  InnerNode1.GetName();
  graphStatus status = InnerNode1.GetName(InnerNodeInfoAscendStr);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  // type
  InnerNode1.GetType();
  status = InnerNode1.GetType(InnerNodeInfoAscendStr);
  ASSERT_EQ(status, GRAPH_SUCCESS);

  std::vector<std::pair<std::string, int32_t>> pairList;
  std::vector<std::pair<AscendString, int32_t>> pairAscendList;
  pairList = InnerNode1.GetInputs();
  status = InnerNode1.GetInputs(pairAscendList);
  ASSERT_EQ(status, GRAPH_SUCCESS);

  pairList = InnerNode1.GetOutputs();
  status = InnerNode1.GetOutputs(pairAscendList);
  ASSERT_EQ(status, GRAPH_SUCCESS);
}

TEST_F(UtestFusionScope, InnerNodeSetIOFormat) {
  graphStatus retGraphStat;

  FusionScopesResult::InnerNodeInfo InnerNode(std::string("FusionNode"));
  InnerNode.SetName(std::string("InnerAdd"));
  InnerNode.SetType(std::string("Add"));
  InnerNode.InsertInput(std::string("Input"), 1);
  EXPECT_NE(InnerNode.BuildInnerNode(), ge::GRAPH_PARAM_INVALID);
  EXPECT_NE(InnerNode.MutableOperator(), nullptr);
  // Used to test function overloading
  retGraphStat = InnerNode.SetInputFormat(std::string("InputName1"), std::string("InputFormat1"));
  EXPECT_NE(retGraphStat, ge::GRAPH_PARAM_INVALID);
  retGraphStat = InnerNode.SetInputFormat("InputName2", "InputFormat2");
  EXPECT_NE(retGraphStat, ge::GRAPH_PARAM_INVALID);

  retGraphStat = InnerNode.SetOutputFormat(std::string("OutputName1"), std::string("OutputFormat1"));
  EXPECT_NE(retGraphStat, ge::GRAPH_PARAM_INVALID);
  retGraphStat = InnerNode.SetOutputFormat("OutputName2", "OutputFormat2");
  EXPECT_NE(retGraphStat, ge::GRAPH_PARAM_INVALID);

  retGraphStat =
      InnerNode.SetDynamicInputFormat(std::string("DynamicInputName1"), 0, std::string("DynamicInputFormat1"));
  EXPECT_NE(retGraphStat, ge::GRAPH_PARAM_INVALID);
  retGraphStat = InnerNode.SetDynamicInputFormat("DynamicInputName2", 1, "DynamicInputFormat2");
  EXPECT_NE(retGraphStat, ge::GRAPH_PARAM_INVALID);

  retGraphStat =
      InnerNode.SetDynamicOutputFormat(std::string("DynamicOutputName1"), 0, std::string("DynamicOutputFormat1"));
  EXPECT_NE(retGraphStat, ge::GRAPH_PARAM_INVALID);
  retGraphStat = InnerNode.SetDynamicOutputFormat("DynamicOutputName2", 1, "DynamicOutputFormat2");
  EXPECT_NE(retGraphStat, ge::GRAPH_PARAM_INVALID);
}
