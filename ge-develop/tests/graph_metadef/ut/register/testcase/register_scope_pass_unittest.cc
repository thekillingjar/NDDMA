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
#include <gmock/gmock.h>
#include <vector>
#include <iostream>
#include "graph_metadef/graph/debug/ge_util.h"

#include "register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"
#include "register/scope/scope_pass_impl.h"
#include "register/scope/scope_pass_registry_impl.h"

using namespace ge;
class UtestScopePass : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestScopePass, ScopesResultImplFail) {
  ScopesResult scopeRstOri;
  scopeRstOri.impl_.reset();

  std::vector<OperatorPtr> nodes;
  scopeRstOri.SetNodes(nodes);

  std::vector<Scope *> scopes;
  scopeRstOri.SetScopes(scopes);

  ScopesResult scopeRst1(scopeRstOri);
  EXPECT_EQ(scopeRst1.impl_->GetScopes().empty(), true);

  ScopesResult scopeRst2;
  scopeRst2 = scopeRst2;
  EXPECT_EQ(scopeRst2.impl_->GetScopes().empty(), true);

  ScopesResult scopeRst3;
  scopeRst3 = scopeRstOri;
  EXPECT_EQ(scopeRst3.impl_->GetScopes().empty(), true);
}

TEST_F(UtestScopePass, ScopesResultRegister) {
  ScopesResult scopeRstOri;
  std::vector<Scope *> scopes;
  std::vector<OperatorPtr> nodes;
  // test for initialized
  std::vector<Scope *> ScopeList = scopeRstOri.impl_->GetScopes();
  EXPECT_EQ(ScopeList.empty(), true);
  std::vector<OperatorPtr> NodeList = scopeRstOri.impl_->GetNodes();
  EXPECT_EQ(NodeList.empty(), true);

  // add scope
  Scope scope1;
  scope1.Init("scope1", "type1");
  scopes.push_back(&scope1);
  Scope scope2;
  scope2.Init("scope2", "type2");
  scopes.push_back(&scope2);
  scopeRstOri.SetScopes(scopes);
  // add node
  OperatorPtr node1(new (std::nothrow) ge::Operator("add", "Add"));
  nodes.push_back(node1);
  OperatorPtr node2(new (std::nothrow) ge::Operator("sub", "Sub"));
  nodes.push_back(node2);
  OperatorPtr node3(new (std::nothrow) ge::Operator("mul", "Mul"));
  nodes.push_back(node3);
  scopeRstOri.SetNodes(nodes);

  ScopesResult scopeRst1(scopeRstOri);
  ScopeList = scopeRst1.impl_->GetScopes();
  EXPECT_EQ(ScopeList.empty(), false);
  NodeList = scopeRst1.impl_->GetNodes();
  EXPECT_EQ(NodeList.empty(), false);

  ScopesResult scopeRst2;
  scopeRst2 = scopeRst1;
  ScopeList = scopeRst2.impl_->GetScopes();
  EXPECT_EQ(ScopeList.empty(), false);
  NodeList = scopeRst2.impl_->GetNodes();
  EXPECT_EQ(NodeList.empty(), false);
}

namespace {
class ScopePass1 : public ScopeBasePass {
public:
  ScopePattern *scoPattern1;
  ScopePattern *scoPattern2;

protected:
  std::vector<ScopeFusionPatterns> DefinePatterns() {
    std::vector<std::vector<std::vector<ScopePattern *>>> scoPattern;
    std::vector<std::vector<ScopePattern *>> scoPatternSub;
    std::vector<ScopePattern *> scoPatternSubSub1;
    std::vector<ScopePattern *> scoPatternSubSub2;

    scoPattern1 = new ScopePattern();
    scoPattern2 = new ScopePattern();
    scoPatternSubSub1.push_back(scoPattern1);
    scoPatternSubSub2.push_back(scoPattern2);

    scoPatternSub.push_back(scoPatternSubSub1);
    scoPatternSub.push_back(scoPatternSubSub2);

    scoPattern.push_back(scoPatternSub);
    return scoPattern;
  }
  std::string PassName() {
    return std::string("passName1");
  }
  Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scope_graph,
                               std::vector<ScopesResult> &results) {
    return SUCCESS;
  }
  void GenerateFusionResult(const std::vector<Scope *> &scopes,
                            FusionScopesResult *fusion_rlt) {
    return;
  }
};

class ScopePass2 : public ScopeBasePass {
protected:
  std::vector<ScopeFusionPatterns> DefinePatterns() {
    std::vector<std::vector<std::vector<ScopePattern *>>> scoPattern;
    return scoPattern;
  }
  std::string PassName() {
    return std::string("passName2");
  }
  Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scope_graph,
                               std::vector<ScopesResult> &results) {
    return FAILED;
  }
  void GenerateFusionResult(const std::vector<Scope *> &scopes,
                            FusionScopesResult *fusion_rlt) {
    return;
  }
};

class ScopePass3 : public ScopeBasePass {
public:
  ScopePattern *scoPattern1;
  ScopePattern *scoPattern2;

protected:
  std::vector<ScopeFusionPatterns> DefinePatterns() {
    std::vector<std::vector<std::vector<ScopePattern *>>> scoPattern;
    std::vector<std::vector<ScopePattern *>> scoPatternSub;
    std::vector<ScopePattern *> scoPatternSubSub1;
    std::vector<ScopePattern *> scoPatternSubSub2;

    scoPattern1 = new ScopePattern();
    scoPattern2 = new ScopePattern();
    scoPatternSubSub1.push_back(scoPattern1);
    scoPatternSubSub2.push_back(scoPattern2);

    scoPatternSub.push_back(scoPatternSubSub1);
    scoPatternSub.push_back(scoPatternSubSub2);

    scoPattern.push_back(scoPatternSub);
    return scoPattern;
  }
  std::string PassName() {
    return std::string("passName1");
  }
  Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scope_graph,
                               std::vector<ScopesResult> &results) {
    return FAILED;
  }
  void GenerateFusionResult(const std::vector<Scope *> &scopes,
                            FusionScopesResult *fusion_rlt) {
    return;
  }
};

class ScopePass4 : public ScopeBasePass {
public:
  ScopePattern *scoPattern1;
  ScopePattern *scoPattern2;

protected:
  std::vector<ScopeFusionPatterns> DefinePatterns() {
    std::vector<std::vector<std::vector<ScopePattern *>>> scoPattern;
    std::vector<std::vector<ScopePattern *>> scoPatternSub;
    std::vector<ScopePattern *> scoPatternSubSub1;
    std::vector<ScopePattern *> scoPatternSubSub2;

    scoPattern1 = new ScopePattern();
    scoPattern2 = new ScopePattern();
    scoPatternSubSub1.push_back(scoPattern1);
    scoPatternSubSub2.push_back(scoPattern2);

    scoPatternSub.push_back(scoPatternSubSub1);
    scoPatternSub.push_back(scoPatternSubSub2);

    scoPattern.push_back(scoPatternSub);
    return scoPattern;
  }
  std::string PassName() {
    return std::string("passName1");
  }
  Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scope_graph,
                               std::vector<ScopesResult> &results) {
    ScopesResult Scope1;
    results.push_back(Scope1);
    return SUCCESS;
  }
  void GenerateFusionResult(const std::vector<Scope *> &scopes,
                            FusionScopesResult *fusion_rlt) {
    fusion_rlt->impl_->SetType(kScopeInvalidType);
    return;
  }
};


void CreateGraph(domi::tensorflow::GraphDef &graph_def) {
  // 1. add node
  auto placeholder0 = graph_def.add_node();
  auto placeholder1 = graph_def.add_node();
  auto add0 = graph_def.add_node();
  auto add1 = graph_def.add_node();
  auto mul0 = graph_def.add_node();
  auto mul1 = graph_def.add_node();
  auto mul2 = graph_def.add_node();
  auto add2 = graph_def.add_node();
  auto retval0 = graph_def.add_node();
  auto retval1 = graph_def.add_node();
  auto retval2 = graph_def.add_node();

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

  mul0->set_name("mul0");
  mul0->set_op("Mul");
  mul1->set_name("mul1");
  mul1->set_op("Mul");
  mul2->set_name("mul1/mul2");
  mul2->set_op("Mul");

  retval0->set_name("retval0");
  retval0->set_op("_RetVal");
  retval1->set_name("retval1");
  retval1->set_op("_RetVal");
  retval2->set_name("retval2");
  retval2->set_op("_RetVal");

  // 3. add edges
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");

  mul0->add_input("placeholder0");
  mul0->add_input("placeholder1");

  mul1->add_input("placeholder0");
  mul1->add_input("add0");
  mul1->add_input("^mul0");

  mul2->add_input("mul0");
  mul2->add_input("add0");

  add1->add_input("mul0");
  add1->add_input("placeholder1");

  add2->add_input("mul1");
  add2->add_input("mul0");

  retval0->add_input("add2:0");
  retval1->add_input("add1:0");
  retval2->add_input("mul2:0");
}
}

TEST_F(UtestScopePass, ScopePassRun1) {
  // no scope match
  Status retStatus;
  domi::tensorflow::GraphDef graph_def;
  CreateGraph(graph_def);

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  retStatus = scope_graph->Init();
  ASSERT_EQ(retStatus, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  ScopePass1 scoBasePass;
  retStatus = scoBasePass.impl_->Run(scope_graph);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestScopePass, ScopePassRun4) {
  // no scope match
  Status retStatus;
  domi::tensorflow::GraphDef graph_def;
  CreateGraph(graph_def);

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  retStatus = scope_graph->Init();
  ASSERT_EQ(retStatus, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  ScopePass4 scoBasePass;
  retStatus = scoBasePass.impl_->Run(scope_graph);
  EXPECT_EQ(retStatus, domi::SCOPE_NOT_CHANGED);
}

TEST_F(UtestScopePass, ScopePassRun2) {
  // MatchAllBatches failed
  Status retStatus;
  domi::tensorflow::GraphDef graph_def;
  CreateGraph(graph_def);

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  retStatus = scope_graph->Init();
  ASSERT_EQ(retStatus, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  ScopePass2 scoBasePass;
  retStatus = scoBasePass.impl_->Run(scope_graph);
  EXPECT_EQ(retStatus, domi::SCOPE_NOT_CHANGED);
}

TEST_F(UtestScopePass, ScopePassRun3) {
  // LastMatchScopesAndOPs failed
  Status retStatus;
  domi::tensorflow::GraphDef graph_def;
  CreateGraph(graph_def);

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  retStatus = scope_graph->Init();
  ASSERT_EQ(retStatus, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  ScopePass3 scoBasePass;
  retStatus = scoBasePass.impl_->Run(scope_graph);
  EXPECT_EQ(retStatus, domi::SCOPE_NOT_CHANGED);
}

TEST_F(UtestScopePass, AddFusionScopesResultToScopeGraph1) {
  Status retStatus;
  std::vector<ScopesResult> scope_results;
  domi::tensorflow::GraphDef graph_def;
  CreateGraph(graph_def);

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  retStatus = scope_graph->Init();
  ASSERT_EQ(retStatus, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  ScopePass1 scoBasePass;
  ScopesResult scopeRst;
  std::vector<Scope *> scopes;
  std::vector<OperatorPtr> nodes;
  // add scope
  Scope scope1;
  scope1.Init("scope1", "type1");
  scopes.push_back(&scope1);
  Scope scope2;
  scope2.Init("scope2", "type2");
  scopes.push_back(&scope2);
  scopeRst.SetScopes(scopes);
  // add node
  OperatorPtr node1(new (std::nothrow) ge::Operator("add", "Add"));
  nodes.push_back(node1);
  OperatorPtr node2(new (std::nothrow) ge::Operator("sub", "Sub"));
  nodes.push_back(node2);
  OperatorPtr node3(new (std::nothrow) ge::Operator("mul", "Mul"));
  nodes.push_back(node3);
  scopeRst.SetNodes(nodes);
  // add scope result
  scope_results.push_back(scopeRst);
  retStatus = scoBasePass.impl_->AddFusionScopesResultToScopeGraph(scope_graph, scope_results);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestScopePass, ScopePassWithWrongInput) {
  ScopePass1 scoBasePass;
  const std::vector<ScopePattern *> patternlist;
  std::vector<Scope *> results;
  bool retBool;
  Status retStatus;

  retBool = scoBasePass.impl_->MatchOneBatch(nullptr, patternlist, results);
  EXPECT_EQ(retBool, false);

  retBool = scoBasePass.impl_->MatchOneScope(nullptr, nullptr, results);
  EXPECT_EQ(retBool, false);

  retBool = scoBasePass.impl_->MatchAllBatches(nullptr, results);
  EXPECT_EQ(retBool, false);

  std::shared_ptr<ScopeGraph> scope_graph;
  scope_graph.reset();
  retStatus = scoBasePass.impl_->PrintFusionScopeInfo(scope_graph);
  EXPECT_EQ(retStatus, PARAM_INVALID);
}

TEST_F(UtestScopePass, MatchOneScope) {
  bool retBool;
  Status retStatus;
  domi::tensorflow::GraphDef graph_def;
  CreateGraph(graph_def);

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  retStatus = scope_graph->Init();
  ASSERT_EQ(retStatus, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  const ScopeTree *scopeTree = scope_graph->GetScopeTree();
  std::vector<Scope *> scopes = scopeTree->impl_->scopes_;

  ScopePattern scoPattern;
  NodeOpTypeFeature feature("nodeType", 0);
  scoPattern.AddNodeOpTypeFeature(feature);

  std::vector<Scope *> results;
  ScopePass1 scoBasePass;
  for (auto scope : scopes)
  {
    retBool = scoBasePass.impl_->MatchOneScope(&scoPattern, scope, results);
    EXPECT_EQ(retBool, false);
  }
}

TEST_F(UtestScopePass, PrintFusionScopeInfo) {
  Status retStatus;
  domi::tensorflow::GraphDef graph_def;
  CreateGraph(graph_def);

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  retStatus = scope_graph->Init();
  ASSERT_EQ(retStatus, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  FusionScopesResult *fusionResult = new (std::nothrow) FusionScopesResult();
  ASSERT_NE(fusionResult, nullptr);
  retStatus = fusionResult->Init();
  EXPECT_EQ(retStatus, SUCCESS);
  // init
  fusionResult->SetName("fusionRstName");
  fusionResult->SetType("fusionRstype");
  fusionResult->SetDescription("fusionRstDesc");
  // add nodes for check fusionOp
  std::vector<OperatorPtr> fusionRstNodes;
  OperatorPtr op1(new (std::nothrow) ge::Operator("Sub", "Sub"));
  fusionRstNodes.push_back(op1);
  OperatorPtr op2(new (std::nothrow) ge::Operator("Mul", "Mul"));
  fusionRstNodes.push_back(op2);
  fusionResult->impl_->AddNodes(fusionRstNodes);
  // insert inputs outputs
  std::vector<int32_t> index_map = {1, 2};
  fusionResult->InsertInputs("Sub", index_map);
  fusionResult->InsertOutputs("Mul", index_map);
  // add scopes
  Scope scope0;
  scope0.Init("scope0", "type0");
  Scope scope1;
  scope1.Init("scope1", "type1");
  std::vector<Scope *> scopes = {&scope0, &scope1};
  fusionResult->impl_->AddScopes(scopes);

  impl->AddFusionScopesResult(fusionResult);

  ScopePass1 scoBasePass;
  retStatus = scoBasePass.impl_->PrintFusionScopeInfo(scope_graph);
  EXPECT_EQ(retStatus, SUCCESS);
}
