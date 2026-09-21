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
#include <nlohmann/json.hpp>

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "pass_manager.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/swap_merge_cast_fusion_pass.h"

using namespace ge;
using namespace fe;

class fusion_pass_swap_merge_cast_ut : public testing::Test {
protected:
  void SetUp() {
    REGISTER_PASS("SwapMergeCastFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, SwapMergeCastFusionPass);
  }

  void TearDown() {

  }

protected:
  static ComputeGraphPtr CreateSwapMergeCastGraph1()
  {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_switch = std::make_shared<OpDesc>("switch", "Switch");
    OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    OpDescPtr op_desc_merge = std::make_shared<OpDesc>("merge", "Merge");
    OpDescPtr op_desc_cast = std::make_shared<OpDesc>("cast", "Cast");
    OpDescPtr op_desc_netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");
    OpDescPtr op_desc_other = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT16);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {8, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT16);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    GeTensorDesc tensor_desc_d(shape_c);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_e;
    GeShape shape_e(dim_e);
    GeTensorDesc tensor_desc_e(shape_e);
    tensor_desc_e.SetFormat(FORMAT_ND);
    tensor_desc_e.SetOriginFormat(FORMAT_ND);
    tensor_desc_e.SetDataType(DT_INT32);
    tensor_desc_e.SetOriginDataType(DT_INT32);

    op_desc_switch->AddOutputDesc(tensor_desc_a);
    op_desc_switch->AddOutputDesc(tensor_desc_b);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_a);

    op_desc_relu2->AddInputDesc(tensor_desc_b);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_merge->AddInputDesc(tensor_desc_a);
    op_desc_merge->AddInputDesc(tensor_desc_b);
    op_desc_merge->AddOutputDesc(tensor_desc_c);
    op_desc_merge->AddOutputDesc(tensor_desc_e);

    op_desc_other->AddInputDesc(tensor_desc_e);

    op_desc_cast->AddInputDesc(tensor_desc_c);
    op_desc_cast->AddOutputDesc(tensor_desc_d);

    op_desc_netoutput->AddInputDesc(tensor_desc_d);

    NodePtr node_switch = graph->AddNode(op_desc_switch);
    NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    NodePtr node_merge = graph->AddNode(op_desc_merge);
    NodePtr node_cast = graph->AddNode(op_desc_cast);
    NodePtr node_netoutput = graph->AddNode(op_desc_netoutput);
    NodePtr node_other = graph->AddNode(op_desc_other);

    GraphUtils::AddEdge(node_switch->GetOutDataAnchor(0), node_relu1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_switch->GetOutDataAnchor(1), node_relu2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu1->GetOutDataAnchor(0), node_merge->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_merge->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_merge->GetOutDataAnchor(0), node_cast->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_merge->GetOutDataAnchor(1), node_other->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));

    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph2() {
    ComputeGraphPtr graph = CreateSwapMergeCastGraph1();
    OpDescPtr op_desc_some = std::make_shared<OpDesc>("some_node", "Some");
    vector<int64_t> dim = {8, 4, 64, 64};
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT16);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    op_desc_some->AddInputDesc(tensor_desc);
    op_desc_some->AddOutputDesc(tensor_desc);

    NodePtr node_some = graph->AddNode(op_desc_some);

    for (NodePtr node : graph->GetDirectNode()) {
      if (node->GetType() == "Merge") {
        GraphUtils::AddEdge(node->GetOutDataAnchor(0), node_some->GetInDataAnchor(0));
      }
    }
    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph3() {
    ComputeGraphPtr graph = CreateSwapMergeCastGraph1();
    OpDescPtr op_desc_some = std::make_shared<OpDesc>("some_node", "Some");
    vector<int64_t> dim = {8, 4, 64, 64};
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    op_desc_some->AddInputDesc(tensor_desc);
    op_desc_some->AddOutputDesc(tensor_desc);

    NodePtr node_some = graph->AddNode(op_desc_some);

    for (NodePtr node : graph->GetDirectNode()) {
      if (node->GetType() == "Cast") {
        GraphUtils::AddEdge(node->GetOutDataAnchor(0), node_some->GetInDataAnchor(0));
      }
    }
    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph4() {
    ComputeGraphPtr graph = CreateSwapMergeCastGraph1();

    for (NodePtr node : graph->GetDirectNode()) {
      if (node->GetType() == "NetOutput") {
        auto op_desc = node->GetOpDesc();
        ge::OpDescUtilsEx::SetType(op_desc, "NetOut");
      }
    }
    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph5() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_switch = std::make_shared<OpDesc>("switch", "Switch");
    OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    OpDescPtr op_desc_merge = std::make_shared<OpDesc>("merge", "Merge");
    OpDescPtr op_desc_cast = std::make_shared<OpDesc>("cast", "Cast");
    OpDescPtr op_desc_netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");
    OpDescPtr op_desc_other = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT16);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {8, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT16);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    GeTensorDesc tensor_desc_d(shape_c);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_e;
    GeShape shape_e(dim_e);
    GeTensorDesc tensor_desc_e(shape_e);
    tensor_desc_e.SetFormat(FORMAT_ND);
    tensor_desc_e.SetOriginFormat(FORMAT_ND);
    tensor_desc_e.SetDataType(DT_INT32);
    tensor_desc_e.SetOriginDataType(DT_INT32);

    op_desc_switch->AddOutputDesc(tensor_desc_a);
    op_desc_switch->AddOutputDesc(tensor_desc_b);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_a);

    op_desc_relu2->AddInputDesc(tensor_desc_b);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_merge->AddInputDesc(tensor_desc_a);
    op_desc_merge->AddInputDesc(tensor_desc_b);
    op_desc_merge->AddOutputDesc(tensor_desc_c);
    op_desc_merge->AddOutputDesc(tensor_desc_e);

    op_desc_other->AddInputDesc(tensor_desc_e);

    op_desc_cast->AddInputDesc(tensor_desc_c);
    op_desc_cast->AddOutputDesc(tensor_desc_d);

    op_desc_netoutput->AddInputDesc(tensor_desc_d);

    NodePtr node_switch = graph->AddNode(op_desc_switch);
    NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    NodePtr node_merge = graph->AddNode(op_desc_merge);
    NodePtr node_cast = graph->AddNode(op_desc_cast);
    NodePtr node_netoutput = graph->AddNode(op_desc_netoutput);
    NodePtr node_other = graph->AddNode(op_desc_other);

    GraphUtils::AddEdge(node_switch->GetOutDataAnchor(0), node_relu1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_switch->GetOutDataAnchor(1), node_relu2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu1->GetOutDataAnchor(0), node_merge->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_merge->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_merge->GetOutDataAnchor(0), node_cast->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_merge->GetOutDataAnchor(1), node_other->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));

    GraphUtils::AddEdge(node_relu1->GetOutControlAnchor(), node_cast->GetInControlAnchor());
    GraphUtils::AddEdge(node_relu2->GetOutControlAnchor(), node_cast->GetInControlAnchor());
    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph6()
  {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "Switch");
    OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    OpDescPtr op_desc_merge1 = std::make_shared<OpDesc>("merge1", "Merge");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_other1 = std::make_shared<OpDesc>("other1", "Other");

    OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "Switch");
    OpDescPtr op_desc_relu3 = std::make_shared<OpDesc>("relu3", "Relu");
    OpDescPtr op_desc_relu4 = std::make_shared<OpDesc>("relu4", "Relu");
    OpDescPtr op_desc_merge2 = std::make_shared<OpDesc>("merge2", "Merge");
    OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
    OpDescPtr op_desc_other2 = std::make_shared<OpDesc>("other2", "Other");

    OpDescPtr op_desc_netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT16);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {8, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT16);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    GeTensorDesc tensor_desc_d(shape_c);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_e;
    GeShape shape_e(dim_e);
    GeTensorDesc tensor_desc_e(shape_e);
    tensor_desc_e.SetFormat(FORMAT_ND);
    tensor_desc_e.SetOriginFormat(FORMAT_ND);
    tensor_desc_e.SetDataType(DT_INT32);
    tensor_desc_e.SetOriginDataType(DT_INT32);

    op_desc_switch1->AddOutputDesc(tensor_desc_a);
    op_desc_switch1->AddOutputDesc(tensor_desc_b);

    op_desc_switch2->AddOutputDesc(tensor_desc_a);
    op_desc_switch2->AddOutputDesc(tensor_desc_b);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_a);
    op_desc_relu2->AddInputDesc(tensor_desc_b);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_relu3->AddInputDesc(tensor_desc_a);
    op_desc_relu3->AddOutputDesc(tensor_desc_a);
    op_desc_relu4->AddInputDesc(tensor_desc_b);
    op_desc_relu4->AddOutputDesc(tensor_desc_b);

    op_desc_merge1->AddInputDesc(tensor_desc_a);
    op_desc_merge1->AddInputDesc(tensor_desc_b);
    op_desc_merge1->AddOutputDesc(tensor_desc_c);
    op_desc_merge1->AddOutputDesc(tensor_desc_e);

    op_desc_merge2->AddInputDesc(tensor_desc_a);
    op_desc_merge2->AddInputDesc(tensor_desc_b);
    op_desc_merge2->AddOutputDesc(tensor_desc_c);
    op_desc_merge2->AddOutputDesc(tensor_desc_e);

    op_desc_other1->AddInputDesc(tensor_desc_e);

    op_desc_other2->AddInputDesc(tensor_desc_e);

    op_desc_cast1->AddInputDesc(tensor_desc_c);
    op_desc_cast1->AddOutputDesc(tensor_desc_d);

    op_desc_cast2->AddInputDesc(tensor_desc_c);
    op_desc_cast2->AddOutputDesc(tensor_desc_d);

    op_desc_netoutput->AddInputDesc(tensor_desc_d);
    op_desc_netoutput->AddInputDesc(tensor_desc_d);

    NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
    NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    NodePtr node_merge1 = graph->AddNode(op_desc_merge1);
    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_other1 = graph->AddNode(op_desc_other1);

    NodePtr node_switch2 = graph->AddNode(op_desc_switch2);
    NodePtr node_relu3 = graph->AddNode(op_desc_relu3);
    NodePtr node_relu4 = graph->AddNode(op_desc_relu4);
    NodePtr node_merge2 = graph->AddNode(op_desc_merge2);
    NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
    NodePtr node_other2 = graph->AddNode(op_desc_other2);

    NodePtr node_netoutput = graph->AddNode(op_desc_netoutput);

    GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_relu1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(1), node_relu2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu1->GetOutDataAnchor(0), node_merge1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_merge1->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_merge1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_merge1->GetOutDataAnchor(1), node_other1->GetInDataAnchor(0));

    GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), node_relu3->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(1), node_relu4->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu3->GetOutDataAnchor(0), node_merge2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu4->GetOutDataAnchor(0), node_merge2->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_merge2->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_merge2->GetOutDataAnchor(1), node_other2->GetInDataAnchor(0));

    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));

    return graph;
  }
};

TEST_F(fusion_pass_swap_merge_cast_ut, fusion_pass_swap_merge_cast_ut_1) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph1();

  std::map<string, FusionPassRegistry::CreateFn> create_fns =
          FusionPassRegistry::GetInstance().GetCreateFnByType(SECOND_ROUND_BUILT_IN_GRAPH_PASS);
  auto iter = create_fns.find("SwapMergeCastFusionPass");
  Status status = fe::FAILED;
  if (iter != create_fns.end()) {
    auto graph_fusion_pass_base_ptr = std::unique_ptr<PatternFusionBasePass>(
            dynamic_cast<PatternFusionBasePass *>(iter->second()));
    if (graph_fusion_pass_base_ptr != nullptr) {
      graph_fusion_pass_base_ptr->SetName(iter->first);
      status = graph_fusion_pass_base_ptr->Run(*graph);
    }
  }

  EXPECT_EQ(fe::SUCCESS, status);
  vector<int64_t> dim_a = {8, 4, 16, 16};
  vector<int64_t> dim_b = {1, 4, 64, 64};
  vector<int64_t> dim_c = {8, 4, 64, 64};
  for (NodePtr node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Merge") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);

      EXPECT_EQ(op_desc->MutableInputDesc(1)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDims(), dim_b);

      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDims(), dim_c);
    }
    if (op_desc->GetType() == "Cast") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
    }
    if (op_desc->GetName() == "relu1") {
      NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_a);
    }
    if (op_desc->GetName() == "relu2") {
      NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_b);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_b);
    }
  }
}

TEST_F(fusion_pass_swap_merge_cast_ut, fusion_pass_swap_merge_cast_ut_2) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph2();

  SwapMergeCastFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_pass_swap_merge_cast_ut, fusion_pass_swap_merge_cast_ut_3) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph3();

  std::map<string, FusionPassRegistry::CreateFn> create_fns =
          FusionPassRegistry::GetInstance().GetCreateFnByType(SECOND_ROUND_BUILT_IN_GRAPH_PASS);
  auto iter = create_fns.find("SwapMergeCastFusionPass");
  Status status = fe::FAILED;
  if (iter != create_fns.end()) {
    auto graph_fusion_pass_base_ptr = std::unique_ptr<PatternFusionBasePass>(
            dynamic_cast<PatternFusionBasePass *>(iter->second()));
    if (graph_fusion_pass_base_ptr != nullptr) {
      graph_fusion_pass_base_ptr->SetName(iter->first);
      status = graph_fusion_pass_base_ptr->Run(*graph);
    }
  }
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_pass_swap_merge_cast_ut, fusion_pass_swap_merge_cast_ut_4) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph4();

  SwapMergeCastFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes);
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(fusion_pass_swap_merge_cast_ut, fusion_pass_swap_merge_cast_ut_5) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph5();

  std::map<string, FusionPassRegistry::CreateFn> create_fns =
          FusionPassRegistry::GetInstance().GetCreateFnByType(SECOND_ROUND_BUILT_IN_GRAPH_PASS);
  auto iter = create_fns.find("SwapMergeCastFusionPass");
  Status status = fe::FAILED;
  if (iter != create_fns.end()) {
    auto graph_fusion_pass_base_ptr = std::unique_ptr<PatternFusionBasePass>(
            dynamic_cast<PatternFusionBasePass *>(iter->second()));
    if (graph_fusion_pass_base_ptr != nullptr) {
      graph_fusion_pass_base_ptr->SetName(iter->first);
      status = graph_fusion_pass_base_ptr->Run(*graph);
    }
  }

  EXPECT_EQ(fe::SUCCESS, status);
  vector<int64_t> dim_a = {8, 4, 16, 16};
  vector<int64_t> dim_b = {1, 4, 64, 64};
  vector<int64_t> dim_c = {8, 4, 64, 64};
  for (NodePtr node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Merge") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);

      EXPECT_EQ(op_desc->MutableInputDesc(1)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDims(), dim_b);

      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDims(), dim_c);
    }
    if (op_desc->GetType() == "Cast") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
    }
    if (op_desc->GetName() == "relu1") {
      NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_a);
      EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetType(), "NetOutput");
    }
    if (op_desc->GetName() == "relu2") {
      NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_b);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_b);
      EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetType(), "NetOutput");
    }
  }
}


TEST_F(fusion_pass_swap_merge_cast_ut, fusion_pass_swap_merge_cast_ut_6) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph6();

  std::map<string, FusionPassRegistry::CreateFn> create_fns =
          FusionPassRegistry::GetInstance().GetCreateFnByType(SECOND_ROUND_BUILT_IN_GRAPH_PASS);
  auto iter = create_fns.find("SwapMergeCastFusionPass");
  Status status = fe::FAILED;
  if (iter != create_fns.end()) {
    auto graph_fusion_pass_base_ptr = std::unique_ptr<PatternFusionBasePass>(
            dynamic_cast<PatternFusionBasePass *>(iter->second()));
    if (graph_fusion_pass_base_ptr != nullptr) {
      graph_fusion_pass_base_ptr->SetName(iter->first);
      status = graph_fusion_pass_base_ptr->Run(*graph);
    }
  }

  EXPECT_EQ(fe::SUCCESS, status);
  vector<int64_t> dim_a = {8, 4, 16, 16};
  vector<int64_t> dim_b = {1, 4, 64, 64};
  vector<int64_t> dim_c = {8, 4, 64, 64};
  for (NodePtr node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Merge") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);

      EXPECT_EQ(op_desc->MutableInputDesc(1)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDims(), dim_b);

      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDims(), dim_c);

      EXPECT_EQ(node->GetOutDataNodes().at(0)->GetType(), "NetOutput");
    }
    if (op_desc->GetType() == "Cast") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
    }
    if (op_desc->GetName() == "relu1" || op_desc->GetName() == "relu3") {
      NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_a);
    }
    if (op_desc->GetName() == "relu2" || op_desc->GetName() == "relu4") {
      NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_b);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_b);
    }
  }
}
