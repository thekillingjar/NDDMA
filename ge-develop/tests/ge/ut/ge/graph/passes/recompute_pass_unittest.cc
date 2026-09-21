/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/recompute_pass.h"

#include <gtest/gtest.h>
#include "framework/common/ge_types.h"
#include "graph/ge_local_context.h"
#include "graph/tuning_utils.h"
#include "depends/mmpa/src/mmpa_stub.h"

namespace ge {
class ComputeGraph;
class MockMmpa : public ge::MmpaStubApiGe {
 public:
  static uint32_t MdeRecomputeOptimizeV2(std::shared_ptr<ge::ComputeGraph> &, const uint64_t) {
    return 0U;
  }
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "MdeRecomputeOptimizeV2") {
      return (void *)&MdeRecomputeOptimizeV2;
    }
    return dlsym(handle, func_name);
  }
};
class UtestRecomputePass : public testing::Test {
 protected:
  void SetUp() {
    ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  }
  void TearDown() {
    ge::MmpaStub::GetInstance().Reset();
  }
};

namespace {
void make_graph_can_recompute(ComputeGraphPtr &graph) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  ge::OpDescPtr reshape_desc = std::make_shared<ge::OpDesc>("Reshape_ReshapeRecoveryPass_1", RESHAPE);
  reshape_desc->AddInputDesc(scalar_tensor);
  reshape_desc->AddInputDesc(scalar_tensor);
  reshape_desc->AddOutputDesc(scalar_tensor);
  auto reshape_node = graph->AddNode(reshape_desc);

  auto x_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
  (void)ge::AttrUtils::SetBool(x_desc, "_backward", true);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto y_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
  y_desc->AddOutputDesc(scalar_tensor);
  auto y_node = graph->AddNode(y_desc);

  auto z_desc = std::make_shared<OpDesc>("z", VARIABLEV2);
  (void)ge::AttrUtils::SetBool(z_desc, "_recompute", true);
  z_desc->AddOutputDesc(scalar_tensor);
  auto z_node = graph->AddNode(z_desc);

  auto add_desc = std::make_shared<OpDesc>("Add", ADD);
  (void)ge::AttrUtils::SetBool(add_desc, "_recompute", true);
  add_desc->AddInputDesc(scalar_tensor);
  add_desc->AddInputDesc(scalar_tensor);
  add_desc->AddOutputDesc(scalar_tensor);
  auto add_node = graph->AddNode(add_desc);

  auto add_desc1 = std::make_shared<OpDesc>("gradients/Add1", ADD);
  (void)ge::AttrUtils::SetBool(add_desc1, "_backward", true);
  add_desc1->AddInputDesc(scalar_tensor);
  add_desc1->AddInputDesc(scalar_tensor);
  add_desc1->AddOutputDesc(scalar_tensor);
  auto add_node1 = graph->AddNode(add_desc1);

  auto mul_desc = std::make_shared<OpDesc>("matmul", MATMUL);
  (void)ge::AttrUtils::SetBool(mul_desc, "_recompute", true);
  mul_desc->AddInputDesc(scalar_tensor);
  mul_desc->AddInputDesc(scalar_tensor);
  mul_desc->AddOutputDesc(scalar_tensor);
  auto mul_node = graph->AddNode(mul_desc);

  auto sqrt_desc0 = std::make_shared<OpDesc>("sqrt", SQRT);
  sqrt_desc0->AddInputDesc(scalar_tensor);
  sqrt_desc0->AddOutputDesc(scalar_tensor);
  auto sqrt_node0 = graph->AddNode(sqrt_desc0);

  auto pow_desc0 = std::make_shared<OpDesc>("pow", POW);
  (void)ge::AttrUtils::SetBool(pow_desc0, "_recompute", true);
  GeTensorDesc scalar_tensor1(GeShape(), ge::FORMAT_NCHW, ge::DT_RESOURCE);
  pow_desc0->AddInputDesc(scalar_tensor1);
  pow_desc0->AddOutputDesc(scalar_tensor1);
  auto pow_node0 = graph->AddNode(pow_desc0);

  auto pow_desc1 = std::make_shared<OpDesc>("pow_1", POW);
  (void)ge::AttrUtils::SetBool(pow_desc1, "_recompute", true);
  pow_desc1->AddInputDesc(scalar_tensor);
  pow_desc1->AddOutputDesc(scalar_tensor);
  auto pow_node1 = graph->AddNode(pow_desc1);

  auto pow_desc2 = std::make_shared<OpDesc>("pow_2", POW);
  pow_desc2->AddInputDesc(scalar_tensor);
  pow_desc2->AddOutputDesc(scalar_tensor);
  auto pow_node2 = graph->AddNode(pow_desc2);

  auto mul_desc1 = std::make_shared<OpDesc>("gradients/matmul", MATMUL);
  (void)ge::AttrUtils::SetBool(mul_desc1, "_backward", true);
  mul_desc1->AddInputDesc(scalar_tensor);
  mul_desc1->AddInputDesc(scalar_tensor);
  mul_desc1->AddOutputDesc(scalar_tensor);
  auto mul_node1 = graph->AddNode(mul_desc1);

  auto sqrt_desc = std::make_shared<OpDesc>("gradients/sqrt", SQRT);
  (void)ge::AttrUtils::SetBool(sqrt_desc, "_backward", true);
  sqrt_desc->AddInputDesc(scalar_tensor);
  sqrt_desc->AddOutputDesc(scalar_tensor);
  auto sqrt_node = graph->AddNode(sqrt_desc);

  auto sqrt_desc1 = std::make_shared<OpDesc>("gradients/sqrt_1", SQRT);
  (void)ge::AttrUtils::SetBool(sqrt_desc1, "_backward", true);
  sqrt_desc1->AddInputDesc(scalar_tensor);
  sqrt_desc1->AddOutputDesc(scalar_tensor);
  auto sqrt_node1 = graph->AddNode(sqrt_desc1);

  auto mul_desc2 = std::make_shared<OpDesc>("gradients/matmul_1", MATMUL);
  (void)ge::AttrUtils::SetBool(mul_desc2, "_backward", true);
  mul_desc2->AddInputDesc(scalar_tensor);
  mul_desc2->AddInputDesc(scalar_tensor);
  mul_desc2->AddOutputDesc(scalar_tensor);
  auto mul_node2 = graph->AddNode(mul_desc2);

  auto output_desc = std::make_shared<OpDesc>("NetOutput", "NetOutput");
  output_desc->AddInputDesc(scalar_tensor);
  output_desc->AddOutputDesc(scalar_tensor);
  auto output_node = graph->AddNode(output_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(z_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), pow_node0->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node0->GetOutDataAnchor(0), sqrt_node0->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(sqrt_node0->GetOutDataAnchor(0), pow_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node1->GetOutDataAnchor(0), pow_node2->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node2->GetOutDataAnchor(0), add_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node1->GetOutDataAnchor(0), add_node1->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(add_node1->GetOutDataAnchor(0), mul_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), mul_node1->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(mul_node1->GetOutDataAnchor(0), sqrt_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(sqrt_node1->GetOutDataAnchor(0), mul_node2->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), mul_node2->GetInDataAnchor(1));
  (void)GraphUtils::AddEdge(add_node1->GetOutDataAnchor(0), sqrt_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(sqrt_node->GetOutControlAnchor(), mul_node1->GetInControlAnchor());
  (void)GraphUtils::AddEdge(mul_node2->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
}

void make_graph_can_recompute1(ComputeGraphPtr &graph, bool create_cycle = false) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

  auto x_desc = std::make_shared<OpDesc>("x", DATA);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);

  auto pow_a_desc = std::make_shared<OpDesc>("pow_a", POW);
  (void)ge::AttrUtils::SetBool(pow_a_desc, "_recompute", true);
  pow_a_desc->AddInputDesc(scalar_tensor);
  pow_a_desc->AddOutputDesc(scalar_tensor);
  auto pow_a_node = graph->AddNode(pow_a_desc);

  auto pow_b_desc = std::make_shared<OpDesc>("pow_b", POW);
  pow_b_desc->AddInputDesc(scalar_tensor);
  pow_b_desc->AddOutputDesc(scalar_tensor);
  auto pow_b_node = graph->AddNode(pow_b_desc);

  auto pow_desc0 = std::make_shared<OpDesc>("pow", POW);
  (void)ge::AttrUtils::SetBool(pow_desc0, "_recompute", true);
  pow_desc0->AddInputDesc(scalar_tensor);
  pow_desc0->AddOutputDesc(scalar_tensor);
  auto pow_node0 = graph->AddNode(pow_desc0);

  auto pow_desc1 = std::make_shared<OpDesc>("pow_1", POW);
  (void)ge::AttrUtils::SetBool(pow_desc1, "_recompute", true);
  pow_desc1->AddInputDesc(scalar_tensor);
  pow_desc1->AddOutputDesc(scalar_tensor);
  auto pow_node1 = graph->AddNode(pow_desc1);

  auto sqrt_desc0 = std::make_shared<OpDesc>("sqrt", SQRT);
  sqrt_desc0->AddInputDesc(scalar_tensor);
  sqrt_desc0->AddOutputDesc(scalar_tensor);
  auto sqrt_node0 = graph->AddNode(sqrt_desc0);

  auto sqrt_desc1 = std::make_shared<OpDesc>("gradients/sqrt", SQRT);
  (void)ge::AttrUtils::SetBool(sqrt_desc1, "_backward", true);
  sqrt_desc1->AddInputDesc(scalar_tensor);
  sqrt_desc1->AddOutputDesc(scalar_tensor);
  auto sqrt_node1 = graph->AddNode(sqrt_desc1);

  auto pow_desc2 = std::make_shared<OpDesc>("gradients/pow_1", POW);
  (void)ge::AttrUtils::SetBool(pow_desc2, "_backward", true);
  pow_desc2->AddInputDesc(scalar_tensor);
  pow_desc2->AddOutputDesc(scalar_tensor);
  auto pow_node2 = graph->AddNode(pow_desc2);

  auto addn_desc = std::make_shared<OpDesc>("gradients/AddN", ADDN);
  (void)ge::AttrUtils::SetBool(addn_desc, "_backward", true);
  addn_desc->AddInputDesc(scalar_tensor);
  addn_desc->AddInputDesc(scalar_tensor);
  addn_desc->AddInputDesc(scalar_tensor);
  addn_desc->AddOutputDesc(scalar_tensor);
  auto addn_node = graph->AddNode(addn_desc);

  auto output_desc = std::make_shared<OpDesc>("NetOutput", "NetOutput");
  output_desc->AddInputDesc(scalar_tensor);
  output_desc->AddOutputDesc(scalar_tensor);
  auto output_node = graph->AddNode(output_desc);

  (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), pow_a_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_a_node->GetOutDataAnchor(0), pow_b_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_b_node->GetOutDataAnchor(0), pow_node0->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node0->GetOutDataAnchor(0), pow_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node0->GetOutDataAnchor(0), addn_node->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node1->GetOutDataAnchor(0), sqrt_node0->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node1->GetOutDataAnchor(0), pow_node2->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(pow_node2->GetOutDataAnchor(0), addn_node->GetInDataAnchor(1));
  if (create_cycle) {
    (void)GraphUtils::AddEdge(pow_node2->GetOutControlAnchor(), sqrt_node1->GetInControlAnchor());
  }
  (void)GraphUtils::AddEdge(sqrt_node0->GetOutDataAnchor(0), sqrt_node1->GetInDataAnchor(0));
  (void)GraphUtils::AddEdge(sqrt_node1->GetOutDataAnchor(0), addn_node->GetInDataAnchor(2));
  (void)GraphUtils::AddEdge(addn_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
}
}  // namespace

TEST_F(UtestRecomputePass, test_normal_graph) {
  map<std::string, std::string> options{{RESOURCE_CONFIG_PATH, "/tmp"}};
  GetThreadLocalContext().SetSessionOption(options);
  map<std::string, std::string> graph_options{{RECOMPUTE, "manual"}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  RecomputePass recompute_pass;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph_can_recompute(graph);
  EXPECT_EQ(recompute_pass.Run(graph), SUCCESS);
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetName() == "gradients/matmul") {
      EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "matmul_recompute_copy");
    }
    if (node->GetName() == "gradients/Add1") {
      EXPECT_EQ(node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "pow_1");
    }
    if (node->GetName() == "Add_recompute_copy") {
      EXPECT_EQ(node->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 1);
      EXPECT_EQ(node->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetName(),
                "gradients/sqrt");
    }
    if (node->GetName() == "z") {
      bool is_recompute = false;
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_recompute", is_recompute);
      EXPECT_TRUE(is_recompute == false);
    }
    if (node->GetName() == "pow") {
      bool is_recompute = false;
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_recompute", is_recompute);
      EXPECT_TRUE(is_recompute == false);
    }
    if (node->GetName() == "Reshape_ReshapeRecoveryPass_1") {
      bool is_backward = false;
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_backward", is_backward);
      EXPECT_EQ(is_backward, true);
    }
  }
}

TEST_F(UtestRecomputePass, test_small_topo_bp_node_graph) {
  map<std::string, std::string> options{{RESOURCE_CONFIG_PATH, "/tmp"}};
  GetThreadLocalContext().SetSessionOption(options);
  RecomputePass recompute_pass;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph_can_recompute1(graph);
  EXPECT_EQ(recompute_pass.Run(graph), SUCCESS);
  auto bp_pow_1 = graph->FindNode("gradients/pow_1");
  EXPECT_EQ(bp_pow_1 != nullptr, true);
  EXPECT_EQ(bp_pow_1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "pow_1_recompute_copy");
  auto pow_recompute_copy = graph->FindNode("pow_recompute_copy");
  EXPECT_EQ(pow_recompute_copy != nullptr, true);
  EXPECT_EQ(pow_recompute_copy->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 1);
  EXPECT_EQ(pow_recompute_copy->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetName(),
            "gradients/sqrt");
}

TEST_F(UtestRecomputePass, test_small_topo_bp_node_create_cycle_graph) {
  map<std::string, std::string> options{{RESOURCE_CONFIG_PATH, "/tmp"}};
  GetThreadLocalContext().SetSessionOption(options);
  RecomputePass recompute_pass;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph_can_recompute1(graph, true);
  EXPECT_EQ(recompute_pass.Run(graph), SUCCESS);
  auto bp_pow_1 = graph->FindNode("gradients/pow_1");
  EXPECT_EQ(bp_pow_1 != nullptr, true);
  EXPECT_EQ(bp_pow_1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "pow_1");
  auto pow_recompute_copy = graph->FindNode("pow_recompute_copy");
  EXPECT_EQ(pow_recompute_copy != nullptr, true);
  EXPECT_EQ(pow_recompute_copy->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 1);
  EXPECT_EQ(pow_recompute_copy->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetName(),
            "gradients/sqrt");
}

TEST_F(UtestRecomputePass, test_auto_recompute) {
  map<std::string, std::string> graph_options{{RECOMPUTE, "auto"}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  RecomputePass recompute_pass;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto x_desc = std::make_shared<OpDesc>("x", DATA);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);
  EXPECT_EQ(recompute_pass.Run(graph), FAILED); // 去掉了MDAT的依赖 直接返回不支持
}

TEST_F(UtestRecomputePass, test_aoe_dump) {
  map<std::string, std::string> options{{RESOURCE_CONFIG_PATH, "/tmp"}, {BUILD_MODE, BUILD_MODE_TUNING}};
  GetThreadLocalContext().SetSessionOption(options);
  map<std::string, std::string> graph_options{{RECOMPUTE, "auto"}, {TUNING_PATH, "./"}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  RecomputePass recompute_pass;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto x_desc = std::make_shared<OpDesc>("x", DATA);
  x_desc->AddOutputDesc(scalar_tensor);
  auto x_node = graph->AddNode(x_desc);
  EXPECT_EQ(recompute_pass.Run(graph), SUCCESS);
}

TEST_F(UtestRecomputePass, test_subgraph_not_recompute) {
  RecomputePass recompute_pass;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>("test_subgraph");
  subgraph->SetParentGraph(graph);
  EXPECT_EQ(recompute_pass.Run(subgraph), SUCCESS);
}
} // namespace ge
