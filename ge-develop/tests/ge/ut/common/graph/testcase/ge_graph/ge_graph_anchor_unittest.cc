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

#include "macro_utils/dt_public_scope.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/graph_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;
using namespace std;

class UtestGeAnchor : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeAnchor, data_anchor_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr_1 = std::make_shared<OpDesc>("in_op_1", "float");
  in_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_1 = graph_ptr->AddNode(in_op_ptr_1);
  InDataAnchorPtr in_data_anchor = in_owner_node_1->GetInDataAnchor(0);

  OpDescPtr in_op_ptr_2 = std::make_shared<OpDesc>("in_op_2", "float");
  in_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddInputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddOutputDesc("z", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_2 = graph_ptr->AddNode(in_op_ptr_2);
  InDataAnchorPtr in_data_anchor_x = in_owner_node_2->GetInDataAnchor(0);
  InDataAnchorPtr in_data_anchor_y = in_owner_node_2->GetInDataAnchor(1);
  InControlAnchorPtr in_control_anchor = in_owner_node_2->GetInControlAnchor();

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "float");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph_ptr->AddNode(out_op_ptr_1);
  OutDataAnchorPtr out_data_anchor_1 = out_owner_node_1->GetOutDataAnchor(0);

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "float");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph_ptr->AddNode(out_op_ptr_2);
  OutDataAnchorPtr out_data_anchor_2 = out_owner_node_2->GetOutDataAnchor(0);

  EXPECT_EQ((in_data_anchor->LinkFrom(out_data_anchor_1)), GRAPH_SUCCESS);
  EXPECT_EQ(out_data_anchor_1->LinkTo(in_data_anchor_x), GRAPH_SUCCESS);
  EXPECT_EQ(in_data_anchor_y->LinkFrom(out_data_anchor_2), GRAPH_SUCCESS);
  EXPECT_EQ(out_data_anchor_2->LinkTo(in_control_anchor), GRAPH_SUCCESS);
  EXPECT_EQ(in_control_anchor->GetPeerOutDataAnchors().size(), 1);
  EXPECT_EQ(out_data_anchor_2->GetPeerAnchors().size(), 2);
  EXPECT_EQ(out_data_anchor_2->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(out_data_anchor_2->GetPeerInControlAnchors().size(), 1);
  EXPECT_EQ(out_data_anchor_1->GetPeerAnchors().size(), 2);
  EXPECT_NE(in_data_anchor_y->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(in_data_anchor_x->GetIdx(), 0);
  EXPECT_NE(in_data_anchor_y->GetOwnerNode(), nullptr);
  EXPECT_EQ(out_data_anchor_1->GetPeerInDataAnchors().size(), 2);
  EXPECT_EQ(in_data_anchor_x->Unlink(in_data_anchor_y), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->Unlink(out_data_anchor_2), GRAPH_FAILED);
  EXPECT_EQ(out_data_anchor_2->Unlink(in_data_anchor_x), GRAPH_FAILED);
  out_data_anchor_1->UnlinkAll();
  EXPECT_EQ(out_data_anchor_1->GetPeerInDataAnchors().size(), 0);
  out_data_anchor_2->UnlinkAll();
  EXPECT_EQ(out_data_anchor_2->GetPeerAnchors().size(), 0);
}

TEST_F(UtestGeAnchor, data_anchor_exception_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr = std::make_shared<OpDesc>("in_op_1", "float");
  in_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node = graph_ptr->AddNode(in_op_ptr);
  InDataAnchorPtr in_data_anchor = in_owner_node->GetInDataAnchor(0);

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "float");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph_ptr->AddNode(out_op_ptr_1);
  OutDataAnchorPtr out_data_anchor_1 = out_owner_node_1->GetOutDataAnchor(0);

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "float");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph_ptr->AddNode(out_op_ptr_2);

  OutDataAnchorPtr out_data_anchor_2 = out_owner_node_2->GetOutDataAnchor(0);

  EXPECT_EQ(in_data_anchor->LinkFrom(nullptr), GRAPH_FAILED);
  EXPECT_EQ(out_data_anchor_2->LinkTo(InDataAnchorPtr(nullptr)), GRAPH_FAILED);
  EXPECT_EQ(out_data_anchor_2->LinkTo(InControlAnchorPtr(nullptr)), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->Unlink(nullptr), GRAPH_FAILED);
  in_data_anchor->LinkFrom(out_data_anchor_1);
  EXPECT_EQ(out_data_anchor_2->LinkTo(in_data_anchor), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->LinkFrom(out_data_anchor_2), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->Unlink(out_data_anchor_2), GRAPH_FAILED);
  in_data_anchor->Unlink(out_data_anchor_1);
  EXPECT_EQ(in_data_anchor->GetPeerOutAnchor(), nullptr);
}

TEST_F(UtestGeAnchor, control_anchor_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr_1 = std::make_shared<OpDesc>("in_op_1", "float");
  in_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_1 = graph_ptr->AddNode(in_op_ptr_1);
  InControlAnchorPtr in_control_anchor_1 = in_owner_node_1->GetInControlAnchor();

  OpDescPtr in_op_ptr_2 = std::make_shared<OpDesc>("in_op_2", "float");
  in_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddInputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddOutputDesc("z", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_2 = graph_ptr->AddNode(in_op_ptr_2);
  InControlAnchorPtr in_control_anchor_2 = in_owner_node_2->GetInControlAnchor();

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "float");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph_ptr->AddNode(out_op_ptr_1);
  OutControlAnchorPtr out_control_anchor_1 = out_owner_node_1->GetOutControlAnchor();

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "float");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph_ptr->AddNode(out_op_ptr_2);

  OutControlAnchorPtr out_control_anchor_2 = out_owner_node_2->GetOutControlAnchor();

  EXPECT_EQ(in_control_anchor_1->LinkFrom(out_control_anchor_1), GRAPH_SUCCESS);
  EXPECT_EQ(out_control_anchor_1->LinkTo(in_control_anchor_2), GRAPH_SUCCESS);
  EXPECT_EQ(in_control_anchor_2->LinkFrom(out_control_anchor_2), GRAPH_SUCCESS);
  EXPECT_EQ(in_control_anchor_1->GetPeerAnchors().size(), 1);
  EXPECT_EQ(in_control_anchor_2->GetPeerOutControlAnchors().size(), 2);
  EXPECT_NE(in_control_anchor_2->GetOwnerNode(), nullptr);
  EXPECT_EQ(out_control_anchor_1->GetPeerInControlAnchors().size(), 2);

  EXPECT_EQ(in_control_anchor_1->Unlink(out_control_anchor_2), GRAPH_FAILED);
  EXPECT_EQ(out_control_anchor_2->Unlink(in_control_anchor_1), GRAPH_FAILED);
  EXPECT_EQ(in_control_anchor_1->Unlink(in_control_anchor_2), GRAPH_FAILED);

  EXPECT_EQ(out_control_anchor_2->Unlink(in_control_anchor_2), GRAPH_SUCCESS);
  EXPECT_EQ(in_control_anchor_2->GetPeerOutControlAnchors().size(), 1);
  EXPECT_EQ(out_control_anchor_1->GetPeerAnchors().size(), 2);
  out_control_anchor_1->UnlinkAll();
  EXPECT_EQ(out_control_anchor_1->GetPeerAnchors().size(), 0);
}

TEST_F(UtestGeAnchor, control_anchor_exception_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr = std::make_shared<OpDesc>("in_op_1", "float");
  in_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node = graph_ptr->AddNode(in_op_ptr);
  InControlAnchorPtr in_control_anchor = in_owner_node->GetInControlAnchor();

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "float");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph_ptr->AddNode(out_op_ptr_1);
  OutControlAnchorPtr out_control_anchor_1 = out_owner_node_1->GetOutControlAnchor();

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "float");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph_ptr->AddNode(out_op_ptr_2);

  OutControlAnchorPtr out_control_anchor_2 = out_owner_node_2->GetOutControlAnchor();

  EXPECT_EQ(in_control_anchor->LinkFrom(nullptr), GRAPH_FAILED);
  EXPECT_EQ(out_control_anchor_1->LinkTo(InControlAnchorPtr(nullptr)), GRAPH_FAILED);
  EXPECT_EQ(in_control_anchor->Unlink(nullptr), GRAPH_FAILED);
  in_control_anchor->LinkFrom(out_control_anchor_1);
  EXPECT_EQ(in_control_anchor->Unlink(out_control_anchor_2), GRAPH_FAILED);
  in_control_anchor->Unlink(out_control_anchor_1);
  EXPECT_EQ(in_control_anchor->GetPeerOutControlAnchors().size(), 0);
}

TEST_F(UtestGeAnchor, graph_utils_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr conv_op_ptr = std::make_shared<OpDesc>("conv", "float");
  conv_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW));
  conv_op_ptr->AddInputDesc("w", GeTensorDesc(GeShape({32, 16, 3, 3}), FORMAT_FRACTAL_Z));
  conv_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr conv_node = graph_ptr->AddNode(conv_op_ptr);

  OpDescPtr bn_op_ptr = std::make_shared<OpDesc>("bn", "float");
  bn_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  bn_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr bn_node = graph_ptr->AddNode(bn_op_ptr);

  EXPECT_EQ(GraphUtils::AddEdge(nullptr, conv_node->GetInDataAnchor(0)), GRAPH_FAILED);

  EXPECT_EQ(GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0)), GRAPH_SUCCESS);

  EXPECT_EQ(GraphUtils::AddEdge(OutControlAnchorPtr(nullptr), bn_node->GetInControlAnchor()), GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::AddEdge(conv_node->GetOutControlAnchor(), bn_node->GetInControlAnchor()), GRAPH_SUCCESS);

  OpDescPtr relu_op_ptr = std::make_shared<OpDesc>("relu", "float");
  relu_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  relu_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr relu_node = graph_ptr->AddNode(relu_op_ptr);

  EXPECT_EQ(GraphUtils::ReplaceEdgeDst(conv_node->GetOutControlAnchor(), bn_node->GetInControlAnchor(),
                                       relu_node->GetInControlAnchor()),
            GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::ReplaceEdgeDst(conv_node->GetOutControlAnchor(), bn_node->GetInControlAnchor(),
                                       relu_node->GetInControlAnchor()),
            GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutControlAnchor(), bn_node->GetInControlAnchor()), GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutControlAnchor(), relu_node->GetInControlAnchor()), GRAPH_SUCCESS);

  EXPECT_EQ(GraphUtils::AddEdge(OutDataAnchorPtr(nullptr), bn_node->GetInControlAnchor()), GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInControlAnchor()), GRAPH_FAILED);
}
