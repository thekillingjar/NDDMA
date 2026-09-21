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
#include "graph/node.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace ge;

class UtestGeNode : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeNode, node) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  OpDescPtr desc_ptr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(desc_ptr2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);
  NodePtr n2 = graph_ptr->AddNode(desc_ptr);
  NodePtr n3 = graph_ptr->AddNode(desc_ptr);
  NodePtr n4 = graph_ptr->AddNode(desc_ptr);

  EXPECT_EQ(n3->Init(), GRAPH_SUCCESS);
  EXPECT_EQ(n4->Init(), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(n3->GetOutDataAnchor(0), n4->GetInDataAnchor(0)), GRAPH_SUCCESS);

  EXPECT_EQ(n3->GetOwnerComputeGraph(), graph_ptr);
  EXPECT_EQ(n3->GetName(), "name1");
  EXPECT_EQ(n3->GetOpDesc(), desc_ptr);
  int i = 0;
  for (auto in : n3->GetAllOutDataAnchors()) {
    EXPECT_EQ(in->GetIdx(), i++);
  }
  i = 0;
  for (auto in : n3->GetAllInDataAnchors()) {
    EXPECT_EQ(in->GetIdx(), i++);
  }
  EXPECT_EQ(n3->GetInControlAnchor() != nullptr, true);
  EXPECT_EQ(n3->GetOutControlAnchor() != nullptr, true);

  for (auto innode : n4->GetInDataNodes()) {
    EXPECT_EQ(innode, n3);
  }

  for (auto outnode : n3->GetOutDataNodes()) {
    EXPECT_EQ(outnode, n4);
  }
}

TEST_F(UtestGeNode, out_nodes) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  OpDescPtr desc_ptr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(desc_ptr2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);
  NodePtr n2 = graph_ptr->AddNode(desc_ptr);
  NodePtr n4 = graph_ptr->AddNode(desc_ptr);

  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutControlAnchor(), n4->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(n2->GetOutDataAnchor(0), n4->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(n1->GetOutDataNodes().size(), 1);
  EXPECT_EQ(n1->GetOutDataNodes().at(0), n2);
  EXPECT_EQ(n1->GetOutControlNodes().size(), 1);
  EXPECT_EQ(n1->GetOutControlNodes().at(0), n4);
  EXPECT_EQ(n1->GetOutAllNodes().size(), 2);
  EXPECT_EQ(n1->GetOutAllNodes().at(0), n2);
  EXPECT_EQ(n1->GetOutAllNodes().at(1), n4);
  EXPECT_EQ(n4->GetInControlNodes().size(), 1);
  EXPECT_EQ(n4->GetInDataNodes().size(), 1);
  EXPECT_EQ(n4->GetInAllNodes().size(), 2);

  EXPECT_EQ(n1->GetOutDataNodesAndAnchors().size(), 1);
  EXPECT_EQ(n1->GetOutDataNodesAndAnchors().at(0).first, n2);
  EXPECT_EQ(n1->GetOutDataNodesAndAnchors().at(0).second, n2->GetAllInDataAnchors().at(0));
  EXPECT_EQ(n2->GetInDataNodesAndAnchors().size(), 1);
  EXPECT_EQ(n2->GetInDataNodesAndAnchors().at(0).first, n1);
  EXPECT_EQ(n2->GetInDataNodesAndAnchors().at(0).second, n1->GetAllOutDataAnchors().at(0));
}

TEST_F(UtestGeNode, update_opdesc) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  OpDescPtr desc_ptr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(desc_ptr2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);

  EXPECT_EQ(n1->UpdateOpDesc(desc_ptr2), GRAPH_SUCCESS);
}

TEST_F(UtestGeNode, add_link_from_fail) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);

  NodePtr node_ptr = std::make_shared<Node>();
  EXPECT_EQ(n1->AddLinkFrom(node_ptr), GRAPH_PARAM_INVALID);
  EXPECT_EQ(n1->AddLinkFrom(1, node_ptr), GRAPH_PARAM_INVALID);
  EXPECT_EQ(n1->AddLinkFrom("test", node_ptr), GRAPH_PARAM_INVALID);
  EXPECT_EQ(n1->AddLinkFromForParse(node_ptr), GRAPH_PARAM_INVALID);
}

TEST_F(UtestGeNode, verify_failed) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);

  EXPECT_EQ(NodeUtilsEx::Verify(n1), GRAPH_SUCCESS);
}

TEST_F(UtestGeNode, infer_origin_format_success) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);

  EXPECT_EQ(NodeUtilsEx::InferOriginFormat(n1), GRAPH_SUCCESS);
}

TEST_F(UtestGeNode, node_anchor_is_equal) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  OpDescPtr desc_ptr_src = std::make_shared<OpDesc>("str_node", "type");
  EXPECT_EQ(desc_ptr_src->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr_src->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);
  NodePtr str_node = graph_ptr->AddNode(desc_ptr_src);

  OpDescPtr desc_ptr_peer = std::make_shared<OpDesc>("peer_node", "type");
  EXPECT_EQ(desc_ptr_peer->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr_peer->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);
  NodePtr peer_node = graph_ptr->AddNode(desc_ptr_peer);
  EXPECT_EQ(peer_node->AddLinkFrom(str_node), GRAPH_SUCCESS);
  EXPECT_EQ(str_node->NodeAnchorIsEqual(str_node->GetOutAnchor(0), str_node->GetOutAnchor(0), 0), true);
}
