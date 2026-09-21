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
#include "graph/utils/anchor_utils.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;

class UtestGeAnchorUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeAnchorUtils, base) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);
  InDataAnchorPtr a1 = std::make_shared<InDataAnchor>(n1, 0);

  // has control edge
  OpDescPtr desc_ptr1 = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr1->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr1->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  OpDescPtr desc_ptr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(desc_ptr2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graph_ptr1 = std::make_shared<ComputeGraph>("name");
  n1 = graph_ptr1->AddNode(desc_ptr1);
  NodePtr n2 = graph_ptr1->AddNode(desc_ptr2);

  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutControlAnchor(), n2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::RemoveEdge(n1->GetOutControlAnchor(), n2->GetInControlAnchor()), GRAPH_SUCCESS);

  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::RemoveEdge(n1->GetOutDataAnchor(0), n2->GetInControlAnchor()), GRAPH_SUCCESS);

  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::RemoveEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0)), GRAPH_SUCCESS);
}
