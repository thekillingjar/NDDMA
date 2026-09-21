/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/shape_operate_op_remove_pass.h"

#include <gtest/gtest.h>

using namespace ge;

class UtestGraphPassesShapeOperateOpRemovePass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  NodePtr AddNode(ComputeGraphPtr graph, const string &name, const string &type, int32_t in_anchors_num = 2,
                  int32_t out_anchors_num = 2) {
    GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
    for (int32_t i = 0; i < in_anchors_num; i++) {
      op_desc->AddInputDesc(tensor_desc);
    }
    for (int32_t i = 0; i < out_anchors_num; i++) {
      op_desc->AddOutputDesc(tensor_desc);
    }

    NodePtr node = graph->AddNode(op_desc);
    return node;
  }
};

TEST_F(UtestGraphPassesShapeOperateOpRemovePass, squeeze_and_squeeze) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  NodePtr transpose_node = AddNode(graph, "transpose1", PERMUTE);
  NodePtr squeeze_node = AddNode(graph, "squeeze1", SQUEEZE);

  GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), squeeze_node->GetInDataAnchor(0));

  ge::ShapeOperateOpRemovePass shape_operate_op_pass;
  Status status = shape_operate_op_pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);
  NodePtr found_node = graph->FindNode("transpose1");
  EXPECT_EQ(transpose_node, found_node);
}
