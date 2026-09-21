/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/permute_pass.h"

#include <gtest/gtest.h>
#include <string>

using namespace ge;

class UtestGraphPassesPermutePass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  NodePtr AddNode(ComputeGraphPtr graph, const string &name, const string &type, int32_t in_anchors_num = 1,
                  int32_t out_anchors_num = 1) {
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

  ComputeGraphPtr CreatePadGraph() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

    NodePtr data_node = AddNode(graph, "data_op", DATA);

    NodePtr transpose_node = AddNode(graph, "transpose1", PERMUTE);
    vector<int64_t> order_list = {0, 3, 1, 2};
    AttrUtils::SetListInt(transpose_node->GetOpDesc(), PERMUTE_ATTR_ORDER, order_list);
    AttrUtils::SetInt(transpose_node->GetOpDesc(), ATTR_NAME_FORMAT, (int64_t)DT_INT32);

    NodePtr conv_node = AddNode(graph, "conv1", CONVOLUTION);
    NodePtr conv2_node = AddNode(graph, "conv2", CONVOLUTION);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(0));

    return graph;
  }
};

TEST_F(UtestGraphPassesPermutePass, transpose_and_conv3) {
  ComputeGraphPtr graph = CreatePadGraph();

  ge::PermutePass permute_pass;
  Status status = permute_pass.Run(graph);

  EXPECT_EQ(SUCCESS, status);
}
