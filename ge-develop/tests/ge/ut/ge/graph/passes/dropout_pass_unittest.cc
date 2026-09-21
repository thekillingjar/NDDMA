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

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/standard_optimize/remove_unsupported_op/dropout_pass.h"

#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/passes/pass_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
namespace ge {
class UtestGraphPassesDropoutPass : public Test {
 protected:
  NodePtr AddNode(ComputeGraphPtr graph, const string &name, const string &type, int32_t in_anchors_num = 1,
                  int32_t out_anchors_num = 1) {
    GeTensorDesc tensor_desc;
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
TEST_F(UtestGraphPassesDropoutPass, dropout_remove_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr dropout_node = AddNode(graph, "dropout", DROPOUT);
  NodePtr reduce_min_node = AddNode(graph, "reduceMin", REDUCEMIN);
  NodePtr reduce_max_node = AddNode(graph, "reduceMax", REDUCEMAX);

  GraphUtils::AddEdge(reduce_max_node->GetOutDataAnchor(0), dropout_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dropout_node->GetOutDataAnchor(0), reduce_min_node->GetInDataAnchor(0));
  vector<bool> is_input_const_vec = {true};
  reduce_min_node->GetOpDesc()->SetIsInputConst(is_input_const_vec);

  DropOutPass drop_out_pass;
  Status status = drop_out_pass.Run(dropout_node);
  EXPECT_EQ(SUCCESS, status);
  is_input_const_vec = reduce_min_node->GetOpDesc()->GetIsInputConst();
  EXPECT_EQ(is_input_const_vec[0], true);
  NodePtr found_node = graph->FindNode("dropout");
  EXPECT_EQ(nullptr, found_node);

  NodePtr node = std::make_shared<Node>();
  status = drop_out_pass.Run(node);
  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesDropoutPass, dropout_remove_fail1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr dropout_node = AddNode(graph, "dropout", DROPOUT, 0, 1);
  NodePtr reduce_min_node = AddNode(graph, "reduceMin", REDUCEMIN);
  GraphUtils::AddEdge(dropout_node->GetOutDataAnchor(0), reduce_min_node->GetInDataAnchor(0));

  DropOutPass drop_out_pass;
  Status status = drop_out_pass.Run(dropout_node);
  EXPECT_EQ(FAILED, status);
}

TEST_F(UtestGraphPassesDropoutPass, dropout_square) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr dropout_node = AddNode(graph, "dropout", DROPOUT);
  NodePtr square_node = AddNode(graph, "square", SQUARE);
  NodePtr softplus_node = AddNode(graph, "softplus", SOFTPLUS);
  NodePtr const_node = AddNode(graph, "const", CONSTANT);

  GraphUtils::AddEdge(square_node->GetOutControlAnchor(), dropout_node->GetInControlAnchor());
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), dropout_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dropout_node->GetOutDataAnchor(0), softplus_node->GetInDataAnchor(0));

  DropOutPass drop_out_pass;
  Status status = drop_out_pass.Run(dropout_node);
  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(square_node->GetOutControlAnchor()->GetPeerInControlAnchors().at(0), softplus_node->GetInControlAnchor());
  EXPECT_EQ(const_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0), softplus_node->GetInDataAnchor(0));
}
}  // namespace ge
