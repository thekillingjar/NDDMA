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
#include "graph/passes/standard_optimize/unused_const_pass.h"

#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/passes/pass_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
namespace ge {
class UtestGraphPassesUnusedConstPass : public Test {
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

TEST_F(UtestGraphPassesUnusedConstPass, unused_const_remove_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr unused_const_node = AddNode(graph, "unusedConst", UNUSEDCONST);
  NodePtr reduce_min_node = AddNode(graph, "reduceMin", REDUCEMIN);

  GraphUtils::AddEdge(unused_const_node->GetOutDataAnchor(0), reduce_min_node->GetInDataAnchor(0));

  UnusedConstPass unused_const_pass;
  Status status = unused_const_pass.Run(unused_const_node);
  EXPECT_EQ(SUCCESS, status);
  NodePtr found_node = graph->FindNode("unusedConst");
  EXPECT_EQ(nullptr, found_node);
}

TEST_F(UtestGraphPassesUnusedConstPass, unused_const_remove_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr unused_const_node = shared_ptr<Node>(new (std::nothrow) Node(nullptr, graph));

  UnusedConstPass unused_const_pass;
  Status status = unused_const_pass.Run(unused_const_node);
  EXPECT_EQ(ge::PARAM_INVALID, status);
}
}  // namespace ge
