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
#include "graph/passes/standard_optimize/remove_unsupported_op/snapshot_pass.h"

#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/passes/pass_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace std;
using namespace ge;

class UtestSnapshotPass : public Test {
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

TEST_F(UtestSnapshotPass, succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  NodePtr prevent_grad = AddNode(graph, "PreventGradient", PREVENTGRADIENT);
  NodePtr snapshot = AddNode(graph, "Snapshot", SNAPSHOT);
  NodePtr constant = AddNode(graph, "Const", CONSTANT);
  GraphUtils::AddEdge(prevent_grad->GetOutDataAnchor(0), snapshot->GetInDataAnchor(0));
  GraphUtils::AddEdge(snapshot->GetOutDataAnchor(0), constant->GetInDataAnchor(0));

  SnapshotPass pass;
  Status status = pass.Run(snapshot);
  EXPECT_EQ(status, SUCCESS);

  NodePtr found_node = graph->FindNode("Snapshot");
  EXPECT_EQ(found_node, nullptr);

  status = pass.Run(prevent_grad);
  EXPECT_EQ(status, SUCCESS);

  string type2 = "FrameworkOp";
  auto op_desc = snapshot->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, type2);
  status = pass.Run(snapshot);
}
