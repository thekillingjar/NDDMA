/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/model.h"

#include <gtest/gtest.h>

#include "graph/compute_graph.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils_ex.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestGeModelUnittest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

ge::ComputeGraphPtr CreateSaveGraph() {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  // variable1
  ge::OpDescPtr variable_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(variable_op, "Variable");
  variable_op->SetName("Variable1");
  variable_op->AddInputDesc(ge::GeTensorDesc());
  variable_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr variable_node = graph->AddNode(variable_op);
  // save1
  ge::OpDescPtr save_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(save_op, "Save");
  save_op->SetName("Save1");
  save_op->AddInputDesc(ge::GeTensorDesc());
  save_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr save_node = graph->AddNode(save_op);

  // add edge
  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), save_node->GetInDataAnchor(0));

  return graph;
}

TEST_F(UtestGeModelUnittest, save_model_to_file_success) {
  ge::ComputeGraphPtr compute_graph = CreateSaveGraph();
  auto all_nodes = compute_graph->GetAllNodes();
  for (auto node : all_nodes) {
    auto op_desc = node->GetOpDesc();
    GeTensorDesc weight_desc;
    op_desc->AddOptionalInputDesc("test", weight_desc);
    for (auto in_anchor_ptr : node->GetAllInDataAnchors()) {
      op_desc->IsOptionalInput(in_anchor_ptr->GetIdx());
    }
  }
  EXPECT_NO_THROW(ge::Graph ge_graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph));

  setenv("DUMP_MODEL", "1", true);
  setenv("DUMP_MODEL", "0", true);
}
