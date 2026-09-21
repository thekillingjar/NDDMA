/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/save_pass.h"

#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "ge/ge_api.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/passes/pass_manager.h"
#include "api/gelib/gelib.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestGraphPassesSavePass : public testing::Test {
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

  vector<ge::NodePtr> targets{save_node};
  graph->SetGraphTargetNodesInfo(targets);

  // add edge
  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), save_node->GetInDataAnchor(0));

  return graph;
}

TEST_F(UtestGraphPassesSavePass, cover_run_success) {
  ge::ComputeGraphPtr compute_graph = CreateSaveGraph();
  ge::PassManager pass_managers;
  pass_managers.AddPass("SavePass", new (std::nothrow) SavePass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
}
