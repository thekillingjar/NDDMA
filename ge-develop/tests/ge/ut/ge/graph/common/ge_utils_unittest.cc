/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <gtest/gtest.h>
#include "common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/graph_utils_ex.h"
#include "api/gelib/gelib.h"
#include "ge/ge_api.h"

#include "common/env_path.h"
#include "common/share_graph.h"
#include "ge/ge_utils.h"

namespace ge {
namespace {
const std::string kAicoreEngineName = "AIcoreEngine";
}
class UtestGeUtils : public testing::Test {
 public:
 static void SetUpTestSuite() {
   std::map<string, string> options;
   EXPECT_EQ(GELib::Initialize(options), SUCCESS);
   auto instance = GELib::GetInstance();
   EXPECT_NE(instance, nullptr);
 }
 static void TearDownTestSuite() {
   GEFinalize();
   GELib::GetInstance()->Finalize();
 }
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestGeUtils, CheckNodeSupportOnAicore_NotSupport) {
  // make node
  OpDescPtr add_op = std::make_shared<OpDesc>("add", "Add");
  AttrUtils::SetBool(add_op, "is_supported", false);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto add_node = graph->AddNode(add_op);
  auto add_gnode = NodeAdapter::Node2GNode(add_node);

  bool is_support = false;
  AscendString unsupported_reason;
  EXPECT_NE(GeUtils::CheckNodeSupportOnAicore(add_gnode, is_support, unsupported_reason), SUCCESS);
  EXPECT_TRUE(!is_support);
  EXPECT_STREQ(unsupported_reason.GetString(), "Can not get op info by op type Add");
}

TEST_F(UtestGeUtils, InferShape_SUCCESS) {
  auto compute_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);

  const auto stub_infer_func = [](Operator &op) {
    auto input_shape = op.GetInputDesc(0).GetShape();
    auto output_desc = op.GetOutputDesc(0);
    output_desc.SetShape(input_shape);
    op.UpdateOutputDesc((int)0, output_desc);
    return GRAPH_SUCCESS;
  };
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto output_shape_before_infer = node->GetOpDesc()->GetOutputDesc(0).GetShape();
    node->GetOpDesc()->AddInferFunc(stub_infer_func);
  }

  std::vector<Shape> input_shapes = {Shape({1,2,3,4}), Shape({1,2,3,4})};
  EXPECT_EQ(GeUtils::InferShape(*graph, input_shapes), SUCCESS);
  auto add = compute_graph->FindFirstNodeMatchType(ADD);
  auto output_shape_after_infer = add->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(output_shape_after_infer.ToString(), "1,2,3,4");
}

TEST_F(UtestGeUtils, InferShape_input_shape_count_not_match) {
  auto compute_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);

  std::vector<Shape> input_shapes = {Shape({1,2,3,4})};
  EXPECT_NE(GeUtils::InferShape(*graph, input_shapes), SUCCESS);
}
}  // namespace ge
