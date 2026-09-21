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
#include <string>
#include "utils/graph_utils.h"
#include "common/ge_inner_error_codes.h"
#include "pass/concat_from_sequence_pass.h"
#include "graph/compute_graph.h"

#include "graph_metadef/graph/debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/mem_utils.h"

using namespace ge;
using namespace std;

namespace aicpu {
namespace {
  const string OP_SEQUENCEINSERT = "SequenceInsert";
  const string OP_ADD = "Add";
  const string OP_CONCATFROMSEQUENCE = "ConcatFromSequence";
  const string OP_NETOUTPUT = "NetOutput";
class UTEST_graph_passes_concatFromSequence_pass : public testing::Test {
 protected:
  OpDescPtr CreateOpDesc(const std::string name, const std::string type, uint32_t input_num, uint32_t output_num) {
    GeTensorDesc int32_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
    if (op_desc == nullptr) {
      return nullptr;
    }
    for (uint32_t i = 0; i < input_num; i++) {
      op_desc->AddInputDesc(int32_tensor);
    }
    for (uint32_t i = 0; i < output_num; i++) {
      op_desc->AddOutputDesc(int32_tensor);
    }
    return op_desc;
  }

  uint32_t GetNodeNum(const ComputeGraphPtr graph, const std::string &type) {
    uint32_t num = 0;
    for (auto &node : graph->GetDirectNode()) {
      if (node->GetType() == type) {
        num++;
      }
    }
    return num;
  }
};
} // namespace

TEST_F(UTEST_graph_passes_concatFromSequence_pass, concatFromSequence_pass_run_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  AttrUtils::SetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  NodePtr concatFromSequence_node = graph->AddNode(CreateOpDesc("concatFromSequence", OP_CONCATFROMSEQUENCE, 1, 1));
  NodePtr add_node = graph->AddNode(CreateOpDesc("add", OP_ADD, 2, 1));
  NodePtr sequenceInsert_node = graph->AddNode(CreateOpDesc("sequenceInsert", OP_SEQUENCEINSERT, 3, 1));
  NodePtr output_node = graph->AddNode(CreateOpDesc("Node_Output", OP_NETOUTPUT, 1, 1));

  EXPECT_EQ(GraphUtils::AddEdge(sequenceInsert_node->GetOutDataAnchor(0), concatFromSequence_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(concatFromSequence_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);

  ConcatFromSequencePass concatFromSequence_pass;
  EXPECT_EQ(concatFromSequence_pass.Run(*graph), SUCCESS);
}

TEST_F(UTEST_graph_passes_concatFromSequence_pass, concatFromSequence_pass_new_axis_run_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  AttrUtils::SetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  NodePtr concatFromSequence_node = graph->AddNode(CreateOpDesc("concatFromSequence", OP_CONCATFROMSEQUENCE, 1, 1));
  NodePtr add_node = graph->AddNode(CreateOpDesc("add", OP_ADD, 2, 1));
  NodePtr sequenceInsert_node = graph->AddNode(CreateOpDesc("sequenceInsert", OP_SEQUENCEINSERT, 3, 1));
  NodePtr output_node = graph->AddNode(CreateOpDesc("Node_Output", OP_NETOUTPUT, 1, 1));

  auto new_axis = 1;
  AttrUtils::SetInt(concatFromSequence_node->GetOpDesc(), "new_axis", new_axis);
  EXPECT_EQ(GraphUtils::AddEdge(sequenceInsert_node->GetOutDataAnchor(0), concatFromSequence_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(concatFromSequence_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);

  ConcatFromSequencePass concatFromSequence_pass;
  EXPECT_EQ(concatFromSequence_pass.Run(*graph), SUCCESS);
}

} // namespace aicpu
