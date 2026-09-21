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
#include "graph/passes/pass_manager.h"
#include "common/ge_inner_error_codes.h"
#include "graph/passes/control_flow_and_stream/for_pass.h"

namespace ge {
namespace {
class UTEST_graph_passes_for_pass : public testing::Test {
 protected:
  OpDescPtr CreateOpDesc(const std::string name, const std::string type, uint32_t input_num, uint32_t output_num) {
    GeTensorDesc int32_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
    OpDescPtr op_desc = shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name, type));
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

  uint32_t GetNodeNum(const ComputeGraphPtr &graph, const std::string &type) {
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

TEST_F(UTEST_graph_passes_for_pass, no_cond_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr data_node = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));

  ForPass for_pass;
  EXPECT_EQ(for_pass.Run(data_node), SUCCESS);
}

TEST_F(UTEST_graph_passes_for_pass, run_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  AttrUtils::SetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  NodePtr ctrl_in_node = graph->AddNode(CreateOpDesc("ctrl", DATA, 1, 1));
  NodePtr start_node = graph->AddNode(CreateOpDesc("start", DATA, 1, 1));
  NodePtr limit_node = graph->AddNode(CreateOpDesc("limit", DATA, 1, 1));
  NodePtr delta_node = graph->AddNode(CreateOpDesc("delta", DATA, 1, 1));
  NodePtr data_node = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr for_node = graph->AddNode(CreateOpDesc("for", FOR, 4, 1));
  NodePtr output_node = graph->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));

  EXPECT_EQ(GraphUtils::AddEdge(start_node->GetOutDataAnchor(0), for_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(limit_node->GetOutDataAnchor(0), for_node->GetInDataAnchor(1)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(delta_node->GetOutDataAnchor(0), for_node->GetInDataAnchor(2)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), for_node->GetInDataAnchor(3)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(for_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);

  EXPECT_EQ(GraphUtils::AddEdge(ctrl_in_node->GetOutControlAnchor(), for_node->GetInControlAnchor()), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(for_node->GetOutControlAnchor(), output_node->GetInControlAnchor()), SUCCESS);

  ComputeGraphPtr for_body = std::make_shared<ComputeGraph>("for_body");
  for_body->SetParentNode(for_node);
  for_body->SetParentGraph(graph);
  for_node->GetOpDesc()->AddSubgraphName("body");
  for_node->GetOpDesc()->SetSubgraphInstanceName(0, "for_body");
  graph->AddSubgraph(for_body);
  EXPECT_EQ(graph->GetSubgraph("for_body"), for_body);

  {
    // For body
    NodePtr data_node = for_body->AddNode(CreateOpDesc("data", DATA, 1, 1));
    NodePtr output_node = for_body->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  ForPass for_pass;
  EXPECT_EQ(for_pass.Run(for_node), SUCCESS);
  EXPECT_EQ(GetNodeNum(graph, FOR), 0);
  EXPECT_EQ(GetNodeNum(graph, WHILE), 1);
  EXPECT_EQ(graph->GetDirectNode().size(), 12);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 3);
}

} // namespace ge


