/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge/ge_api.h"

#include "ge/ut/ge/test_tools_task_info.h"
#include "ge/ge_ir_build.h"
#include "framework/executor/ge_executor.h"
#include "graph/execute/model_executor.h"
#include "graph/utils/graph_utils_ex.h"

using namespace std;
using namespace testing;

namespace ge {
class CtrlFlowV2Test : public testing::Test {
 protected:
  void SetUp() {
    GeExecutor::Initialize({});
  }
  void TearDown() {
    GeExecutor::FinalizeEx();
  }

  static OpDescPtr CreateOpDesc(const std::string &name,
                                const std::string &type,
                                uint32_t input_num,
                                uint32_t output_num) {
    GeTensorDesc int32_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
    OpDescPtr op_desc = shared_ptr<OpDesc>(new(std::nothrow) OpDesc(name, type));
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
};

TEST_F(CtrlFlowV2Test, TestForOp) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test_graph");
  AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  NodePtr ctrl_in_node = compute_graph->AddNode(CreateOpDesc("ctrl", DATA, 1, 1));
  AttrUtils::SetInt(ctrl_in_node->GetOpDesc(), ATTR_NAME_INDEX, 0);
  NodePtr start_node = compute_graph->AddNode(CreateOpDesc("start", DATA, 1, 1));
  AttrUtils::SetInt(start_node->GetOpDesc(), ATTR_NAME_INDEX, 1);
  NodePtr limit_node = compute_graph->AddNode(CreateOpDesc("limit", DATA, 1, 1));
  AttrUtils::SetInt(limit_node->GetOpDesc(), ATTR_NAME_INDEX, 2);
  NodePtr delta_node = compute_graph->AddNode(CreateOpDesc("delta", DATA, 1, 1));
  AttrUtils::SetInt(delta_node->GetOpDesc(), ATTR_NAME_INDEX, 3);
  NodePtr data_node = compute_graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_INDEX, 4);
  NodePtr for_node = compute_graph->AddNode(CreateOpDesc("for", FOR, 4, 1));
  NodePtr output_node = compute_graph->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));

  EXPECT_EQ(GraphUtils::AddEdge(start_node->GetOutDataAnchor(0), for_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(limit_node->GetOutDataAnchor(0), for_node->GetInDataAnchor(1)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(delta_node->GetOutDataAnchor(0), for_node->GetInDataAnchor(2)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), for_node->GetInDataAnchor(3)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(for_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);

  EXPECT_EQ(GraphUtils::AddEdge(ctrl_in_node->GetOutControlAnchor(), for_node->GetInControlAnchor()), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(for_node->GetOutControlAnchor(), output_node->GetInControlAnchor()), SUCCESS);

  ComputeGraphPtr for_body = std::make_shared<ComputeGraph>("orig_for_body");
  AttrUtils::SetStr(for_body, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  for_body->SetParentNode(for_node);
  for_body->SetParentGraph(compute_graph);
  for_node->GetOpDesc()->AddSubgraphName("body");
  for_node->GetOpDesc()->SetSubgraphInstanceName(0, "orig_for_body");
  compute_graph->AddSubgraph(for_body);
  EXPECT_EQ(compute_graph->GetSubgraph("orig_for_body"), for_body);

  {
    // For body
    auto subgraph_data_node = for_body->AddNode(CreateOpDesc("subgraph_data", DATA, 1, 1));
    EXPECT_TRUE(AttrUtils::SetInt(subgraph_data_node->GetOpDesc(), ATTR_NAME_INDEX, 0));
    auto subgraph_stream_op = CreateOpDesc("subgraph_stream", RELU, 1, 1);
    EXPECT_TRUE(AttrUtils::SetStr(subgraph_stream_op, ATTR_NAME_STREAM_LABEL, "stream_label"));
    auto subgraph_stream_node = for_body->AddNode(subgraph_stream_op);
    auto subgraph_active_op = CreateOpDesc("subgraph_active", STREAMACTIVE, 1, 1);
    EXPECT_TRUE(AttrUtils::SetListStr(subgraph_active_op, ATTR_NAME_ACTIVE_LABEL_LIST, {"label1"}));
    auto subgraph_active_node = for_body->AddNode(subgraph_active_op);
    auto subgraph_output_node = for_body->AddNode(CreateOpDesc("subgraph-net-output", NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(subgraph_data_node->GetOutDataAnchor(0), subgraph_stream_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(subgraph_stream_node->GetOutDataAnchor(0), subgraph_active_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(subgraph_active_node->GetOutDataAnchor(0), subgraph_output_node->GetInDataAnchor(0)), SUCCESS);
  }

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options;
  ModelBufferData model_buffer_data{};
  EXPECT_EQ(aclgrphBuildModel(graph, options, model_buffer_data), SUCCESS);
}
}  // namespace ge

