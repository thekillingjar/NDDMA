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
#include "graph/passes/control_flow_and_stream/cond_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"
#include "ge_graph_dsl/graph_dsl.h"

using namespace std;
using namespace ge;

class UtestCondPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
  OpDescPtr CreateOpDesc(const string &name, const string &type, const GeTensorDesc &in_tensor, uint32_t input_num,
                         const GeTensorDesc &out_tensor, uint32_t output_num) {
    OpDescPtr op_desc = shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name, type));
    if (op_desc == nullptr) {
      return nullptr;
    }
    for (uint32_t i = 0; i < input_num; i++) {
      op_desc->AddInputDesc(in_tensor);
    }
    for (uint32_t i = 0; i < output_num; i++) {
      op_desc->AddOutputDesc(out_tensor);
    }
    return op_desc;
  }

  Status RunGraphPass(const GeTensorDesc &tensor_desc, const ComputeGraphPtr &graph){
    NodePtr data_node = graph->AddNode(CreateOpDesc("data", DATA, tensor_desc, 1, tensor_desc, 1));
    NodePtr if_node = graph->AddNode(CreateOpDesc("if", IF, tensor_desc, 2, tensor_desc, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0)), SUCCESS);

    CondPass pass;
    return pass.Run(if_node);
  }

  uint32_t GetNodeNum(const ComputeGraphPtr &graph, const string &type) {
    uint32_t num = 0;
    for (auto &node : graph->GetDirectNode()) {
      if (node->GetType() == type) {
        num++;
      }
    }
    return num;
  }
};

TEST_F(UtestCondPass, no_cond_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr data_node = graph->AddNode(CreateOpDesc("data", DATA, GeTensorDesc(), 1, GeTensorDesc(), 1));

  CondPass pass;
  auto ret = pass.Run(data_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCondPass, not_scalar_succ) {
  GeTensorDesc not_scalar_tensor(GeShape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr data_node = graph->AddNode(CreateOpDesc("data", DATA, not_scalar_tensor, 1, not_scalar_tensor, 1));
  NodePtr if_node = graph->AddNode(CreateOpDesc("if", IF, not_scalar_tensor, 2, not_scalar_tensor, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0)), SUCCESS);

  CondPass pass;
  auto ret = pass.Run(if_node);
  EXPECT_EQ(ret, SUCCESS);
  auto size_node_num = GetNodeNum(graph, SIZE);
  EXPECT_EQ(size_node_num, 1);
}

TEST_F(UtestCondPass, string_scalar_succ) {
  GeTensorDesc string_scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_STRING);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");

  auto ret = RunGraphPass(string_scalar_tensor, graph);
  EXPECT_EQ(ret, SUCCESS);
  auto node_num = GetNodeNum(graph, "StringLength");
  EXPECT_EQ(node_num, 1);
  node_num = GetNodeNum(graph, CAST);
  EXPECT_EQ(node_num, 0);
}

TEST_F(UtestCondPass, bool_scalar_succ) {
  GeTensorDesc bool_scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");

  auto ret = RunGraphPass(bool_scalar_tensor, graph);
  EXPECT_EQ(ret, SUCCESS);
  auto node_num = GetNodeNum(graph, CAST);
  EXPECT_EQ(node_num, 1);
}

TEST_F(UtestCondPass, float_scalar_succ) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");

  auto ret = RunGraphPass(scalar_tensor, graph);
  EXPECT_EQ(ret, SUCCESS);
  auto node_num = GetNodeNum(graph, CAST);
  EXPECT_EQ(node_num, 1);
}

TEST_F(UtestCondPass, double_scalar_succ) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_DOUBLE);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");

  auto ret = RunGraphPass(scalar_tensor, graph);
  EXPECT_EQ(ret, SUCCESS);
  auto node_num = GetNodeNum(graph, CAST);
  EXPECT_EQ(node_num, 1);
}

TEST_F(UtestCondPass, uint8_scalar_succ) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_UINT8);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");

  auto ret = RunGraphPass(scalar_tensor, graph);
  EXPECT_EQ(ret, SUCCESS);
  auto node_num = GetNodeNum(graph, CAST);
  EXPECT_EQ(node_num, 1);
}

TEST_F(UtestCondPass, int32_scalar_succ) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");

  auto ret = RunGraphPass(scalar_tensor, graph);
  EXPECT_EQ(ret, SUCCESS);
  auto node_num = GetNodeNum(graph, CAST);
  EXPECT_EQ(node_num, 0);
}

TEST_F(UtestCondPass, int64_scalar_succ) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT64);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");

  auto ret = RunGraphPass(scalar_tensor, graph);
  EXPECT_EQ(ret, SUCCESS);
  auto node_num = GetNodeNum(graph, CAST);
  EXPECT_EQ(node_num, 1);
}

TEST_F(UtestCondPass, other_scalar_fail) {
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_RESOURCE);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");

  auto ret = RunGraphPass(scalar_tensor, graph);
  EXPECT_EQ(ret, FAILED);
  auto node_num = GetNodeNum(graph, CAST);
  EXPECT_EQ(node_num, 0);
}

TEST_F(UtestCondPass, unknown_rank_succ) {
  GeTensorDesc scalar_tensor(GeShape({-2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");

  auto ret = RunGraphPass(scalar_tensor, graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 2);
}

TEST_F(UtestCondPass, no_peer_fail) {
  GeTensorDesc not_scalar_tensor(GeShape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr if_node = graph->AddNode(CreateOpDesc("if", IF, not_scalar_tensor, 2, not_scalar_tensor, 1));

  CondPass pass;
  auto ret = pass.Run(if_node);
  EXPECT_EQ(ret, FAILED);
}

// int8, int16, int64
TEST_F(UtestCondPass, while_no_cond_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr while_node = graph->AddNode(CreateOpDesc("while", WHILE, GeTensorDesc(), 1, GeTensorDesc(), 1));

  CondPass pass;
  auto ret = pass.Run(while_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCondPass, while_not_scalar_success) {
  GeTensorDesc not_scalar_tensor(GeShape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr while_node = graph->AddNode(CreateOpDesc("while", WHILE, not_scalar_tensor, 1, not_scalar_tensor, 1));

  std::string cond_graph_name = "cond_graph";
  ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>(cond_graph_name);
  cond_graph->SetParentNode(while_node);
  cond_graph->SetParentGraph(graph);
  while_node->GetOpDesc()->AddSubgraphName(cond_graph_name);
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph_name);
  EXPECT_EQ(graph->AddSubgraph(cond_graph_name, cond_graph), GRAPH_SUCCESS);

  {
    NodePtr data_node = cond_graph->AddNode(CreateOpDesc("data", DATA, not_scalar_tensor, 1, not_scalar_tensor, 1));
    NodePtr output_node = cond_graph->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, not_scalar_tensor, 1, not_scalar_tensor, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  CondPass pass;
  auto ret = pass.Run(while_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCondPass, while_not_scalar_succ) {
  GeTensorDesc not_scalar_tensor(GeShape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr while_node = graph->AddNode(CreateOpDesc("while", WHILE, not_scalar_tensor, 1, not_scalar_tensor, 1));

  std::string cond_graph_name = "cond";
  ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>(cond_graph_name);
  cond_graph->SetParentNode(while_node);
  cond_graph->SetParentGraph(graph);
  while_node->GetOpDesc()->AddSubgraphName(cond_graph_name);
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph_name);
  EXPECT_EQ(graph->AddSubgraph(cond_graph_name, cond_graph), GRAPH_SUCCESS);

  {
    NodePtr data_node = cond_graph->AddNode(CreateOpDesc("data", DATA, not_scalar_tensor, 1, not_scalar_tensor, 1));
    NodePtr output_node = cond_graph->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, not_scalar_tensor, 1, not_scalar_tensor, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  CondPass pass;
  auto ret = pass.Run(while_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCondPass, while_cond_output_not_one_fail) {
  GeTensorDesc not_scalar_tensor(GeShape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr while_node = graph->AddNode(CreateOpDesc("while", WHILE, not_scalar_tensor, 1, not_scalar_tensor, 1));

  std::string cond_graph_name = "cond";
  ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>(cond_graph_name);
  cond_graph->SetParentNode(while_node);
  cond_graph->SetParentGraph(graph);
  while_node->GetOpDesc()->AddSubgraphName(cond_graph_name);
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph_name);
  EXPECT_EQ(graph->AddSubgraph(cond_graph_name, cond_graph), GRAPH_SUCCESS);

  {
    NodePtr data_node = cond_graph->AddNode(CreateOpDesc("data", DATA, not_scalar_tensor, 1, not_scalar_tensor, 1));
    NodePtr output_node = cond_graph->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, not_scalar_tensor, 2, not_scalar_tensor, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(1)), SUCCESS);
  }

  CondPass pass;
  auto ret = pass.Run(while_node);
  EXPECT_EQ(ret, FAILED);
}
