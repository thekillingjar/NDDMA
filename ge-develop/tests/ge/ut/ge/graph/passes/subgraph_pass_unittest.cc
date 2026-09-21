/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/memory_conflict/subgraph_pass.h"
#include "graph/passes/pass_manager.h"
#include "common/share_graph.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace {
class UtestGraphPassesSubgraphPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

OpDescPtr CreateOpDesc(const std::string name, const std::string type, uint32_t input_num, uint32_t output_num) {
  OpDescPtr op_desc = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name, type));
  if (op_desc == nullptr) {
    return nullptr;
  }
  for (uint32_t i = 0; i < input_num; i++) {
    op_desc->AddInputDesc(GeTensorDesc());
  }
  for (uint32_t i = 0; i < output_num; i++) {
    op_desc->AddOutputDesc(GeTensorDesc());
  }
  return op_desc;
}

bool CheckMemcpyExist(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == IDENTITY) {
      return true;
    }
  }
  return false;
}

uint32_t CheckMemcpyNum(const ComputeGraphPtr &graph) {
  uint32_t num = 0;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == IDENTITY) {
      num++;
    }
  }
  return num;
}
} // namespace

TEST_F(UtestGraphPassesSubgraphPass, add_memcpy_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr function_node = graph->AddNode(CreateOpDesc("Case", CASE, 2, 1));
  NodePtr data_node_0 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr data_node_1 = graph->AddNode(CreateOpDesc("data1", DATA, 1, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), function_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);

  std::string subgraph_name1 = "instance_branch1";
  ComputeGraphPtr subgraph1 = std::make_shared<ComputeGraph>(subgraph_name1);
  subgraph1->SetParentNode(function_node);
  subgraph1->SetParentGraph(graph);
  ge::GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph1);
  size_t index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  function_node->GetOpDesc()->AddSubgraphName("branch1");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name1);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  std::string subgraph_name2 = "instance_branch2";
  ComputeGraphPtr subgraph2 = std::make_shared<ComputeGraph>(subgraph_name2);
  subgraph2->SetParentNode(function_node);
  subgraph2->SetParentGraph(graph);
  ge::GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph2);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 1);
  function_node->GetOpDesc()->AddSubgraphName("branch2");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name2);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);

  {
    // Const->NetOutput in subgraph
    NodePtr const_node = subgraph1->AddNode(CreateOpDesc("const", CONSTANTOP, 0, 1));
    NodePtr output_node = subgraph1->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  {
    // Data->NetOutput in subgraph but not while body
    NodePtr data_node = subgraph2->AddNode(CreateOpDesc("data", DATA, 1, 1));
    NodePtr output_node = subgraph2->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  }

  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(graph));
  // TODO:
  // EXPECT_EQ(pass_managers.Run(subgraph1), SUCCESS);
  // EXPECT_TRUE(CheckMemcpyExist(subgraph1));
  // EXPECT_EQ(pass_managers.Run(subgraph2), SUCCESS);
  // EXPECT_TRUE(CheckMemcpyExist(subgraph2));

  {
    // Input->FunctionOp and Input link to other nodes
    NodePtr identity_node = graph->AddNode(CreateOpDesc("Identity", RELU, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), identity_node->GetInDataAnchor(0)), SUCCESS);
  }
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(graph));

  {
    // Data->NetOutput in root_graph
    NodePtr output_node = graph->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(graph));

  {
    // Const->FunctionOp
    EXPECT_EQ(GraphUtils::RemoveEdge(data_node_1->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);
    NodePtr const_node = graph->AddNode(CreateOpDesc("const", CONSTANTOP, 0, 1));
    EXPECT_EQ(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);
  }
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(graph));
}

TEST_F(UtestGraphPassesSubgraphPass, while_subgraph_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr function_node = graph->AddNode(CreateOpDesc("While", WHILE, 2, 2));

  std::string subgraph_name1 = "instance_cond";
  ComputeGraphPtr subgraph1 = std::make_shared<ComputeGraph>(subgraph_name1);
  size_t index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  function_node->GetOpDesc()->AddSubgraphName("cond");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name1);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  subgraph1->SetParentNode(function_node);
  subgraph1->SetParentGraph(graph);
  graph->AddSubgraph(subgraph1);

  std::string subgraph_name2 = "instance_body";
  ComputeGraphPtr subgraph2 = std::make_shared<ComputeGraph>(subgraph_name2);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 1);
  function_node->GetOpDesc()->AddSubgraphName("body");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name2);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  subgraph2->SetParentNode(function_node);
  subgraph2->SetParentGraph(graph);
  graph->AddSubgraph(subgraph2);

  // Input->While
  NodePtr data_node_0 = graph->AddNode(CreateOpDesc("data_0", DATA, 1, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), function_node->GetInDataAnchor(0)), SUCCESS);

  NodePtr data_node_1 = graph->AddNode(CreateOpDesc("data_1", DATA, 1, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);

  {
    // Data->FunctionOp in while cond subgraph
    NodePtr data_node = subgraph1->AddNode(CreateOpDesc("data", DATA, 1, 1));
    NodePtr output_node = subgraph1->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  }

  {
    // Data->NetOutput in while body subgraph
    NodePtr data_node = subgraph2->AddNode(CreateOpDesc("data", DATA, 1, 1));
    NodePtr output_node = subgraph2->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  }

  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(graph));
  EXPECT_EQ(pass_managers.Run(subgraph1), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(subgraph1));
  EXPECT_EQ(pass_managers.Run(subgraph2), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(subgraph2));

  {
    // Input->While and Input link to other nodes
    NodePtr identity_node = graph->AddNode(CreateOpDesc("Identity", RELU, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), identity_node->GetInDataAnchor(0)), SUCCESS);
  }
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_TRUE(CheckMemcpyExist(graph));
}

TEST_F(UtestGraphPassesSubgraphPass, constant_case_subgraph_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr function_node = graph->AddNode(CreateOpDesc("Case", CASE, 2, 1));
  NodePtr data_node_0 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr data_node_1 = graph->AddNode(CreateOpDesc("data1", CONSTANTOP, 1, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), function_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);

  std::string subgraph_name1 = "instance_branch1";
  ComputeGraphPtr subgraph1 = std::make_shared<ComputeGraph>(subgraph_name1);
  size_t index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  function_node->GetOpDesc()->AddSubgraphName("branch1");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name1);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  std::string subgraph_name2 = "instance_branch2";
  ComputeGraphPtr subgraph2 = std::make_shared<ComputeGraph>(subgraph_name2);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 1);
  function_node->GetOpDesc()->AddSubgraphName("branch2");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name2);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);

  subgraph1->SetParentNode(function_node);
  subgraph1->SetParentGraph(graph);
  graph->AddSubgraph(subgraph1);
  subgraph2->SetParentNode(function_node);
  subgraph2->SetParentGraph(graph);
  graph->AddSubgraph(subgraph2);

  {
    // Const->FunctionOp in subgraph
    NodePtr data_node = subgraph1->AddNode(CreateOpDesc("data", DATA, 0, 1));
    NodePtr output_node = subgraph1->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  }

  {
    // Data->FunctionOp in subgraph but not while body
    NodePtr data_node = subgraph2->AddNode(CreateOpDesc("data", DATA, 1, 1));
    NodePtr output_node = subgraph2->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  }

  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new SubgraphPass);
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(graph));
}

TEST_F(UtestGraphPassesSubgraphPass, constant_while_subgraph_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr function_node = graph->AddNode(CreateOpDesc("While", WHILE, 2, 1));

  // Input->While
  NodePtr data_node_0 = graph->AddNode(CreateOpDesc("data_0", DATA, 1, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), function_node->GetInDataAnchor(0)), SUCCESS);
  NodePtr data_node_1 = graph->AddNode(CreateOpDesc("data_1", CONSTANTOP, 1, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);

  std::string subgraph_name1 = "instance_cond";
  ComputeGraphPtr subgraph1 = std::make_shared<ComputeGraph>(subgraph_name1);
  size_t index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  function_node->GetOpDesc()->AddSubgraphName("cond");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name1);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  std::string subgraph_name2 = "instance_body";
  ComputeGraphPtr subgraph2 = std::make_shared<ComputeGraph>(subgraph_name2);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 1);
  function_node->GetOpDesc()->AddSubgraphName("body");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name2);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);

  subgraph1->SetParentNode(function_node);
  subgraph1->SetParentGraph(graph);
  graph->AddSubgraph(subgraph1);
  subgraph2->SetParentNode(function_node);
  subgraph2->SetParentGraph(graph);
  graph->AddSubgraph(subgraph2);

  {
    // Constant->FunctionOp in while cond subgraph
    NodePtr data_node = subgraph1->AddNode(CreateOpDesc("data", DATA, 1, 1));
    NodePtr output_node = subgraph1->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  }

  {
    // Constant->FunctionOp in while body subgraph
    NodePtr data_node = subgraph2->AddNode(CreateOpDesc("data", DATA, 1, 1));
    NodePtr output_node = subgraph2->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  }

  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new SubgraphPass);
  // TODO:
  // EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  // EXPECT_TRUE(CheckMemcpyExist(graph));
}

TEST_F(UtestGraphPassesSubgraphPass, reassign_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr data_node = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr function_node = graph->AddNode(CreateOpDesc("Case", CASE, 2, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), function_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);

  std::string subgraph_name1 = "instance_branch1";
  ComputeGraphPtr subgraph1 = std::make_shared<ComputeGraph>(subgraph_name1);
  subgraph1->SetParentNode(function_node);
  subgraph1->SetParentGraph(graph);
  ge::GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph1);
  size_t index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  function_node->GetOpDesc()->AddSubgraphName("branch1");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name1);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  std::string subgraph_name2 = "instance_branch2";
  ComputeGraphPtr subgraph2 = std::make_shared<ComputeGraph>(subgraph_name2);
  subgraph2->SetParentNode(function_node);
  subgraph2->SetParentGraph(graph);
  ge::GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph2);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 1);
  function_node->GetOpDesc()->AddSubgraphName("branch2");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name2);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);

  std::string subgraph_name3 = "instance_branch3";
  ComputeGraphPtr subgraph3 = std::make_shared<ComputeGraph>(subgraph_name3);
  subgraph3->SetParentNode(function_node);
  subgraph3->SetParentGraph(graph);
  ge::GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph3);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 2);
  function_node->GetOpDesc()->AddSubgraphName("branch3");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 3);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name3);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 3);

  std::string subgraph_name4 = "instance_branch4";
  ComputeGraphPtr subgraph4 = std::make_shared<ComputeGraph>(subgraph_name4);
  subgraph4->SetParentNode(function_node);
  subgraph4->SetParentGraph(graph);
  ge::GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph4);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 3);
  function_node->GetOpDesc()->AddSubgraphName("branch4");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 4);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name4);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 4);

  std::string subgraph_name5 = "instance_branch5";
  ComputeGraphPtr subgraph5 = std::make_shared<ComputeGraph>(subgraph_name5);
  subgraph5->SetParentNode(function_node);
  subgraph5->SetParentGraph(graph);
  ge::GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph5);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 4);
  function_node->GetOpDesc()->AddSubgraphName("branch5");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 5);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name5);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 5);

  {
    // AtomicOp->NetOutput in subgraph
    OpDescPtr atomic_desc = CreateOpDesc("Atomic", "Atomic", 1, 1);
    AttrUtils::SetBool(atomic_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, true);
    AttrUtils::SetListInt(atomic_desc, ATOMIC_ATTR_OUTPUT_INDEX, { 0, 1 });
    NodePtr atomic_node = subgraph1->AddNode(atomic_desc);
    NodePtr output_node = subgraph1->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(atomic_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  {
    // OutputContinuesRequiredOp->NetOutput in subgraph
    OpDescPtr continues_desc = CreateOpDesc("OutputContinues", "OutputContinues", 1, 1);
    AttrUtils::SetBool(continues_desc, ATTR_NAME_CONTINUOUS_OUTPUT, true);
    NodePtr continues_node = subgraph2->AddNode(continues_desc);
    NodePtr output_node = subgraph2->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(continues_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  {
    // NoPaddingOutputContinuesRequiredOp->NetOutput in subgraph
    OpDescPtr nopadding_continues_desc = CreateOpDesc("OutputContinues", "OutputContinues", 1, 1);
    AttrUtils::SetBool(nopadding_continues_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
    NodePtr nopadding_continues_node = subgraph3->AddNode(nopadding_continues_desc);
    NodePtr output_node = subgraph3->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(nopadding_continues_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  {
    // DataInputContinuesRequiredOp->NetOutput in subgraph
    OpDescPtr data_desc = CreateOpDesc("data", DATA, 1, 1);
    AttrUtils::SetInt(data_desc, ATTR_NAME_PARENT_NODE_INDEX, 1);
    NodePtr data_node = subgraph4->AddNode(data_desc);
    OpDescPtr continues_desc = CreateOpDesc("InputContinues", "InputContinues", 1, 1);
    AttrUtils::SetBool(continues_desc, ATTR_NAME_CONTINUOUS_INPUT, true);
    NodePtr continues_node = subgraph4->AddNode(continues_desc);
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), continues_node->GetInDataAnchor(0)), SUCCESS);
    NodePtr output_node = subgraph4->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(continues_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  {
    // DataInputContinuesRequiredOp->NetOutput in subgraph
    OpDescPtr data_desc = CreateOpDesc("data", DATA, 1, 1);
    AttrUtils::SetInt(data_desc, ATTR_NAME_PARENT_NODE_INDEX, 1);
    NodePtr data_node = subgraph5->AddNode(data_desc);
    OpDescPtr nopadding_continues_desc = CreateOpDesc("InputContinues", "InputContinues", 1, 1);
    AttrUtils::SetBool(nopadding_continues_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    NodePtr nopadding_continues_node = subgraph5->AddNode(nopadding_continues_desc);
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), nopadding_continues_node->GetInDataAnchor(0)), SUCCESS);
    NodePtr output_node = subgraph5->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(nopadding_continues_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(graph));
  // TODO:
  // EXPECT_EQ(pass_managers.Run(subgraph1), SUCCESS);
  // EXPECT_TRUE(CheckMemcpyExist(subgraph1));
  EXPECT_EQ(pass_managers.Run(subgraph2), SUCCESS);
  EXPECT_TRUE(CheckMemcpyExist(subgraph2));
  EXPECT_EQ(pass_managers.Run(subgraph3), SUCCESS);
  EXPECT_TRUE(CheckMemcpyExist(subgraph3));
  EXPECT_EQ(pass_managers.Run(subgraph4), SUCCESS);
  EXPECT_TRUE(CheckMemcpyExist(subgraph4));
  EXPECT_EQ(pass_managers.Run(subgraph5), SUCCESS);
  EXPECT_TRUE(CheckMemcpyExist(subgraph5));
}

TEST_F(UtestGraphPassesSubgraphPass, while_body_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr data_node1 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr data_node2 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr function_node = graph->AddNode(CreateOpDesc("While", WHILE, 2, 2));
  EXPECT_EQ(GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), function_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);

  std::string subgraph_name0 = "instance_cond";
  ComputeGraphPtr subgraph0 = std::make_shared<ComputeGraph>(subgraph_name0);
  size_t index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  function_node->GetOpDesc()->AddSubgraphName("cond");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name0);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  subgraph0->SetParentNode(function_node);
  subgraph0->SetParentGraph(graph);
  graph->AddSubgraph(subgraph0);

  std::string subgraph_name1 = "instance_body";
  ComputeGraphPtr subgraph1 = std::make_shared<ComputeGraph>(subgraph_name1);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 1);
  function_node->GetOpDesc()->AddSubgraphName("body");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name1);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  subgraph1->SetParentNode(function_node);
  subgraph1->SetParentGraph(graph);
  graph->AddSubgraph(subgraph1);

  NodePtr while_node = subgraph1->AddNode(CreateOpDesc("While", WHILE, 2, 2));
  {
    // index-exchange in while body subgraph
    NodePtr data_node1 = subgraph1->AddNode(CreateOpDesc("data", DATA, 1, 1));
    AttrUtils::SetInt(data_node1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    NodePtr data_node2 = subgraph1->AddNode(CreateOpDesc("data", DATA, 1, 1));
    AttrUtils::SetInt(data_node2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
    NodePtr output_node = subgraph1->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 2, 2));
    AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1);
    EXPECT_EQ(GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), while_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), while_node->GetInDataAnchor(1)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(while_node->GetOutDataAnchor(1), output_node->GetInDataAnchor(1)), SUCCESS);
  }

  std::string subsubgraph_name0 = "instance_sub_cond";
  ComputeGraphPtr subsubgraph0 = std::make_shared<ComputeGraph>(subsubgraph_name0);
  index = while_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  while_node->GetOpDesc()->AddSubgraphName("cond");
  EXPECT_EQ(while_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  while_node->GetOpDesc()->SetSubgraphInstanceName(index, subsubgraph_name0);
  EXPECT_EQ(while_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  subsubgraph0->SetParentNode(while_node);
  subsubgraph0->SetParentGraph(subgraph1);
  graph->AddSubgraph(subsubgraph0);

  std::string subsubgraph_name1 = "instance_sub_body";
  ComputeGraphPtr subsubgraph1 = std::make_shared<ComputeGraph>(subsubgraph_name1);
  index = while_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 1);
  while_node->GetOpDesc()->AddSubgraphName("body");
  EXPECT_EQ(while_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  while_node->GetOpDesc()->SetSubgraphInstanceName(index, subsubgraph_name1);
  EXPECT_EQ(while_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  subsubgraph1->SetParentNode(while_node);
  subsubgraph1->SetParentGraph(subgraph1);
  graph->AddSubgraph(subsubgraph1);

  {
    // index-exchange in while body subgraph
    NodePtr data_node1 = subsubgraph1->AddNode(CreateOpDesc("data", DATA, 1, 1));
    AttrUtils::SetInt(data_node1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    NodePtr data_node2 = subsubgraph1->AddNode(CreateOpDesc("data", DATA, 1, 1));
    AttrUtils::SetInt(data_node2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
    NodePtr output_node = subsubgraph1->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 2, 2));
    AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1);
    EXPECT_EQ(GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), output_node->GetInDataAnchor(1)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(graph));
  EXPECT_EQ(CheckMemcpyNum(subgraph1), 3); // 1 for while input link to other nodes, 1 for data exchange in while-body
  EXPECT_EQ(CheckMemcpyNum(subsubgraph1), 2);

  {
    auto output_node = subsubgraph1->FindNode(NODE_NAME_NET_OUTPUT);
    output_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-1,-1}));
    output_node->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({-1,-1}));
    PassManager pass_managers;
    pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
    EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  }
}

TEST_F(UtestGraphPassesSubgraphPass, while_body_no_body_subgraph) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr data_node1 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr data_node2 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr function_node = graph->AddNode(CreateOpDesc("While", WHILE, 2, 2));
  EXPECT_EQ(GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), function_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);

  std::string subgraph_name = "instance_body";
  ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>(subgraph_name);
  size_t index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  function_node->GetOpDesc()->AddSubgraphName("body");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  subgraph->SetParentNode(function_node);
  subgraph->SetParentGraph(graph);

  {
    // index-exchange in while body subgraph
    NodePtr data_node1 = subgraph->AddNode(CreateOpDesc("data", DATA, 1, 1));
    AttrUtils::SetInt(data_node1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    NodePtr data_node2 = subgraph->AddNode(CreateOpDesc("data", DATA, 1, 1));
    AttrUtils::SetInt(data_node2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
    NodePtr output_node = subgraph->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 2, 2));
    AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1);
    EXPECT_EQ(GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), output_node->GetInDataAnchor(1)), SUCCESS);
  }

  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_NE(pass_managers.Run(graph), SUCCESS);
}

TEST_F(UtestGraphPassesSubgraphPass, while_body_no_net_output) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr data_node1 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr data_node2 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr function_node = graph->AddNode(CreateOpDesc("While", WHILE, 2, 2));
  EXPECT_EQ(GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), function_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);

  std::string subgraph_name = "instance_body";
  ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>(subgraph_name);
  size_t index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  function_node->GetOpDesc()->AddSubgraphName("body");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  subgraph->SetParentNode(function_node);
  subgraph->SetParentGraph(graph);
  graph->AddSubgraph(subgraph);

  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_NE(pass_managers.Run(graph), SUCCESS);
}
/*
 * 当while在partitioncall中时，若其输入为const且在另外一个partitioncall中，这种场景也需要插入identity
 * 否则静态while执行时会将输入改坏
 * +----------------------------------------------------------------------------------------------------------------+
 * | Partitioncall2                          +---------------------------+   +------------------------------------+ |
 * |                                         | Cond Graph                |   | Body Graph                         | |
 * |                   NetOutput             |      NetOutput            |   |      NetOutput  <------------+     | |
 * |                     |                   |         |                 |   |      /       \               |     | |
 * |                     +-------------------+        Foo                |   |    Bar       Add             |     | |
 * |                     |0,1,2              |      /     \              |   |    |        /   \            |     | |
 * |       +------>    while  <---------+    | input0   max_value(Const) |   |  input0 input1  one(Const) input2  | |
 * |       |             |              |    +---------------------------+   +------------------------------------+ |
 * |input(Data)  loop_counter(Data)   max_iterations(Data)                                                          |
 * +----------------------------------------------------------------------------------------------------------------+
 *             |0              |1      |2
 *             |               |       |
 *         input(Data)         |0      |1
 *                +---------------------------------------------------------+
 *                |Partitioncall1                                           |
 *                |                    NetOutput<-----------+               |
 *                |                    |                    |               |
 *                |    loop_counter(Const)  max_iterations(Const)           |
 *                +---------------------------------------------------------+
 */
TEST_F(UtestGraphPassesSubgraphPass, while_input_is_const_which_cross_paritioncall) {
  auto graph = gert::ShareGraph::WhileGraphInPartitionCall(true);
  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  // TODO:
  // for (const auto &node : graph->GetAllNodes()) {
  //   if (node->GetType() == WHILE) {
  //     // CHECK INPUT 1\2 is identity
  //     auto peer_node_1 = node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  //     EXPECT_EQ(peer_node_1->GetType(), IDENTITY);
  //     auto peer_node_2 = node->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode();
  //     EXPECT_EQ(peer_node_2->GetType(), IDENTITY);
  //   }
  // }
}

/**
 * 
 *                          +-----------+  +-----------+
 *                          |Then Graph |  |Else Graph |
 *                          |           |  |           |
 *                          | NetOutput |  | NetOutput |
 *       NetOutput          |  \ | /    |  |   |       |
 *         \ | /            |StaticFoo  |  | StaticFoo |
 *          if  <---------> |   |       |  |   |       |
 *        /    \            | Data(0)   |  | Data(1)   |
 *      Data   Data         +-----------+  +-----------+
 *                  
 *                        ||
 *                        ||
 *
 *                          Then Graph                              
 *                          +-----------------+  Else Graph   
 *                          |   NetOutput     |  +-----------+
 *                          |   /  |   \      |  |           |
 *                          |Iden Iden  |     |  | NetOutput |
 *        NetOutput         |   \  |   /      |  |     |     |
 *         \ | /            |   StaticFoo     |  | StaticFoo |
 *          if  <---------> |      |          |  |     |     |
 *        /    \            |    Data(0)      |  | Data(1)   |
 *      Data   Data         +-----------------+  +-----------+
 * 
 */
TEST_F(UtestGraphPassesSubgraphPass, condition_subgraph_symbol_conflict) {
  auto graph = gert::ShareGraph::IfWithKnownSubGraphAndMultiOutputs("test");
  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == IF) {
      auto then_graph = NodeUtils::GetSubgraph(*node, 0);
      auto netoutput_of_then = then_graph->FindFirstNodeMatchType(NETOUTPUT);
      std::unordered_map<std::string, int> node_count{{IDENTITY, 0}, {"StaticFoo", 0}};
      std::unordered_map<std::string, int> expect_node_count{{IDENTITY, 2}, {"StaticFoo", 1}};
      // collect inputs of netoutput
      for (const auto &in_node : netoutput_of_then->GetInDataNodes()) {
        node_count[in_node->GetType()]++;
      }
      for (const auto &node_type_2_count : node_count) {
        EXPECT_EQ(node_type_2_count.second, expect_node_count[node_type_2_count.first]);
      }

      auto else_graph = NodeUtils::GetSubgraph(*node, 1);
      auto netoutput_of_else = else_graph->FindFirstNodeMatchType(NETOUTPUT);
      std::unordered_map<std::string, int> expect_node_count_else{{"StaticFoo", 0}};
      // collect inputs of netoutput
      for (const auto &in_node : netoutput_of_else->GetInDataNodes()) {
        expect_node_count_else[in_node->GetType()]++;
      }
      for (const auto &node_type_2_count : expect_node_count_else) {
        EXPECT_EQ(node_type_2_count.second, 1);
      }
    }
  }
}

/*
 *     data
 *      |
 *      a
 *      |
 *  partitioned_call    +----------------------+
 *      |               | inner_data           |
 *      |               |     |                |
 *      |               |  reshape1            |
 *      |               |     |                |
 *      |               | continue_node        |
 *      |               |     |                |
 *      |               |  reshape2            |
 *      |               |     |                |
 *      b               | netoutput2           |
 *      |               +----------------------+
 *    netoutput1
 */
TEST_F(UtestGraphPassesSubgraphPass, ContinueNodeWithSubGraphEdgeThroughRefNode_InsertMemCopy) {
  const auto inner_data = OP_CFG(DATA).ParentNodeIndex(0);
  const auto reshape = OP_CFG(RELU).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("inner_data", inner_data)->NODE("reshape1", reshape)
              ->NODE("continue_node", RELU)->NODE("reshape2", reshape)->NODE("netoutput2", NETOUTPUT));
  };
  sub_1.Layout();
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("a", RELU)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
              ->NODE("b", RELU)->NODE("netoutput1", NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(false);
  auto partitioned_call1_graph = compute_graph->GetAllSubgraphs().at(0);
  auto netoutput2 = partitioned_call1_graph->FindNode("netoutput2");
  AttrUtils::SetInt(netoutput2->GetOpDescBarePtr()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto continue_node = partitioned_call1_graph->FindNode("continue_node");
  AttrUtils::SetBool(continue_node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  AttrUtils::SetBool(continue_node->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);

  compute_graph->TopologicalSorting();
  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_EQ(pass_managers.Run(partitioned_call1_graph), SUCCESS);
  EXPECT_EQ(netoutput2->GetInNodes().at(0)->GetType(), IDENTITY);

  auto data_node = partitioned_call1_graph->FindNode("inner_data");
  EXPECT_EQ(data_node->GetOutNodes().at(0)->GetType(), IDENTITY);
}

TEST_F(UtestGraphPassesSubgraphPass, hcom_to_netoutput_discard_reuse_input_attr) {
  std::vector<int64_t> shape{1, -1};
  auto data_1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("data_1");
  auto hcom_1 = OP_CFG(HCOMBROADCAST)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(2)
      .Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
      .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true)
      .Build("hcom_1");
  DEF_GRAPH(subgraph_1) {
    CHAIN(NODE(data_1)->NODE(hcom_1)->NODE("output_1", NETOUTPUT));
  };
  auto subgraph = ToComputeGraph(subgraph_1);
  ge::AttrUtils::SetInt(subgraph->FindNode("data_1")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto hcom_op = subgraph->FindNode("hcom_1")->GetOpDesc();
  auto tensor_desc = hcom_op->MutableOutputDesc(0);
  ge::TensorUtils::SetReuseInput(*tensor_desc, true);
  hcom_op->UpdateOutputDesc(0, *tensor_desc);

  bool reuse_flag = false;
  TensorUtils::GetReuseInput(hcom_op->GetOutputDesc(0), reuse_flag);
  EXPECT_EQ(reuse_flag, true);

  DEF_GRAPH(main_graph_1) {
    CHAIN(NODE("data_0", DATA)
              ->EDGE(0, 0)
              ->NODE("partitioncall_0", PARTITIONEDCALL)
              ->EDGE(0, 0)
              ->NODE("output_0", NETOUTPUT));
  };
  auto main_graph = ToComputeGraph(main_graph_1);

  auto p1 = main_graph->FindNode("partitioncall_0");
  p1->GetOpDesc()->AddSubgraphName(subgraph->GetName());
  p1->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  subgraph->SetParentGraph(main_graph);
  subgraph->SetParentNode(p1);
  ge::AttrUtils::SetInt(main_graph->FindNode("data_0")->GetOpDesc(), "index", 0);
  PassManager pass_managers;
  pass_managers.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_EQ(pass_managers.Run(subgraph), SUCCESS);
  EXPECT_NE(subgraph->FindNode("output_1_input_0_Memcpy"), nullptr);
  bool after_reuse_flag = false;
  TensorUtils::GetReuseInput(subgraph->FindNode("output_1_input_0_Memcpy")->GetOpDesc()->GetOutputDesc(0),
                             after_reuse_flag);
  EXPECT_TRUE(after_reuse_flag == false);
}
}  // namespace ge
