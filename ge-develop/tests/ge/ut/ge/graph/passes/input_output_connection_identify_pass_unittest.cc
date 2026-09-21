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

#include "common/plugin/ge_make_unique_util.h"
#include "graph/node.h"
#include "graph/compute_graph.h"
#include "graph/passes/feature/input_output_connection_identify_pass.h"
#include "graph/passes/pass_manager.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
namespace {
class UtestGraphPassesInputOutputIdentifyPass : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}

  OpDescPtr CreateOpDesc(const std::string name, const std::string type, uint32_t input_num, uint32_t output_num) {
    OpDescPtr op_desc = shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name, type));
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

  uint32_t CheckInputAttrNum(const ComputeGraphPtr &graph) {
    uint32_t num = 0;
    for (auto &node : graph->GetAllNodes()) {
      if (AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_INPUT)) {
        // std::cout<<"input connect node name "<<node->GetName()<<std::endl;
        num++;
      }
    }
    return num;
  }

  uint32_t CheckOutputAttrNum(const ComputeGraphPtr &graph) {
    uint32_t num = 0;
    for (auto &node : graph->GetAllNodes()) {
      if (AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_OUTPUT)) {
        // std::cout<<" output connect node name "<<node->GetName()<<std::endl;
        num++;
      }
    }
    return num;
  }
};
} // namespace

TEST_F(UtestGraphPassesInputOutputIdentifyPass, sub_graph_input_success) {
  /**
   * root graph:     / identity \
   * root graph:  data->Case->NetOutput
   * root graph: data1 /
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("root_graph");
  NodePtr data_node_0 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));
  NodePtr data_node_1 = graph->AddNode(CreateOpDesc("data1", DATA, 1, 1));
  NodePtr data_node_2 = graph->AddNode(CreateOpDesc("data2", DATA, 1, 1));
  NodePtr const_node = graph->AddNode(CreateOpDesc("const", CONSTANTOP, 0, 1));
  NodePtr function_node = graph->AddNode(CreateOpDesc("Case", SQUEEZE, 4, 1));
  NodePtr part_node = graph->AddNode(CreateOpDesc("part", SQUEEZE, 1, 1));
  NodePtr part_node_1 = graph->AddNode(CreateOpDesc("part1", SQUEEZE, 1, 2));
  NodePtr part_node_2 = graph->AddNode(CreateOpDesc("part2", SQUEEZE, 2, 2));
  NodePtr squeeze_node = graph->AddNode(CreateOpDesc("Squeeze", SQUEEZE, 1, 1));
  NodePtr identity_node = graph->AddNode(CreateOpDesc("Identity", IDENTITY, 1, 1));
  NodePtr output_node = graph->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 3, 1));
  NodePtr out_node_1 = graph->AddNode(CreateOpDesc("out1", IDENTITY, 4, 1));
  NodePtr out_node_2 = graph->AddNode(CreateOpDesc("out2", IDENTITY, 1, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), squeeze_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), identity_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), function_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), function_node->GetInDataAnchor(1)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node_2->GetOutDataAnchor(0), function_node->GetInDataAnchor(2)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), function_node->GetInDataAnchor(3)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(function_node->GetOutDataAnchor(0), part_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(part_node->GetOutDataAnchor(0), part_node_1->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(identity_node->GetOutDataAnchor(0), part_node_2->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(squeeze_node->GetOutDataAnchor(0), part_node_2->GetInDataAnchor(1)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(part_node_1->GetOutDataAnchor(0), out_node_1->GetInDataAnchor(1)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(part_node_1->GetOutDataAnchor(0), out_node_1->GetInDataAnchor(2)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(part_node_1->GetOutDataAnchor(1), out_node_1->GetInDataAnchor(3)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(part_node_1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(part_node_1->GetOutDataAnchor(1), output_node->GetInDataAnchor(1)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(part_node_2->GetOutDataAnchor(0), output_node->GetInDataAnchor(2)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(part_node_2->GetOutDataAnchor(1), out_node_2->GetInDataAnchor(0)), SUCCESS);

  std::string subgraph_name1 = "instance_branch1";
  ComputeGraphPtr subgraph1 = std::make_shared<ComputeGraph>(subgraph_name1);
  subgraph1->SetParentNode(function_node);
  subgraph1->SetParentGraph(graph);
  graph->AddSubgraph(subgraph_name1, subgraph1);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);
  size_t index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  function_node->GetOpDesc()->AddSubgraphName("branch1");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name1);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  {
    // subgraph1: Const->Sqrt->NetOutput
    NodePtr data_node = subgraph1->AddNode(CreateOpDesc("subgraph1_data", DATA, 1, 1));
    EXPECT_EQ(AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
    NodePtr a_node = subgraph1->AddNode(CreateOpDesc("subgraph1_a", SQRT, 1, 1));
    NodePtr data_node_2 = subgraph1->AddNode(CreateOpDesc("subgraph1_data2", DATA, 1, 1));
    EXPECT_EQ(AttrUtils::SetInt(data_node_2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1), true);
    NodePtr b_node = subgraph1->AddNode(CreateOpDesc("subgraph1_b", SQRT, 1, 1));
    NodePtr const_node = subgraph1->AddNode(CreateOpDesc("subgraph1_const", CONSTANTOP, 0, 1));

    NodePtr output_node = subgraph1->AddNode(CreateOpDesc("subgraph1_netoutput", NETOUTPUT, 3, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), a_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(data_node_2->GetOutDataAnchor(0), b_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(1)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(2)), SUCCESS);
  }

  std::string subgraph_name2 = "instance_branch2";
  ComputeGraphPtr subgraph2 = std::make_shared<ComputeGraph>(subgraph_name2);
  subgraph2->SetParentNode(function_node);
  subgraph2->SetParentGraph(graph);
  graph->AddSubgraph(subgraph_name2, subgraph2);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);
  index = function_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 1);
  function_node->GetOpDesc()->AddSubgraphName("branch2");
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  function_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name2);
  EXPECT_EQ(function_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);

  {
    // subgraph2: Data->NetOutput
    NodePtr data_node = subgraph2->AddNode(CreateOpDesc("subgraph2_data1", DATA, 1, 1));
    // EXPECT_EQ(AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3), TRUE);
    NodePtr var_node = subgraph2->AddNode(CreateOpDesc("subgraph2_data2", DATA, 1, 1));
    EXPECT_EQ(AttrUtils::SetInt(var_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2), true);
    NodePtr sg_node = subgraph2->AddNode(CreateOpDesc("subgraph2_sg", SQRT, 1, 1));
    NodePtr output_node = subgraph2->AddNode(CreateOpDesc("subgraph2_netoutput", NETOUTPUT, 2, 1));
    EXPECT_EQ(AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(var_node->GetOutDataAnchor(0), sg_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(sg_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(1)), SUCCESS);

    std::string subgraph_name12 = "instance_branch12";
    ComputeGraphPtr subgraph12 = std::make_shared<ComputeGraph>(subgraph_name12);
    subgraph12->SetParentNode(sg_node);
    subgraph12->SetParentGraph(subgraph2);
    graph->AddSubgraph(subgraph_name12, subgraph12);
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 3);
    index = sg_node->GetOpDesc()->GetSubgraphInstanceNames().size();
    EXPECT_EQ(index, 0);
    sg_node->GetOpDesc()->AddSubgraphName("branch12");
    EXPECT_EQ(sg_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
    sg_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name12);
    EXPECT_EQ(sg_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

    {
      // subgraph12: Const->Sqrt->NetOutput
      NodePtr data_node_1 = subgraph12->AddNode(CreateOpDesc("subgraph12_data1", DATA, 1, 1));
      EXPECT_EQ(AttrUtils::SetInt(data_node_1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
      NodePtr a_node = subgraph12->AddNode(CreateOpDesc("subgraph12_a", SQRT, 2, 1));
      NodePtr data_node_2 = subgraph12->AddNode(CreateOpDesc("subgraph12_data2", DATA, 1, 1));
      NodePtr b_node = subgraph12->AddNode(CreateOpDesc("subgraph12_b", SQRT, 1, 1));
      NodePtr output_node = subgraph12->AddNode(CreateOpDesc("subgraph12_netoutput", NETOUTPUT, 3, 1));
      EXPECT_EQ(AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
      EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), a_node->GetInDataAnchor(1)), SUCCESS);
      EXPECT_EQ(GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
      EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), output_node->GetInDataAnchor(1)), SUCCESS);
      EXPECT_EQ(GraphUtils::AddEdge(data_node_2->GetOutDataAnchor(0), b_node->GetInDataAnchor(0)), SUCCESS);
      EXPECT_EQ(GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(2)), SUCCESS);
      
    }
  }

  std::string subgraph_name3 = "instance_branch3";
  ComputeGraphPtr subgraph3 = std::make_shared<ComputeGraph>(subgraph_name3);
  subgraph3->SetParentNode(part_node_1);
  subgraph3->SetParentGraph(graph);
  graph->AddSubgraph(subgraph_name3, subgraph3);
  index = part_node_1->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  part_node_1->GetOpDesc()->AddSubgraphName("branch3");
  EXPECT_EQ(part_node_1->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  part_node_1->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name3);
  EXPECT_EQ(part_node_1->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  {
    // subgraph3: node->NetOutput 
    NodePtr node_1 = subgraph3->AddNode(CreateOpDesc("subgraph3_node1", VARIABLE, 1, 1));
    EXPECT_EQ(AttrUtils::SetInt(node_1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
    NodePtr node_2 = subgraph3->AddNode(CreateOpDesc("subgraph3_node2", VARIABLE, 1, 1));
    EXPECT_EQ(AttrUtils::SetInt(node_2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
    NodePtr node_a = subgraph3->AddNode(CreateOpDesc("subgraph3_a", SQRT, 1, 1));
    NodePtr node_out = subgraph3->AddNode(CreateOpDesc("subgraph3_out", SQRT, 1, 2));
    NodePtr output_node = subgraph3->AddNode(CreateOpDesc("subgraph3_netoutput", NETOUTPUT, 2, 1));
    EXPECT_EQ(AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
    EXPECT_EQ(AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1), true);
    EXPECT_EQ(GraphUtils::AddEdge(node_1->GetOutDataAnchor(0), node_a->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(node_2->GetOutDataAnchor(0), output_node->GetInDataAnchor(1)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(node_2->GetOutDataAnchor(0), node_out->GetInDataAnchor(0)), SUCCESS);

    std::string subgraph_name32 = "instance_branch32";
    ComputeGraphPtr subgraph32 = std::make_shared<ComputeGraph>(subgraph_name32);
    subgraph32->SetParentNode(node_out);
    subgraph32->SetParentGraph(subgraph3);
    graph->AddSubgraph(subgraph_name32, subgraph32);
    index = node_out->GetOpDesc()->GetSubgraphInstanceNames().size();
    EXPECT_EQ(index, 0);
    node_out->GetOpDesc()->AddSubgraphName("branch32");
    EXPECT_EQ(node_out->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
    node_out->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name32);
    EXPECT_EQ(node_out->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

    {
      // subgraph32: node->NetOutput 
      NodePtr node_1_32 = subgraph32->AddNode(CreateOpDesc("subgraph32_node1", VARIABLE, 1, 1));
      NodePtr node_2_32 = subgraph32->AddNode(CreateOpDesc("subgraph32_node2", VARIABLE, 1, 1));
      EXPECT_EQ(AttrUtils::SetInt(node_2_32->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
      NodePtr node_out_32 = subgraph32->AddNode(CreateOpDesc("subgraph32_out", SQRT, 2, 1));
      NodePtr output_node_32 = subgraph32->AddNode(CreateOpDesc("subgraph32_netoutput", NETOUTPUT, 2, 1));
      EXPECT_EQ(AttrUtils::SetInt(output_node_32->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
      EXPECT_EQ(AttrUtils::SetInt(output_node_32->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1), true);
      EXPECT_EQ(GraphUtils::AddEdge(node_1_32->GetOutDataAnchor(0), output_node_32->GetInDataAnchor(1)), SUCCESS);
      EXPECT_EQ(GraphUtils::AddEdge(node_2_32->GetOutDataAnchor(0), output_node_32->GetInDataAnchor(0)), SUCCESS);
      EXPECT_EQ(GraphUtils::AddEdge(node_2_32->GetOutDataAnchor(0), node_out_32->GetInDataAnchor(0)), SUCCESS);
      EXPECT_EQ(GraphUtils::AddEdge(node_2_32->GetOutDataAnchor(0), node_out_32->GetInDataAnchor(1)), SUCCESS);
    }
  }

  std::string subgraph_name4 = "instance_branch4";
  ComputeGraphPtr subgraph4 = std::make_shared<ComputeGraph>(subgraph_name4);
  subgraph4->SetParentNode(part_node_2);
  subgraph4->SetParentGraph(graph);
  graph->AddSubgraph(subgraph_name4, subgraph4);
  index = part_node_2->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  part_node_2->GetOpDesc()->AddSubgraphName("branch4");
  EXPECT_EQ(part_node_2->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  part_node_2->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name4);
  EXPECT_EQ(part_node_2->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  {
    // subgraph4: node->NetOutput 
    NodePtr node_1 = subgraph4->AddNode(CreateOpDesc("subgraph4_node1", VARIABLE, 1, 1));
    EXPECT_EQ(AttrUtils::SetInt(node_1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
    NodePtr node_2 = subgraph4->AddNode(CreateOpDesc("subgraph4_node2", VARIABLE, 1, 1));
    EXPECT_EQ(AttrUtils::SetInt(node_2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1), true);
    NodePtr node_a = subgraph4->AddNode(CreateOpDesc("subgraph4_a", IDENTITY, 1, 1));
    NodePtr output_node = subgraph4->AddNode(CreateOpDesc("subgraph4_netoutput", NETOUTPUT, 2, 2));
    EXPECT_EQ(AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
    EXPECT_EQ(AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1), true);
    EXPECT_EQ(GraphUtils::AddEdge(node_1->GetOutDataAnchor(0), node_a->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(node_2->GetOutDataAnchor(0), output_node->GetInDataAnchor(1)), SUCCESS);

    std::string subgraph_name42 = "instance_branch42";
    ComputeGraphPtr subgraph42 = std::make_shared<ComputeGraph>(subgraph_name42);
    subgraph42->SetParentNode(node_a);
    subgraph42->SetParentGraph(subgraph4);
    graph->AddSubgraph(subgraph_name42, subgraph42);
    index = node_a->GetOpDesc()->GetSubgraphInstanceNames().size();
    EXPECT_EQ(index, 0);
    node_a->GetOpDesc()->AddSubgraphName("branch42");
    EXPECT_EQ(node_a->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
    node_a->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name42);
    EXPECT_EQ(node_a->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

    {
      // subgraph42: node->NetOutput 
      NodePtr node_1_42 = subgraph42->AddNode(CreateOpDesc("subgraph42_node1", VARIABLE, 1, 1));
      EXPECT_EQ(AttrUtils::SetInt(node_1_42->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
      NodePtr node_2_42 = subgraph42->AddNode(CreateOpDesc("subgraph42_node2", VARIABLE, 1, 1));
      NodePtr output_node = subgraph42->AddNode(CreateOpDesc("subgraph42_netoutput", NETOUTPUT, 1, 1));
      EXPECT_EQ(AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0), true);
      EXPECT_EQ(GraphUtils::AddEdge(node_1_42->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
      EXPECT_EQ(GraphUtils::AddEdge(node_1_42->GetOutDataAnchor(0), node_2_42->GetInDataAnchor(0)), SUCCESS);
    }
  }

  PassManager pass_managers;
  pass_managers.AddPass("InputOutputConnectionIdentifyPass", new (std::nothrow) InputOutputConnectionIdentifyPass);
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
  EXPECT_EQ(pass_managers.Run(subgraph1), SUCCESS);
  EXPECT_EQ(pass_managers.Run(subgraph2), SUCCESS);

  EXPECT_EQ(CheckInputAttrNum(graph), 29);
  EXPECT_EQ(CheckOutputAttrNum(graph), 24);
}

TEST_F(UtestGraphPassesInputOutputIdentifyPass, nullptr_graph_failed) {
  ge::ComputeGraphPtr graph = nullptr;
  auto my_pass = new (std::nothrow) InputOutputConnectionIdentifyPass;
  EXPECT_EQ(my_pass->Run(graph), PARAM_INVALID);
  delete my_pass;
  my_pass = nullptr;
}

TEST_F(UtestGraphPassesInputOutputIdentifyPass, topo_sort_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("root_graph");
  NodePtr data_node_0 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));

  PassManager pass_managers;
  pass_managers.AddPass("InputOutputConnectionIdentifyPass", new (std::nothrow) InputOutputConnectionIdentifyPass);
  EXPECT_EQ(pass_managers.Run(graph), SUCCESS);
}

TEST_F(UtestGraphPassesInputOutputIdentifyPass, UpdateNodeIdxMap_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("root_graph");
  NodePtr data_node_0 = graph->AddNode(CreateOpDesc("data", DATA, 1, 1));

  string symbol_string = "test";
  Node2Indexs in_map;
  Node2Indexs out_map;
  auto my_pass = new (std::nothrow) InputOutputConnectionIdentifyPass;
  EXPECT_EQ(my_pass->UpdateNodeIdxMap(symbol_string, in_map, out_map), PARAM_INVALID);
  delete my_pass;
  my_pass = nullptr;
}
} // namespace ge
