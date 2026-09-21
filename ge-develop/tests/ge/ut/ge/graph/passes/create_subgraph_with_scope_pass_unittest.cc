/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "gtest/gtest.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/multi_batch/create_subgraph_with_scope_pass.h"
#include "macro_utils/dt_public_unscope.h"

#include "framework/common/types.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"
#include "graph/passes/pass_manager.h"
#include "graph/utils/node_utils.h"
#include "graph/ge_local_context.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/share_graph.h"

namespace ge {
class CreateSubgraphWithScopePassTest : public testing::Test {
 public:
  NodePtr MakeNode(const ComputeGraphPtr &graph, int in_num, int out_num, string name, string type,
                   const Format format = FORMAT_ND, const DataType dt = DT_FLOAT) {
    GeTensorDesc test_desc(GeShape(), format, dt);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (auto i = 0; i < in_num; ++i) {
      op_desc->AddInputDesc(test_desc);
    }
    for (auto i = 0; i < out_num; ++i) {
      op_desc->AddOutputDesc(test_desc);
    }
    return graph->AddNode(op_desc);
  }

  void CreateRootGraph(ComputeGraphPtr &graph) {
    auto data0 = MakeNode(graph, 0, 1, "data_0", DATA);
    GeTensorDesc tensor_desc0(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
    data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);

    auto data1 = MakeNode(graph, 0, 1, "data_1", DATA);
    GeTensorDesc tensor_desc1(GeShape({1,224,224,3}), FORMAT_NHWC, DT_FLOAT);
    data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc1);

    auto cast = MakeNode(graph, 1, 1, "cast0", CAST);
    GeTensorDesc tensor_desc2(GeShape({-1,224,224,3}), FORMAT_NHWC, DT_FLOAT);
    cast->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
    cast->GetOpDesc()->UpdateOutputDesc(0, tensor_desc2);

    auto partitioned_call = MakeNode(graph, 2, 1, "partitioned_call", PARTITIONEDCALL);
    GeTensorDesc tensor_desc(GeShape({-1,224,224,3}), FORMAT_NHWC, DT_FLOAT);
    partitioned_call->GetOpDesc()->UpdateInputDesc(0, tensor_desc1);
    partitioned_call->GetOpDesc()->UpdateInputDesc(1, tensor_desc2);
    partitioned_call->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);

    auto cast1 = MakeNode(graph, 1, 1, "cast1", CAST);
    cast1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
    cast1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);

    auto netoutput = MakeNode(graph, 1, 0, "netoutput", NETOUTPUT);
    netoutput->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);

    GraphUtils::AddEdge(data0->GetOutDataAnchor(0), cast->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), partitioned_call->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast->GetOutDataAnchor(0), partitioned_call->GetInDataAnchor(1));
    GraphUtils::AddEdge(partitioned_call->GetOutDataAnchor(0), cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast1->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  }

  void CreateSubGraph(ComputeGraphPtr &graph) {
    ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>("nn_model");
    auto partition_op = graph->FindNode("partitioned_call");
    partition_op->GetOpDesc()->AddSubgraphName(subgraph->GetName());
    partition_op->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
    subgraph->SetParentNode(partition_op);
    subgraph->SetParentGraph(graph);
    graph->AddSubgraph(subgraph->GetName(), subgraph);

    auto data0 = MakeNode(subgraph, 1, 1, "nn_data0", DATA);
    GeTensorDesc tensor_desc0(GeShape({1,224,224,3}), FORMAT_NHWC, DT_FLOAT);
    data0->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
    data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);
    (void)AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);

    auto data1 = MakeNode(subgraph, 1, 1, "nn_data1", DATA);
    GeTensorDesc tensor_desc1(GeShape({-1,224,224,3}), FORMAT_NHWC, DT_FLOAT);
    data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc1);
    data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc1);
    (void)AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

    auto add = MakeNode(subgraph, 2, 1, "nn_add", ADD);
    add->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
    add->GetOpDesc()->UpdateInputDesc(1, tensor_desc1);
    add->GetOpDesc()->UpdateOutputDesc(0, tensor_desc1);

    auto netoutput = MakeNode(subgraph, 1, 0, "nn_netoutput", NETOUTPUT);
    (void)AttrUtils::SetInt(&tensor_desc1, ATTR_NAME_PARENT_NODE_INDEX, 0);
    netoutput->GetOpDesc()->UpdateInputDesc(0, tensor_desc1);

    GraphUtils::AddEdge(data0->GetOutDataAnchor(0), add->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), add->GetInDataAnchor(1));
    GraphUtils::AddEdge(add->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  }
  
  void CreateDynamicDimsGraph(ComputeGraphPtr &graph) {
    auto data0 = MakeNode(graph, 0, 1, "data_0", DATA);
    GeTensorDesc tensor_desc0(GeShape({10}));
    data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);

    auto data1 = MakeNode(graph, 0, 1, "data_1", DATA);
    data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);

    auto add0 = MakeNode(graph, 2, 1, "add0", ADD);
    add0->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
    add0->GetOpDesc()->UpdateInputDesc(1, tensor_desc0);
    add0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);


    auto netoutput = MakeNode(graph, 1, 0, "netoutput", NETOUTPUT);
    netoutput->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);

    GraphUtils::AddEdge(data0->GetOutDataAnchor(0), add0->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), add0->GetInDataAnchor(1));
    GraphUtils::AddEdge(add0->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  }
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(CreateSubgraphWithScopePassTest, base_graph_success) {
  PassManager pass_manager;
  pass_manager.AddPass("CreateSubGraphWithScopePass", new (std::nothrow) CreateSubGraphWithScopePass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  CreateRootGraph(graph);
  CreateSubGraph(graph);

  auto cast = graph->FindNode("cast0");
  AttrUtils::SetStr(cast->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, "0:-1,224,224,3");
  AttrUtils::SetStr(cast->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "1, 2, 4");
  std::vector<NodePtr> scope;
  scope.push_back(graph->FindNode("partitioned_call"));
  scope.push_back(graph->FindNode("cast0"));
  scope.push_back(graph->FindNode("cast1"));
  for (auto n : scope) {
    AttrUtils::SetInt(n->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);
  }

  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  bool is_multi_dims = false;
  auto subgraph = graph->GetSubgraph("nn_model");
  EXPECT_EQ(AttrUtils::GetBool(subgraph, ATTR_NAME_SUBGRAPH_IS_MULTI_DIMS, is_multi_dims), true);
  EXPECT_EQ(is_multi_dims, true);
  EXPECT_EQ(subgraph->GetAllNodesSize(), 6);
}

TEST_F(CreateSubgraphWithScopePassTest, test_empty_dynamic_node) {
  CreateSubGraphWithScopePass subgraph_pass;
  EXPECT_EQ(subgraph_pass.UpdateDynamicConfigAttrs({}), SUCCESS);
}

TEST_F(CreateSubgraphWithScopePassTest, control_edge_failed1) {
  PassManager pass_manager;
  pass_manager.AddPass("CreateSubGraphWithScopePass", new (std::nothrow) CreateSubGraphWithScopePass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  CreateRootGraph(graph);
  CreateSubGraph(graph);

  auto data = graph->FindNode("data_0");
  auto cast = graph->FindNode("cast0");
  (void)GraphUtils::AddEdge(data->GetOutControlAnchor(), cast->GetInControlAnchor());
  std::vector<NodePtr> scope;
  scope.push_back(graph->FindNode("partitioned_call"));
  scope.push_back(graph->FindNode("cast0"));
  scope.push_back(graph->FindNode("cast1"));
  for (auto n : scope) {
    AttrUtils::SetInt(n->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);
  }

  EXPECT_EQ(pass_manager.Run(graph), FAILED);
}

TEST_F(CreateSubgraphWithScopePassTest, control_edge_failed2) {
  PassManager pass_manager;
  pass_manager.AddPass("CreateSubGraphWithScopePass", new (std::nothrow) CreateSubGraphWithScopePass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  CreateRootGraph(graph);
  CreateSubGraph(graph);

  auto cast = graph->FindNode("cast1");
  auto output = graph->FindNode("netoutput");
  (void)GraphUtils::AddEdge(cast->GetOutControlAnchor(), output->GetInControlAnchor());
  std::vector<NodePtr> scope;
  scope.push_back(graph->FindNode("partitioned_call"));
  scope.push_back(graph->FindNode("cast0"));
  scope.push_back(graph->FindNode("cast1"));
  for (auto n : scope) {
    AttrUtils::SetInt(n->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);
  }

  EXPECT_EQ(pass_manager.Run(graph), FAILED);
}

TEST_F(CreateSubgraphWithScopePassTest, partitioned_call_success) {
  PassManager pass_manager;
  pass_manager.AddPass("CreateSubGraphWithScopePass", new (std::nothrow) CreateSubGraphWithScopePass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  CreateRootGraph(graph);
  CreateSubGraph(graph);

  auto partitioned_call = graph->FindNode("partitioned_call");
  AttrUtils::SetInt(partitioned_call->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);
  AttrUtils::SetStr(partitioned_call->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE,
                    "0:1,224,224,3;1:-1,224,224,3");
  AttrUtils::SetStr(partitioned_call->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "1, 2, 4");

  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  bool is_multi_dims = false;
  auto subgraph = graph->GetSubgraph("nn_model");
  EXPECT_EQ(AttrUtils::GetBool(subgraph, ATTR_NAME_SUBGRAPH_IS_MULTI_DIMS, is_multi_dims), true);
  EXPECT_EQ(is_multi_dims, true);
  EXPECT_EQ(subgraph->GetAllNodesSize(), 4);
}

TEST_F(CreateSubgraphWithScopePassTest, new_subgraph_success) {
  PassManager pass_manager;
  pass_manager.AddPass("CreateSubGraphWithScopePass", new (std::nothrow) CreateSubGraphWithScopePass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  auto data0 = MakeNode(graph, 0, 1, "data_0", DATA);
  GeTensorDesc tensor_desc0(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
  data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);

  auto data1 = MakeNode(graph, 0, 1, "data_1", DATA);
  GeTensorDesc tensor_desc1(GeShape({1,224,224,3}), FORMAT_NHWC, DT_FLOAT);
  data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc1);

  auto cast = MakeNode(graph, 1, 1, "cast", CAST);
  GeTensorDesc tensor_desc2(GeShape({-1,224,224,3}), FORMAT_NHWC, DT_FLOAT);
  cast->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
  cast->GetOpDesc()->UpdateOutputDesc(0, tensor_desc2);
  AttrUtils::SetInt(cast->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);
  AttrUtils::SetStr(cast->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, "0:-1,224,224,3");
  AttrUtils::SetStr(cast->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "1, 2, 4");

  auto add = MakeNode(graph, 2, 1, "add", ADD);
  add->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
  add->GetOpDesc()->UpdateInputDesc(1, tensor_desc2);
  add->GetOpDesc()->UpdateOutputDesc(0, tensor_desc2);
  AttrUtils::SetInt(add->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);

  auto netoutput = MakeNode(graph, 1, 0, "netoutput", NETOUTPUT);
  netoutput->GetOpDesc()->UpdateInputDesc(0, tensor_desc2);

  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), cast->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), add->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast->GetOutDataAnchor(0), add->GetInDataAnchor(1));
  GraphUtils::AddEdge(add->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));

  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);
  auto subgraph = graph->GetAllSubgraphs().at(0);
  bool is_multi_dims = false;
  EXPECT_EQ(AttrUtils::GetBool(subgraph, ATTR_NAME_SUBGRAPH_IS_MULTI_DIMS, is_multi_dims), true);
  EXPECT_EQ(is_multi_dims, true);
  EXPECT_EQ(subgraph->GetAllNodesSize(), 5);
}

TEST_F(CreateSubgraphWithScopePassTest, base_dynamic_dims_graph_success) {
  std::map<std::string, std::string> options{
      {"ge.inputShape", "add0:-1;add1:-1"}, {"ge.dynamicDims", "1,1;10,10;20,20"}, {"ge.dynamicNodeType", "1"}};
  GetThreadLocalContext().SetGlobalOption(options);

  PassManager pass_manager;
  pass_manager.AddPass("CreateSubGraphWithScopePass", new (std::nothrow) CreateSubGraphWithScopePass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  CreateDynamicDimsGraph(graph);
  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  auto add0 = graph->FindNode("add0");
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, "0:-1;1:-1");
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "1,1;10,10;20,20");
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(CreateSubgraphWithScopePassTest, one_data_two_output_success) {
    std::map<std::string, std::string> options{
            {"ge.inputShape", "add0:-1"}, {"ge.dynamicDims", "1;10;20"}, {"ge.dynamicNodeType", "1"}};
    GetThreadLocalContext().SetGlobalOption(options);

    PassManager pass_manager;
    pass_manager.AddPass("CreateSubGraphWithScopePass", new (std::nothrow) CreateSubGraphWithScopePass);
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

    auto data0 = MakeNode(graph, 0, 1, "data_0", DATA);
    GeTensorDesc tensor_desc0(GeShape({10}));
    data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);

    auto add0 = MakeNode(graph, 2, 1, "add0", ADD);
    add0->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
    add0->GetOpDesc()->UpdateInputDesc(1, tensor_desc0);
    add0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);

    auto netoutput = MakeNode(graph, 1, 0, "netoutput", NETOUTPUT);
    netoutput->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);

    GraphUtils::AddEdge(data0->GetOutDataAnchor(0), add0->GetInDataAnchor(0));
    GraphUtils::AddEdge(data0->GetOutDataAnchor(0), add0->GetInDataAnchor(1));
    GraphUtils::AddEdge(add0->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
    setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);

    add0 = graph->FindNode("add0");
    AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, "0:-1;1:-1");
    AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "1,1;10,10;20,20");
    EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
    unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(CreateSubgraphWithScopePassTest, output_with_ctrl_edge_success) {
  std::map<std::string, std::string> options{
      {"ge.inputShape", "add0:-1"}, {"ge.dynamicDims", "1;10;20"}, {"ge.dynamicNodeType", "1"}};
  GetThreadLocalContext().SetGlobalOption(options);

  PassManager pass_manager;
  pass_manager.AddPass("CreateSubGraphWithScopePass", new (std::nothrow) CreateSubGraphWithScopePass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  auto data0 = MakeNode(graph, 0, 1, "data_0", DATA);
  GeTensorDesc tensor_desc0(GeShape({10}));
  data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);

  auto add0 = MakeNode(graph, 2, 1, "add0", ADD);
  add0->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
  add0->GetOpDesc()->UpdateInputDesc(1, tensor_desc0);
  add0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);


  auto netoutput = MakeNode(graph, 1, 0, "netoutput", NETOUTPUT);
  netoutput->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);

  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), add0->GetInDataAnchor(0));
  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), add0->GetInDataAnchor(1));
  GraphUtils::AddEdge(add0->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(add0->GetOutControlAnchor(), netoutput->GetInControlAnchor());
  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);

  add0 = graph->FindNode("add0");
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, "0:-1;1:-1");
  AttrUtils::SetStr(add0->GetOpDesc(), ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, "1,1;10,10;20,20");
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(CreateSubgraphWithScopePassTest, global_graph_trans_to_subgraph) {
  std::map<std::string, std::string> options{
          {"ge.inputShape", "data0:1,1,-1,-1;data1:-1,-1;data2:-1,-1;data3:-1,-1;data4:-1,-1;data5:-1,-1;data6:-1,-1;data7:-1;data8:-1,-1;data9:-1,-1"},
          {"ge.dynamicDims", "2,3,2,3,2,3,2,3,2,3,2,3,2,3,1,2,3,2,3;3,4,3,4,3,4,3,4,3,4,3,4,3,4,1,3,4,3,4"},
          {"ge.dynamicNodeType", "1"}};
  GetThreadLocalContext().SetGlobalOption(options);
  auto graph = gert::ShareGraph::IFASingleGraph();
  PassManager pass_manager;
  pass_manager.AddPass("CreateSubGraphWithScopePass", new (std::nothrow) CreateSubGraphWithScopePass);
  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
}
}  // namespace ge
