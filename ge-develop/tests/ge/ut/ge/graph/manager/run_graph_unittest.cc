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
#include <memory>

#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"

#include "macro_utils/dt_public_scope.h"
#include"graph/manager/graph_manager_utils.h"
#include "graph/manager/graph_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
using namespace ge;
using domi::GetContext;

class UtestGraphRunTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { GetContext().out_nodes_map.clear(); }
};

TEST_F(UtestGraphRunTest, RunGraphWithStreamAsync) {
  GraphManager graph_manager;
  GeTensor input0, input1;
  std::vector<GeTensor> inputs{input0, input1};
  std::vector<GeTensor> outputs;
  GraphNodePtr graph_node = std::make_shared<GraphNode>(1);
  graph_manager.AddGraphNode(1, graph_node);
  GraphPtr graph = std::make_shared<Graph>("test");
  graph_node->SetGraph(graph);
  graph_node->SetRunFlag(false);
  graph_node->SetBuildFlag(true);
  Status ret = graph_manager.RunGraphWithStreamAsync(1, nullptr, 0, inputs, outputs);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphRunTest, RunGraphWithStreamAsync1) {
  GraphManager graph_manager;
  GeTensor input0, input1;
  GeTensor output0, output1;
  std::vector<GeTensor> inputs{input0, input1};
  std::vector<GeTensor> outputs{output0, output1};
  GraphNodePtr graph_node = std::make_shared<GraphNode>(1);
  graph_manager.AddGraphNode(1, graph_node);

  auto compute_graph = std::make_shared<ComputeGraph>("test");
  std::vector<std::pair<NodePtr, int32_t>> output_nodes_info;
  OpDescPtr op_desc0 = std::make_shared<OpDesc>("Data", "Data");
  op_desc0->AddInputDesc(GeTensorDesc());
  OpDescPtr op_desc1 = std::make_shared<OpDesc>("Output", "Output");
  op_desc1->AddInputDesc(GeTensorDesc());
  NodePtr out_node0 = compute_graph->AddNode(op_desc0);
  NodePtr out_node1 = compute_graph->AddNode(op_desc1);
  output_nodes_info.emplace_back(std::make_pair(out_node0, 0));
  output_nodes_info.emplace_back(std::make_pair(out_node1, 1));
  compute_graph->SetGraphOutNodesInfo(output_nodes_info);

  GraphPtr graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  graph_node->SetGraph(graph);
  graph_node->SetRunFlag(false);
  graph_node->SetBuildFlag(false);
  Status ret = graph_manager.RunGraphWithStreamAsync(1, nullptr, 0, inputs, outputs);
  EXPECT_NE(ret, SUCCESS);
}
