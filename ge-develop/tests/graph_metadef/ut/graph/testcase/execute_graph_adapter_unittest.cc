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
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"
#include "graph/utils/execute_graph_adapter.h"
#include "share_graph.h"
#include "mmpa/mmpa_api.h"

namespace ge {

using ExecuteSharedGraph = SharedGraph<ExecuteGraphPtr, ut::ExecuteGraphBuilder>;

class UtestExecuteGraphAdapter : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(UtestExecuteGraphAdapter, ConvertExecuteGraphToComputeGraph_Ok_without_subgraph) {
  auto exe_graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  auto compute_graph = ExecuteGraphAdapter::ConvertExecuteGraphToComputeGraph(exe_graph.get());
  EXPECT_NE(compute_graph, nullptr);
  EXPECT_EQ(exe_graph->GetAllNodes().size(), compute_graph->GetAllNodes().size());
}

TEST_F(UtestExecuteGraphAdapter, ConvertExecuteGraphToComputeGraph_Ok_with_subgraph) {
  auto exe_graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  auto attr_name = "use_execute_graph";
  auto attr_value = "1";
  AttrUtils::SetStr(exe_graph, attr_name, attr_value);
  auto ext_attr_name = "FakeExtAttr";
  auto ext_attr_value = "233abc";
  exe_graph->SetExtAttr(ext_attr_name, ext_attr_value);
  auto case0 = ExecuteGraphUtils::FindNodeFromAllNodes(exe_graph.get(), "case0");
  case0->GetOpDescBarePtr()->AddSubgraphName("branch3");
  case0->GetOpDescBarePtr()->SetSubgraphInstanceName(2, "");

  auto compute_graph = ExecuteGraphAdapter::ConvertExecuteGraphToComputeGraph(exe_graph.get());
  EXPECT_NE(compute_graph, nullptr);
  EXPECT_EQ(exe_graph->GetAllNodes().size(), compute_graph->GetAllNodes().size());
  auto value = AttrUtils::GetStr(compute_graph, attr_name);
  EXPECT_EQ(*value, attr_value);
  auto ext_value = compute_graph->TryGetExtAttr(ext_attr_name, "");
  EXPECT_EQ(ext_value, ext_attr_value);
}

TEST_F(UtestExecuteGraphAdapter, ConvertExecuteGraphToComputeGraph_Ok_with_null_edge) {
  auto exe_graph = ExecuteSharedGraph::BuildGraphWithControlEdge();
  auto n4 = ExecuteGraphUtils::FindNodeFromAllNodes(exe_graph.get(), "n4");
  auto out_edges = n4->GetOutEdgesByIndex(0);
  for (auto edge : out_edges) {
    exe_graph->RemoveEdge(edge);
  }
  auto out_ctrl_edge = n4->GetOutEdgesByIndex(-1);
  for (auto edge : out_ctrl_edge) {
    exe_graph->RemoveEdge(edge);
  }
  auto compute_graph = ExecuteGraphAdapter::ConvertExecuteGraphToComputeGraph(exe_graph.get());
  EXPECT_NE(compute_graph, nullptr);
  EXPECT_EQ(exe_graph->GetAllNodes().size(), compute_graph->GetAllNodes().size());
  auto n4_node = compute_graph->FindNode("n4");
  EXPECT_NE(n4_node, nullptr);
  EXPECT_EQ(n4_node->GetOutDataNodes().size(), 0);
}
}  // namespace ge
