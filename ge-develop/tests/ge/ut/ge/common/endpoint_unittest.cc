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
#include "compute_graph.h"
#include "dflow/base/model/model_relation.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "framework/common/types.h"
#include "utils/graph_utils.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/debug/ge_log.h"

#include "macro_utils/dt_public_scope.h"
#include "dflow/base/model/endpoint.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class EndpointTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

 protected:
  static ComputeGraphPtr BuildGraph() {
    NodePtr partitioned_call;
    ComputeGraphPtr root_graph;
    auto builder = ut::GraphBuilder("root_graph");
    auto data0 = builder.AddNode("data0", DATA, 0, 1);
    auto data1 = builder.AddNode("data1", DATA, 0, 1);
    AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
    partitioned_call = builder.AddNode("partitioned_call", PARTITIONEDCALL, 2, 1);
    auto net_output = builder.AddNode("root_netoutput", NETOUTPUT, 1, 1);

    builder.AddDataEdge(data0, 0, partitioned_call, 0);
    builder.AddDataEdge(data1, 0, partitioned_call, 1);
    builder.AddDataEdge(partitioned_call, 0, net_output, 0);
    root_graph = builder.GetGraph();
    {  // build subgraph
      auto builder_subgraph = ut::GraphBuilder("subgraph_1");
      auto data00 = builder_subgraph.AddNode("data00", DATA, 1, 1);
      auto data01 = builder_subgraph.AddNode("data01", DATA, 1, 1);
      auto conv_node = builder_subgraph.AddNode("conv_node", CONV2D, 2, 1);
      auto net_output_subgraph = builder_subgraph.AddNode("subgraph1_netoutput", NETOUTPUT, 1, 1);

      AttrUtils::SetInt(data00->GetOpDesc(), ATTR_NAME_INDEX, 0);
      AttrUtils::SetInt(data01->GetOpDesc(), ATTR_NAME_INDEX, 1);

      AttrUtils::SetInt(data00->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
      AttrUtils::SetInt(data01->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

      builder_subgraph.AddDataEdge(data00, 0, conv_node, 0);
      builder_subgraph.AddDataEdge(data01, 0, conv_node, 1);

      builder_subgraph.AddDataEdge(conv_node, 0, net_output_subgraph, 0);
      auto subgraph = builder_subgraph.GetGraph();

      subgraph->SetParentNode(partitioned_call);
      subgraph->SetParentGraph(root_graph);
      partitioned_call->GetOpDesc()->AddSubgraphName("subgraph_1");
      partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph_1");
      root_graph->AddSubgraph(subgraph);
    }
    root_graph->TopologicalSorting();
    return root_graph;
  }
};

TEST_F(EndpointTest, SetGetAttr_QueueNode_Success) {
  Endpoint queue_def("queue_def_name", EndpointType::kQueue);
  auto queue_node_utils = QueueNodeUtils(queue_def).SetEnqueuePolicy("FIFO").SetNodeAction(kQueueActionControl);

  EXPECT_EQ(queue_node_utils.GetDepth(), 128L);
  EXPECT_EQ(queue_node_utils.GetEnqueuePolicy(), "FIFO");
  EXPECT_EQ(queue_node_utils.GetIsControl(), true);

  queue_node_utils.SetDepth(3L);
  EXPECT_EQ(queue_node_utils.GetDepth(), 3L);
}
}  // namespace ge
