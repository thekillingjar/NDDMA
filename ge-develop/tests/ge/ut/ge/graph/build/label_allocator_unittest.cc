/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "graph/build/label_allocator.h"
#include "macro_utils/dt_public_unscope.h"

#include "compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "utils/graph_utils.h"
#include "graph/passes/graph_builder_utils.h"

#include "framework/common/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/label/label_maker.h"

namespace ge {
class LabelAllocatorTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(LabelAllocatorTest, TestSkipPipelinePartition) {
  auto builder = ut::GraphBuilder("g1");
  auto partitioned_call_1 = builder.AddNode("PartitionedCall1", PARTITIONEDCALL, 0, 1);
  auto net_output = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

  auto op_desc = partitioned_call_1->GetOpDesc();
  auto subgraph = std::make_shared<ComputeGraph>("subgraph-1");
  op_desc->AddSubgraphName("subgraph-1");
  op_desc->SetSubgraphInstanceName(0, "subgraph-1");
  subgraph->SetParentNode(partitioned_call_1);

  builder.AddDataEdge(partitioned_call_1, 0, net_output, 0);
  LabelAllocator label_allocator(subgraph);
  std::set<NodePtr> functional_nodes;
  ASSERT_TRUE(label_allocator.CollectFunctionalNode(subgraph, functional_nodes));
  ASSERT_EQ(functional_nodes.size(), 1);
  ASSERT_EQ(*functional_nodes.begin(), partitioned_call_1);

  auto root_graph = builder.GetGraph();
  functional_nodes.clear();
  ASSERT_TRUE(label_allocator.CollectFunctionalNode(subgraph, functional_nodes));
  ASSERT_EQ(functional_nodes.size(), 1);
  ASSERT_EQ(*functional_nodes.begin(), partitioned_call_1);

  functional_nodes.clear();
  ASSERT_TRUE(label_allocator.CollectFunctionalNode(subgraph, functional_nodes));
  ASSERT_EQ(functional_nodes.size(), 1);
  ASSERT_EQ(*functional_nodes.begin(), partitioned_call_1);

  functional_nodes.clear();
  std::shared_ptr<ge::ComputeGraph> gp = nullptr;
  ASSERT_FALSE(label_allocator.CollectFunctionalNode(gp, functional_nodes));
  subgraph->SetGraphUnknownFlag(true);
  ASSERT_TRUE(label_allocator.CollectFunctionalNode(subgraph, functional_nodes));
  subgraph->SetGraphUnknownFlag(false);
  auto node = subgraph->GetParentNode();
  node->GetOpDesc()->SetAttr("_ffts_sub_graph", AnyValue::CreateFrom<std::string>("sub_graph"));
  node->GetOpDesc()->SetAttr("_ffts_plus_sub_graph", AnyValue::CreateFrom<std::string>("plus_sub_graph"));
  ASSERT_TRUE(label_allocator.CollectFunctionalNode(subgraph, functional_nodes));
  auto owner = node->GetOwnerComputeGraph();
  owner->SetGraphUnknownFlag(true);
  ASSERT_TRUE(label_allocator.CollectFunctionalNode(subgraph, functional_nodes));
}


TEST_F(LabelAllocatorTest, AssignFunctionalLabels) {
  auto builder = ut::GraphBuilder("g1");
  auto partitioned_call_1 = builder.AddNode("PartitionedCall1", PARTITIONEDCALL, 0, 1);
  auto net_output = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

  auto op_desc = partitioned_call_1->GetOpDesc();
  auto subgraph = std::make_shared<ComputeGraph>("subgraph-1");
  op_desc->AddSubgraphName("subgraph-1");
  op_desc->SetSubgraphInstanceName(0, "subgraph-1");
  subgraph->SetParentNode(partitioned_call_1);

  builder.AddDataEdge(partitioned_call_1, 0, net_output, 0);
  LabelAllocator label_allocator(subgraph);
  label_allocator.compute_graph_ = nullptr;
  EXPECT_EQ(label_allocator.AssignFunctionalLabels(), INTERNAL_ERROR);
}
}  // namespace ge
