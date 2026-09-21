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
#include <gmock/gmock.h>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "hybrid/executor/node_state.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/model/graph_item.h"
#include "graph/utils/graph_utils.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "common/profiling/profiling_manager.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestNodeState : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestNodeState, node_state_construction) {
  hybrid::GraphItem *graph_item = nullptr;
  hybrid::GraphExecutionContext *exec_ctx = nullptr;
  hybrid::SubgraphContext subgraph_context(graph_item, exec_ctx);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "Exit");
  NodePtr node = std::make_shared<Node>(op_desc, graph);
  hybrid::NodeItem node_item = hybrid::NodeItem(node);
  node_item.input_start = 0;
  node_item.output_start = 0;
  hybrid::FrameState frame_state = hybrid::FrameState();
  // mock switch
  ProfilingManager::Instance().device_id_.push_back(0);
  EXPECT_NO_THROW(ProfilingProperties::Instance().SetOpDetailProfiling(true));
  hybrid::NodeState node_state(node_item, &subgraph_context, frame_state);
  EXPECT_NO_THROW(ProfilingProperties::Instance().SetOpDetailProfiling(false));
}

//
//TEST_F(UtestNodeState, merge_await_shapes_ready) {
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//
//  const auto data0 = CreateNode(*graph, "data", DATA, 1, 1);
//  const auto data1 = CreateNode(*graph, "data1", DATA, 1, 1);
//  const auto merge1 = CreateNode(*graph, "merge", STREAMMERGE, 2, 2);
//  const auto output1 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);
//
//  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
//  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), merge1->GetInDataAnchor(1));
//  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));
//
//  GraphItem graph_item;
//  GraphExecutionContext graph_context;
//  SubgraphContext subgraph_context(&graph_item, &graph_context);
//
//  std::unique_ptr<NodeItem> node_item;
//  NodeItem::Create(merge1, node_item);
//  NodeState node_state(*node_item, &subgraph_context);
//
//  // Not dynamic.
//  ASSERT_EQ(node_state.AwaitShapesReady(graph_context), SUCCESS);
//
//  // Not set merge index.
//  node_item->is_dynamic = true;
//  ASSERT_EQ(node_state.AwaitShapesReady(graph_context), SUCCESS);
//
//}

} // namespace ge
