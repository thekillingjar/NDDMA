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
#include "framework/common/taskdown_common.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "api/gelib/gelib.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
namespace ge {
using namespace hybrid;
class TaskContextTest : public testing::Test {
 protected:
  void SetUp() {
    ResetNodeIndex();
  }
  void TearDown() {}
};

TEST_F(TaskContextTest, test_attribute_of_context) {
  OpDescPtr op_desc = CreateOpDesc("Add", "Add", 2, 1);
  ge::AttrUtils::SetInt(*op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);
  auto & task_context = *node_state->GetTaskContext();
  task_context.SetStatus(FAILED);
  ASSERT_EQ(task_context.status_, FAILED);
  task_context.stream_id_ = 10;
  ASSERT_EQ(task_context.GetStreamId(), 10);
  task_context.SetOverFlow(false);
  ASSERT_EQ(task_context.IsOverFlow(), false);
  ASSERT_EQ(task_context.IsForceInferShape(), task_context.force_infer_shape_);
  task_context.OnError(SUCCESS);
  ASSERT_EQ(task_context.execution_context_->status, SUCCESS);
  ASSERT_EQ(task_context.Synchronize(), SUCCESS);
  ASSERT_EQ(task_context.RegisterCallback(nullptr), SUCCESS);
  ASSERT_EQ(task_context.GetOutput(-1), nullptr);

  TensorValue tensor_value;
  ASSERT_EQ(task_context.SetOutput(-1, tensor_value), PARAM_INVALID);
  ASSERT_EQ(task_context.GetInput(-1), nullptr);
  ASSERT_EQ(task_context.MutableInput(-1), nullptr);
  ASSERT_EQ(task_context.MutableWorkspace(-1), nullptr);
  ASSERT_NE(task_context.node_item_->node, nullptr);
  ASSERT_NE(task_context.node_item_->node->GetOpDesc(), nullptr);
  task_context.node_item_->node->GetOpDesc()->SetWorkspaceBytes({0});
  task_context.execution_context_->allocator = NpuMemoryAllocator::GetAllocator();
  ASSERT_EQ(task_context.AllocateWorkspaces(), ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED);
}
}  // namespace ge
