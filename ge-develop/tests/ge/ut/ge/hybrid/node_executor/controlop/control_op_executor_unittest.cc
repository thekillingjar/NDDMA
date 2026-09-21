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
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/controlop/control_op_executor.h"
#include "common/model/ge_root_model.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "hybrid/executor/resource_manager.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestControlOpExecutor : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() { }
};


TEST_F(UtestControlOpExecutor, test_IfOpNodeTask) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "if", IF, 2, 1);
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
  graph->SetParentGraph(root_graph);

  node->GetOpDesc()->AddSubgraphName("then_graph");
  node->GetOpDesc()->AddSubgraphName("else_graph");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "then_graph");
  node->GetOpDesc()->SetSubgraphInstanceName(1, "else_graph");

  ComputeGraphPtr then_graph = std::make_shared<ComputeGraph>("then_graph");
  NodeUtils::SetSubgraph(*node, 0, then_graph);

  ComputeGraphPtr else_graph = std::make_shared<ComputeGraph>("else_graph");
  NodeUtils::SetSubgraph(*node, 1, else_graph);

  hybrid_model.subgraph_items_["then_graph"] = std::unique_ptr<GraphItem>(new GraphItem());
  hybrid_model.subgraph_items_["else_graph"] = std::unique_ptr<GraphItem>(new GraphItem());

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;
  ASSERT_EQ(graph_context.callback_manager->Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 2; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  std::shared_ptr<NodeTask> task;
  ControlOpNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  ASSERT_EQ(node_executor.PrepareTask(*task, *node_state->GetTaskContext()), SUCCESS);

  ASSERT_EQ(task->Init(*node_state->GetTaskContext()), SUCCESS);

  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  bool ret_value;
  double value_float = 1;
  TensorValue in_tensor_float(&value_float, sizeof(value_float));
  ASSERT_EQ(dynamic_cast<ControlOpNodeTask *>(task.get())->ToBool(in_tensor_float, DT_FLOAT, ret_value), SUCCESS);

  bool value_bool = true;
  TensorValue in_tensor_bool(&value_bool, sizeof(value_bool));
  ASSERT_EQ(dynamic_cast<ControlOpNodeTask *>(task.get())->ToBool(in_tensor_bool, DT_BOOL, ret_value), SUCCESS);

  ASSERT_EQ(dynamic_cast<ControlOpNodeTask *>(task.get())->ToBool(in_tensor_bool, DT_STRING, ret_value), UNSUPPORTED);
  ASSERT_EQ(graph_context.callback_manager->Destroy(), SUCCESS);
  //ASSERT_EQ(task->ControlOpNodeTask::UpdateArgs(*node_state->GetTaskContext()), UNSUPPORTED);  // ??
}

TEST_F(UtestControlOpExecutor, test_CaseOpNodeTask) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "case", CASE, 2, 2);
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
  graph->SetParentGraph(root_graph);

  node->GetOpDesc()->AddSubgraphName("then_graph");
  node->GetOpDesc()->AddSubgraphName("else_graph");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "then_graph");
  node->GetOpDesc()->SetSubgraphInstanceName(1, "else_graph");

  ComputeGraphPtr then_graph = std::make_shared<ComputeGraph>("then_graph");
  NodeUtils::SetSubgraph(*node, 0, then_graph);

  ComputeGraphPtr else_graph = std::make_shared<ComputeGraph>("else_graph");
  NodeUtils::SetSubgraph(*node, 1, else_graph);

  hybrid_model.subgraph_items_["then_graph"] = std::unique_ptr<GraphItem>(new GraphItem());
  hybrid_model.subgraph_items_["else_graph"] = std::unique_ptr<GraphItem>(new GraphItem());

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->num_outputs = 1;
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;
  ASSERT_EQ(graph_context.callback_manager->Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 2; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  std::shared_ptr<NodeTask> task;
  ControlOpNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  ASSERT_EQ(node_executor.PrepareTask(*task, *node_state->GetTaskContext()), SUCCESS);

  ASSERT_EQ(task->Init(*node_state->GetTaskContext()), SUCCESS);

  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  ASSERT_EQ(graph_context.callback_manager->Destroy(), SUCCESS);
}


TEST_F(UtestControlOpExecutor, test_WhileOpNodeTask) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "while", WHILE, 1, 1);
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
  graph->SetParentGraph(root_graph);

  node->GetOpDesc()->AddSubgraphName("cond_graph");
  node->GetOpDesc()->AddSubgraphName("body_graph");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "cond_graph");
  node->GetOpDesc()->SetSubgraphInstanceName(1, "body_graph");

  ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>("cond_graph");
  NodeUtils::SetSubgraph(*node, 0, cond_graph);

  ComputeGraphPtr else_graph = std::make_shared<ComputeGraph>("body_graph");
  NodeUtils::SetSubgraph(*node, 1, else_graph);

  std::unique_ptr<NodeItem> new_cond_node;
  NodePtr cond_node = CreateNode(*graph, "data", DATA, 1, 1);
  ASSERT_EQ(NodeItem::Create(cond_node, new_cond_node), SUCCESS);
  NodeItem *cond_node_item = new_cond_node.get();
  // this attr is requrid, otherwise task_context on node state can not init successfully
  cond_node_item->input_start = 0; 
  cond_node_item->output_start = 0;
  cond_node_item->num_outputs = 1;
  std::unique_ptr<GraphItem> cond_graph_item = std::unique_ptr<GraphItem>(new GraphItem());
  cond_graph_item->node_items_.emplace_back(cond_node_item);
  cond_graph_item->total_inputs_ = 1;
  cond_graph_item->total_outputs_ = 1;
  cond_graph_item->output_node_ = cond_node_item;

  hybrid_model.subgraph_items_["cond_graph"] = std::move(cond_graph_item);

  hybrid_model.subgraph_items_["body_graph"] = std::unique_ptr<GraphItem>(new GraphItem());

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->num_outputs = 1;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;
  graph_item.output_node_ = node_item;

  //hybrid_model.subgraph_items_["cond_graph"] = std::unique_ptr<GraphItem><&graph_item>;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);

  void *test1 = new uint8_t[1];
  TensorBuffer *tensor_buffer1 = new TensorBuffer(nullptr, test1, 1);

  void *test2 = new uint8_t[1];
  TensorBuffer *tensor_buffer2 = new TensorBuffer(nullptr, test1, 1);

  subgraph_context.all_outputs_ = std::vector<TensorValue>({TensorValue(std::shared_ptr<TensorBuffer>(tensor_buffer1))});
  //subgraph_context.SetOutput(*node_item, 0, TensorValue(std::shared_ptr<TensorBuffer>(tensor_buffer2)));

  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 2; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  std::shared_ptr<NodeTask> task;
  ControlOpNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  ASSERT_EQ(node_executor.PrepareTask(*task, *node_state->GetTaskContext()), SUCCESS);

  ASSERT_EQ(task->Init(*node_state->GetTaskContext()), SUCCESS);

  std::function<void()> done = []() {};
  ASSERT_NE(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  bool is_continue = false;
  ASSERT_NE(dynamic_cast<WhileOpNodeTask *>(task.get())->ExecuteOneLoop(*node_state->GetTaskContext(), is_continue), SUCCESS);

  ASSERT_EQ(dynamic_cast<WhileOpNodeTask *>(task.get())->MoveOutputs2Inputs(*node_state->GetTaskContext()), SUCCESS);  // ??

  ASSERT_EQ(dynamic_cast<WhileOpNodeTask *>(task.get())->MoveOutputs2Inputs(*node_state->GetTaskContext()), SUCCESS);  // ??

  // input output not equal
  const_cast<NodeItem *>(node_state->GetTaskContext()->node_item_)->num_inputs = 10;
  ASSERT_NE(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  delete[] (uint8_t *)test1;
  delete[] (uint8_t *)test2;
  delete tensor_buffer2;
}


TEST_F(UtestControlOpExecutor, test_LoadTaskFail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "matmul", MATMUL, 2, 1);
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
  graph->SetParentGraph(root_graph);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

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

  std::shared_ptr<NodeTask> task;
  ControlOpNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), PARAM_INVALID);
}

}



