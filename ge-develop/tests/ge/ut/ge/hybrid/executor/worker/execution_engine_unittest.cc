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
#include "runtime/rt.h"

#include "macro_utils/dt_public_scope.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/executor/worker/execution_engine.h"
#include "hybrid/executor/subgraph_executor.h"
#include "common/profiling/profiling_properties.h"
#include "common/dump/dump_manager.h"
#include "depends/profiler/src/dump_stub.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace hybrid;


class UtestExecutionEngine : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
  }
};
namespace {
const int kIntBase = 10;
}
namespace {
struct FakeGraphItem : GraphItem {
  FakeGraphItem(NodePtr node, UnknowShapeOpType op_type = DEPEND_COMPUTE) {
    NodeItem::Create(node, node_item);
    node_item->input_start = 0;
    node_item->output_start = 0;
    node_item->is_dynamic = true;
    node_item->shape_inference_type = op_type;
    node_item->is_output_shape_static = false;
    node_item->is_profiling_report = true;
    node_item->to_const_output_id_list.insert(0);
    node_items_.emplace_back(node_item.get());
    total_inputs_ = node->GetAllInAnchors().size();
    total_outputs_ = node->GetAllOutAnchors().size();
  }

  NodeItem *GetNodeItem() {
    return node_item.get();
  }

 private:
  std::unique_ptr<NodeItem> node_item;
};
}  // namespace

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({});
  op_desc->SetOutputOffset({});

  ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  bool support_dynamic = true;
  ge::AttrUtils::GetBool(op_desc, "support_dynamicshape", support_dynamic);
  return op_desc;
}

TEST_F(UtestExecutionEngine, ExecuteAsync_without_kernel_task) {
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  GraphExecutionContext execution_context;
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());
  execution_context.model = &hybrid_model;
  execution_context.profiling_level = 1;
  FakeGraphItem graph_item(node);
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(graph_item.GetNodeItem());
  ASSERT_TRUE(node_state->GetTaskContext() != nullptr);

  std::function<void()> callback;
  SubgraphExecutor executor(hybrid_model.GetRootGraphItem(), &execution_context);
  executor.InitCallback(node_state, callback);
  ExecutionEngine execution_engine;
  ProfilingProperties::Instance().is_load_profiling_ = true;
  EXPECT_EQ(execution_engine.ExecuteAsync(*node_state, node_state->GetTaskContext(), execution_context, callback), INTERNAL_ERROR);
}

TEST_F(UtestExecutionEngine, ExecuteAsync_without_callback_and_kernel_task) {
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  GraphExecutionContext execution_context;
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());
  execution_context.model = &hybrid_model;
  FakeGraphItem graph_item(node, DEPEND_SHAPE_RANGE);
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  ProfilingProperties::Instance().is_load_profiling_ = true;
  auto node_state = subgraph_context.GetNodeState(graph_item.GetNodeItem());
  std::string task_type = "rts";
  uint32_t block_dim = 0;
  node_state->GetTaskContext()->SaveProfilingTaskDescInfo(task_type, block_dim, op_desc->GetType());

  ASSERT_TRUE(node_state->GetTaskContext() != nullptr);

  std::function<void()> callback;
  SubgraphExecutor executor(hybrid_model.GetRootGraphItem(), &execution_context);
  executor.InitCallback(node_state, callback);
  ExecutionEngine execution_engine;
  std::map<std::string, std::string> run_options = {{"ge.exec.enable_exception_dump", "1"}};
  DumpManager::GetInstance().Init(run_options);

  const char_t *const kEnvRecordPath = "CONSTANT_FOLDING_PASS_9";
  char_t record_path[MMPA_MAX_PATH] = "mock_fail";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);

  EXPECT_EQ(execution_engine.ExecuteAsync(*node_state, node_state->GetTaskContext(), execution_context, callback),
            INTERNAL_ERROR);

  unsetenv(kEnvRecordPath);

  NodeDoneCallback node_done_callback(&execution_context, node_state->GetTaskContext());
  node_done_callback.context_->execution_context_->dump_properties.model_dump_properties_map_.clear();
  node_done_callback.context_->execution_context_->dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  EXPECT_EQ(node_done_callback.DumpDynamicNode(), SUCCESS);
  EXPECT_EQ(node_done_callback.PrepareConstInputs(*graph_item.GetNodeItem()), INTERNAL_ERROR);
  std::vector<TaskDescInfo> task_desc_info;
  EXPECT_EQ(node_done_callback.GetTaskDescInfo(*node_state->GetTaskContext(), node, task_desc_info), SUCCESS);


  ShapeInferenceEngine shape_infer_engine(&execution_context, true);
  shape_infer_engine.Config(&subgraph_context);
  EXPECT_EQ(shape_infer_engine.DoInferShape(*node_state), GRAPH_FAILED);
  EXPECT_EQ(shape_infer_engine.PropagateOutputShapes(*node_state), SUCCESS);
  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto tensor_value = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  GeShape ge_shape({16, 16});
  GeTensorDesc desc(ge_shape);
  EXPECT_EQ(shape_infer_engine.UpdateShapeAndValue(graph_item.GetNodeItem(), &tensor_value, &desc), SUCCESS);
  DumpManager::GetInstance().Finalize();
}

TEST_F(UtestExecutionEngine, NodeDoneCallback_SaveDumpOpInfo) {
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  GraphExecutionContext execution_context;
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());
  execution_context.model = &hybrid_model;
  FakeGraphItem graph_item(node, DEPEND_SHAPE_RANGE);
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  ProfilingProperties::Instance().is_load_profiling_ = true;
  auto node_state = subgraph_context.GetNodeState(graph_item.GetNodeItem());
  std::string task_type = "rts";
  uint32_t block_dim = 0;
  node_state->GetTaskContext()->SaveProfilingTaskDescInfo(task_type, block_dim, op_desc->GetType());

  ASSERT_TRUE(node_state->GetTaskContext() != nullptr);

  std::function<void()> callback;
  SubgraphExecutor executor(hybrid_model.GetRootGraphItem(), &execution_context);
  executor.InitCallback(node_state, callback);
  ExecutionEngine execution_engine;
  std::map<std::string, std::string> run_options = {{"ge.exec.enable_exception_dump", "1"}};
  DumpManager::GetInstance().Init(run_options);

  const char_t *const kEnvRecordPath = "CONSTANT_FOLDING_PASS_9";
  char_t record_path[MMPA_MAX_PATH] = "mock_fail";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);

  EXPECT_EQ(execution_engine.ExecuteAsync(*node_state, node_state->GetTaskContext(), execution_context, callback),
            INTERNAL_ERROR);

  unsetenv(kEnvRecordPath);

  ge::DumpStub::GetInstance().ClearOpInfos();
  NodeDoneCallback node_done_callback(&execution_context, node_state->GetTaskContext());
  node_done_callback.context_->execution_context_->dump_properties.model_dump_properties_map_.clear();
  node_done_callback.context_->execution_context_->dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  EXPECT_EQ(node_done_callback.SaveDumpOpInfo(), SUCCESS);

  ASSERT_NE(ge::DumpStub::GetInstance().GetOpInfos().size(), 0U);
  const auto &op_info = ge::DumpStub::GetInstance().GetOpInfos()[0];
  EXPECT_EQ(op_info.opName, "Add");
  ge::DumpStub::GetInstance().ClearOpInfos();
  DumpManager::GetInstance().Finalize();
}