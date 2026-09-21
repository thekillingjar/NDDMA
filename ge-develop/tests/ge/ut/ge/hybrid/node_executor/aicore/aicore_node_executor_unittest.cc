/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "framework/common/taskdown_common.h"
#include "hybrid/executor/rt_callback_manager.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "api/gelib/gelib.h"
#include "graph/bin_cache/node_bin_selector_factory.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/graph.h"
#include "graph/operator_reg.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;
namespace {
  // for fuzz compiler to init
class FakeOpsKernelInfoStore : public OpsKernelInfoStore {
 private:
  Status Initialize(const std::map<std::string, std::string> &options) override {
    return SUCCESS;
  };
  Status Finalize() override {
    return SUCCESS;
  };
  bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override {
    return false;
  };
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override{};

  Status FuzzCompileOp(std::vector<NodePtr> &node_vec) override {
    return SUCCESS;
  };
};

void BuildDefaultTaskDef(domi::TaskDef & task_def) {
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  std::vector<uint8_t> args(100, 0);
  task_def.mutable_kernel()->set_args(args.data(), args.size());
  task_def.mutable_kernel()->set_args_size(100);
  task_def.mutable_kernel()->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  uint16_t args_offset = 20;
  task_def.mutable_kernel()->mutable_context()->set_args_offset(&args_offset, sizeof(args_offset));
}

class FakeOpsKernelBuilder : public OpsKernelBuilder {
 public:
  FakeOpsKernelBuilder() = default;

 private:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  };
  Status Finalize() override {
    return SUCCESS;
  };
  Status CalcOpRunningParam(Node &node) override {
    return SUCCESS;
  };
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
    domi::TaskDef task_def;
    BuildDefaultTaskDef(task_def);
    tasks.push_back(task_def);
    return SUCCESS;
  };
};

void BuildDefaultNodeItem(const NodePtr &node, std::unique_ptr<NodeItem> &node_item) {
  ASSERT_EQ(NodeItem::Create(node, node_item), SUCCESS);
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;
  ASSERT_EQ(node_item->Init(), SUCCESS);
}

class FuzzFailToCompileFusedNodeSelector : public NodeBinSelector {
  public:
  FuzzFailToCompileFusedNodeSelector() {};
  NodeCompileCacheItem *SelectBin(const NodePtr &node, const GEThreadLocalContext *ge_context,
                                  std::vector<domi::TaskDef> &task_defs) override {
    return nullptr;
  };
  Status Initialize() override {
    return SUCCESS;
  };
  private:
  NodeCompileCacheItem cci_;
};
} // namespace

class UtestAiCoreNodeExecutor : public testing::Test {
 protected:
  void SetUp() {
    ResetNodeIndex();
  }
  void TearDown() {}
};

TEST_F(UtestAiCoreNodeExecutor, callback_success) {
  std::unique_ptr<AiCoreOpTask> task1(new AiCoreOpTask());
  OpDescPtr op_desc = CreateOpDesc("Add", "Add", 2, 1, false);
  ge::AttrUtils::SetInt(*op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  domi::TaskDef task_def;
  BuildDefaultTaskDef(task_def);
  auto graph = make_shared<ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  EXPECT_EQ(task1->Init(node, task_def), ge::SUCCESS);

  std::unique_ptr<NodeItem> new_node;
  BuildDefaultNodeItem(node, new_node);

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(new_node.get());
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);


  GraphExecutionContext graph_context;
  graph_context.model = &hybrid_model;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(new_node.get());
  ASSERT_NE(node_state, nullptr);
  auto outputs_shape = reinterpret_cast<uint32_t(*)[9]>(task1->shape_buffer_->GetData());
  outputs_shape[0][0] = 2;
  outputs_shape[0][1] = 1;
  outputs_shape[0][2] = 2;
  std::vector<std::unique_ptr<AiCoreOpTask>> tasks;
  tasks.emplace_back(std::move(task1));
  std::unique_ptr<AiCoreNodeTask> aicore_node_task;
  aicore_node_task.reset(new (std::nothrow) AiCoreNodeTask(std::move(tasks)));
  node_state->GetTaskContext()->execution_context_->dump_properties.is_train_op_debug_ = true;

  ASSERT_EQ(aicore_node_task->ExecuteAsync(*node_state->GetTaskContext(), nullptr), SUCCESS);

  const char_t *const kEnvRecordPath = "CONSTANT_FOLDING_PASS_9";
  char_t record_path[MMPA_MAX_PATH] = "mock_fail";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);

  ASSERT_NE(aicore_node_task->ExecuteAsync(*node_state->GetTaskContext(), nullptr), SUCCESS);

  unsetenv(kEnvRecordPath);

  AiCoreNodeExecutor executor;
  std::shared_ptr<NodeTask> cur_task;
  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def});
  hybrid_model.node_items_[node] = std::move(new_node);
  hybrid_model.SetNodeBinMode(fuzz_compile::kOneNodeSingleBinMode);
  ASSERT_EQ(executor.LoadTask(hybrid_model, node, cur_task), SUCCESS);
  executor.PrepareTask(*cur_task, *node_state->GetTaskContext());

  ASSERT_EQ(aicore_node_task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);
  ASSERT_EQ(aicore_node_task->UpdateTilingData(*node_state->GetTaskContext()), SUCCESS);

  std::unique_ptr<NodeTask> aicpu_task;

  aicore_node_task->aicpu_task_ = std::move(aicpu_task);
  ASSERT_EQ(aicore_node_task->aicpu_task_, nullptr);

  // test select bin
  auto selector = NodeBinSelectorFactory::GetInstance().GetNodeBinSelector(fuzz_compile::kOneNodeSingleBinMode);
  ASSERT_NE(selector, nullptr);
  auto ret = selector->Initialize();
  ASSERT_EQ(ret, SUCCESS);
  (void)aicore_node_task->SetBinSelector(selector);
  ret = aicore_node_task->SelectBin(*node_state->GetTaskContext(), subgraph_context.GetExecutionContext());
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(aicore_node_task->last_bin_, UINT64_MAX); // single mode not upate op task

  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> entry;
  auto callback_manager =
      dynamic_cast<RtCallbackManager *>(node_state->GetTaskContext()->execution_context_->callback_manager);
  callback_manager->callback_queue_.Pop(entry);
  auto cb_func = entry.second.first;
  auto cb_args = entry.second.second;
  cb_func(cb_args);
  (void)rtEventDestroy((rtEvent_t)entry.first);
}

// Used by test_Load_Task_without_task_def
REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2D)

TEST_F(UtestAiCoreNodeExecutor, test_Load_Task_without_task_def) {
  map<std::string, std::string> options;
  options.emplace(BUILD_MODE, BUILD_MODE_TUNING);
  options.emplace(BUILD_STEP, BUILD_STEP_AFTER_BUILDER);
  Status ret = ge::GELib::Initialize(options);
  ASSERT_EQ(ret, SUCCESS);
  OpsKernelInfoStorePtr kernel_ptr = std::make_shared<FakeOpsKernelInfoStore>();
  OpsKernelManager::GetInstance().ops_kernel_store_["AIcoreEngine"] = kernel_ptr;
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_["AIcoreEngine"] = std::make_shared<FakeOpsKernelBuilder>();

  // 1. build hybrid model and node
  OpDescPtr op_desc = CreateOpDesc("conv2d", CONV2D, 1, 1, false);
  auto graph = make_shared<ComputeGraph>("graph");
  auto conv2d_node = graph->AddNode(op_desc);
  conv2d_node->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
  // set compile info on node
  ge::AttrUtils::SetStr(conv2d_node->GetOpDesc(), "compile_info_key", "op_compile_info_key");
  ge::AttrUtils::SetStr(conv2d_node->GetOpDesc(), "compile_info_json", "op_compile_info_json");

  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_unknown_rank");

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.node_bin_mode_ = fuzz_compile::kOneNodeMultipleBinsMode;
  hybrid_model.task_defs_[conv2d_node] = {}; //empty task def
  std::unique_ptr<NodeItem> node_item;
  BuildDefaultNodeItem(conv2d_node, node_item);

  // 2. build task_context with node_state
  GraphExecutionContext graph_context;
  graph_context.model = &hybrid_model;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item.get());
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  auto node_state = subgraph_context.GetNodeState(node_item.get());
  ASSERT_NE(node_state, nullptr);
  // build subgraph_item for fused graph, and set to hybrid model
  hybrid_model.node_items_[conv2d_node] = std::move(node_item);

  // 3. load empty task when fuzz compile
  std::shared_ptr<NodeTask> node_task_after_load;
  AiCoreNodeExecutor executor;
  ASSERT_EQ(executor.LoadTask(hybrid_model, conv2d_node, node_task_after_load), SUCCESS);

  // 4. test select bin
  ASSERT_EQ(node_task_after_load->SelectBin(*node_state->GetTaskContext(), subgraph_context.GetExecutionContext()), SUCCESS);

  // 5. load empty task when norma case, load failed
  hybrid_model.node_bin_mode_ = fuzz_compile::kOneNodeSingleBinMode;
  ASSERT_NE(executor.LoadTask(hybrid_model, conv2d_node, node_task_after_load), SUCCESS);
}
// test fused node (conv2d)
// before fused, its origin graph is : (conv2d->sqrt)
// 1.加载aicore task时，加载fused task
// 2.selectbin时，使能fused task
// 3.execute async时，执行fused task
TEST_F(UtestAiCoreNodeExecutor, test_execute_on_origin_fused_task) {
  // 1. test load task with fused_task
  // (1) build origin fused graph
  DEF_GRAPH(fused_graph) {
      auto data = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
      CHAIN(NODE("data", data)->NODE("conv2d", EXPANDDIMS)->NODE("sqrt", RESHAPE)->NODE("netoutput", NETOUTPUT));
  };
  auto origin_fused_graph = ToComputeGraph(fused_graph);
  auto net_output = origin_fused_graph->FindNode("netoutput");
  AttrUtils::SetInt(net_output->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  // (2) build hybrid model and node
  OpDescPtr op_desc = CreateOpDesc("fused_conv2d", CONV2D, 1, 1, false);
  AttrUtils::SetGraph(op_desc, "_original_fusion_graph", origin_fused_graph);
  auto graph = make_shared<ComputeGraph>("graph");
  auto fused_conv2d_node = graph->AddNode(op_desc);

  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_fusion");

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.node_bin_mode_ = fuzz_compile::kOneNodeMultipleBinsMode;
  domi::TaskDef task_def;
  BuildDefaultTaskDef(task_def);
  hybrid_model.task_defs_[fused_conv2d_node] = std::vector<domi::TaskDef>({task_def});
  std::unique_ptr<NodeItem> node_item;
  BuildDefaultNodeItem(fused_conv2d_node, node_item);

  // (3) build task_context with node_state
  GraphExecutionContext graph_context;
  graph_context.model = &hybrid_model;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item.get());
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  auto node_state = subgraph_context.GetNodeState(node_item.get());
  ASSERT_NE(node_state, nullptr);
  // build subgraph_item for fused graph, and set to hybrid model
  hybrid_model.node_items_[fused_conv2d_node] = std::move(node_item);
  std::unique_ptr<GraphItem> origin_fused_graph_item = std::unique_ptr<GraphItem>(new GraphItem());
  NodeExecutorManager::GetInstance().EnsureInitialized();
  for (const auto &node : origin_fused_graph->GetDirectNode()) {
      if (node->GetType() == DATA) {
        ++origin_fused_graph_item->total_inputs_;
      }
      if (node->GetType() == NETOUTPUT) {
        ++origin_fused_graph_item->total_outputs_;
      }
      std::unique_ptr<NodeItem> node_item_in_sub;
      BuildDefaultNodeItem(node, node_item_in_sub);
      // simulate load task on node in subgraph
      if (node->GetType() != DATA && node->GetType() != NETOUTPUT) {
         node->GetOpDesc()->SetOpKernelLibName("GeLocal");
      }

      // NodeExecutorManager::GetInstance().GetExecutor(*node_item_in_sub, node_item_in_sub->node_executor);
      // const auto &node_ptr = node_item_in_sub->node;
      // node_item_in_sub->node_executor->LoadTask(hybrid_model, node_ptr, node_item_in_sub->kernel_task);
      origin_fused_graph_item->node_items_.emplace_back(node_item_in_sub.get());
      hybrid_model.node_items_[node] = std::move(node_item_in_sub);
  }
  hybrid_model.subgraph_items_[origin_fused_graph->GetName()] = std::move(origin_fused_graph_item);

  // (4) load task of fused_node
  std::shared_ptr<NodeTask> node_task_after_load;
  AiCoreNodeExecutor executor;
  ASSERT_EQ(executor.LoadTask(hybrid_model, fused_conv2d_node, node_task_after_load), SUCCESS);
  AiCoreNodeTask *aicore_node_task = dynamic_cast<AiCoreNodeTask *>(node_task_after_load.get());
  ASSERT_NE(aicore_node_task->fused_task_, nullptr); // after load task, fused_task_ has been load

  // 2. test select bin switch to origin fused graph execution
  // (1) set failed_compile_selector to aicore_node_task
  std::shared_ptr<FuzzFailToCompileFusedNodeSelector> sp = make_shared<FuzzFailToCompileFusedNodeSelector>();
  aicore_node_task->SetBinSelector(sp.get());

  // (2) test select bin to switch to orgin_fused_graph_execution
  ASSERT_EQ(aicore_node_task->SelectBin(*node_state->GetTaskContext(), subgraph_context.GetExecutionContext()), SUCCESS);
  ASSERT_EQ(aicore_node_task->origin_fused_graph_exec_, true);
}

TEST_F(UtestAiCoreNodeExecutor, check_overflow_test) {
  std::unique_ptr<AiCoreOpTask> task1(new AiCoreOpTask());
  OpDescPtr op_desc = CreateOpDesc("Add", "Add", 2, 1, false);
  ge::AttrUtils::SetInt(*op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  domi::TaskDef task_def;
  BuildDefaultTaskDef(task_def);
  auto graph = make_shared<ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  EXPECT_EQ(task1->Init(node, task_def), ge::SUCCESS);

  std::unique_ptr<NodeItem> new_node;
  BuildDefaultNodeItem(node, new_node);

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(new_node.get());
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);

  GraphExecutionContext graph_context;
  graph_context.model = &hybrid_model;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(new_node.get());
  ASSERT_NE(node_state, nullptr);
  auto outputs_shape = reinterpret_cast<uint32_t(*)[9]>(task1->shape_buffer_->GetData());
  outputs_shape[0][0] = 2;
  outputs_shape[0][1] = 1;
  outputs_shape[0][2] = 2;
  std::vector<std::unique_ptr<AiCoreOpTask>> tasks;
  tasks.emplace_back(std::move(task1));
  std::unique_ptr<AiCoreNodeTask> aicore_node_task;
  aicore_node_task.reset(new (std::nothrow) AiCoreNodeTask(std::move(tasks)));

  auto task_context = *node_state->GetTaskContext();
  task_context.execution_context_->dump_properties.is_train_op_debug_ = true;
  const char_t *const kEnvOverFlowPath = "SYNCSTREAM_OVERFLOW_RET";
  char_t over_flow_path[MMPA_MAX_PATH] = "aicore";
  mmSetEnv(kEnvOverFlowPath, &over_flow_path[0U], MMPA_MAX_PATH);
  ASSERT_EQ(aicore_node_task->CheckOverflow(task_context), SUCCESS);
  unsetenv(kEnvOverFlowPath);
}
}  // namespace ge
