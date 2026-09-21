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
#include "hybrid/node_executor/host_cpu/host_cpu_node_executor.h"
#include "common/model/ge_root_model.h"
#include "graph/passes/graph_builder_utils.h"
#include "aicpu_task_struct.h"
#include "graph/manager/mem_manager.h"
#include "host_cpu_engine/host_cpu_engine.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

namespace {
struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[2];
}__attribute__((packed));
}  // namespace

class UtestHostCpuNodeTask : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
struct FakeGraphItem : GraphItem {
  FakeGraphItem(NodePtr node) {
    NodeItem::Create(node, node_item);
    node_item->input_start = 0;
    node_item->output_start = 0;
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

TEST_F(UtestHostCpuNodeTask, test_load) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();

  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  std::unique_ptr<NodeItem> node_item;
  ASSERT_EQ(NodeItem::Create(node, node_item), SUCCESS);
  hybrid_model.node_items_[node] = std::move(node_item);
  hybrid_model.task_defs_[node] = {};

  NodeTaskPtr task = nullptr;
  HostCpuNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), PARAM_INVALID);

  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 2;

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task_def.mutable_kernel()->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  task_def.mutable_kernel()->set_args_size(args.head.length);
  hybrid_model.task_defs_[node] = {task_def};
  hybrid_model.node_items_[node]->num_inputs = 1;
  hybrid_model.node_items_[node]->num_outputs = 1;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), INTERNAL_ERROR);

  domi::TaskDef &host_task_def = hybrid_model.task_defs_[node][0];
  host_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), INTERNAL_ERROR);
  domi::KernelContext *context = host_task_def.mutable_kernel()->mutable_context();
  context->set_kernel_type(8);    // ccKernelType::HOST_CPU
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), INTERNAL_ERROR);
  HostCpuEngine::GetInstance().constant_folding_handle_ = (void *)0x01;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), INTERNAL_ERROR);
}

TEST_F(UtestHostCpuNodeTask, test_execute) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  domi::TaskDef task_def;

  FakeGraphItem graphItem(node);
  auto node_item = graphItem.GetNodeItem();
  SubgraphContext subgraph_context(&graphItem, nullptr);
  HostAicpuNodeTask task(node_item, task_def);
  std::function<void()> call_back = []{};
  FrameState frame_state;
  NodeState node_state(*node_item, &subgraph_context, frame_state);
  TaskContext context(nullptr, &node_state, &subgraph_context);
  ASSERT_EQ(task.ExecuteAsync(context, call_back), INTERNAL_ERROR);

  std::function<uint32_t (void *)> run_cpu_kernel = [](void *){ return 0; };
  task.SetRunKernel(run_cpu_kernel);
  ASSERT_EQ(task.ExecuteAsync(context, call_back), SUCCESS);
}

TEST_F(UtestHostCpuNodeTask, test_update_args) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  FrameState frame_state;
  FakeGraphItem graphItem(node);
  auto node_item = graphItem.GetNodeItem();
  SubgraphContext subgraph_context(nullptr, nullptr);
  NodeState node_state(*node_item, &subgraph_context, frame_state);
  TaskContext context(nullptr, &node_state, &subgraph_context);

  auto *in_addr = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Malloc(204800);
  auto tmp = TensorBuffer::Create(in_addr, 204800);
  std::shared_ptr<TensorBuffer> input_buffer(tmp.release());
  TensorValue input_start[1] = {TensorValue(input_buffer)};
  context.inputs_start_ = input_start;

  auto *out_addr = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Malloc(204800);
  tmp = TensorBuffer::Create(out_addr, 204800);
  std::shared_ptr<TensorBuffer> output_buffer(tmp.release());
  TensorValue output_start[1] = {TensorValue(output_buffer)};
  context.outputs_start_ = output_start;

  domi::TaskDef task_def;
  HostAicpuNodeTask task(node_item, task_def);
  ASSERT_EQ(task.UpdateArgs(context), INTERNAL_ERROR);

  task.args_size_ = sizeof(AicpuTaskStruct);
  task.args_.reset(new(std::nothrow) uint8_t[task.args_size_]());
  task.args_ex_.args = task.args_.get();
  task.args_ex_.argsSize = task.args_size_;
  ASSERT_EQ(task.UpdateArgs(context), SUCCESS);
}
} // namespace ge
