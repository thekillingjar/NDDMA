/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/partitioned_call/partitioned_call_node_executor.h"
#include <memory>

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::DYNAMIC_SUBGRAPH, PartitionedCallNodeExecutor);

PartitionedCallNodeTask::PartitionedCallNodeTask(const GraphItem * const graph_item)
    : NodeTask(), graph_item_(graph_item) {
}

PartitionedCallNodeTask::~PartitionedCallNodeTask() {
  GELOGD("[%s] PartitionedCallNodeTask destroyed.", graph_item_->GetName().c_str());
}

Status PartitionedCallNodeTask::Init(TaskContext &context) {
  const auto execution_context = context.GetExecutionContext();
  subgraph_executor_.reset();
  subgraph_executor_ = MakeUnique<SubgraphExecutor>(graph_item_, execution_context);
  GE_CHECK_NOTNULL(subgraph_executor_);
  GE_CHK_STATUS_RET(subgraph_executor_->Init(), "[Init][SubgraphExecutor]Failed.");
  return SUCCESS;
}

Status PartitionedCallNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GE_CHK_STATUS_RET(subgraph_executor_->ExecuteAsync(context),
                    "[Invoke][ExecuteAsync] failed for[%s]", graph_item_->GetName().c_str());

  const auto callback = [this, done_callback]() {
    Callback(done_callback);
  };

  GE_CHK_STATUS_RET(context.RegisterCallback(callback),
                    "[Register][Callback] failed for [%s]", graph_item_->GetName().c_str());
  GELOGD("[%s] Done executing subgraph successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status PartitionedCallNodeTask::Callback(const std::function<void()> &done_callback) {
  GELOGD("[%s] On subgraph callback", graph_item_->GetName().c_str());
  if (done_callback != nullptr) {
    done_callback();
  }

  GELOGD("[%s] To release sub graph tensors.", graph_item_->GetName().c_str());
  subgraph_executor_.reset();
  GELOGD("[%s] Done releasing sub graph tensors.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status PartitionedCallNodeTask::UpdateArgs(TaskContext &context) {
  (void)context;
  return SUCCESS;
}

Status PartitionedCallNodeExecutor::LoadTask(const ge::hybrid::HybridModel &model,
                                             const ge::NodePtr &node,
                                             std::shared_ptr<NodeTask> &task) const {
  GELOGD("Load dynamic partitioned call: [%s]", node->GetName().c_str());
  const auto &subgraph = NodeUtils::GetSubgraph(*node, 0U);
  GE_CHECK_NOTNULL(subgraph);
  const auto partitioned_call = model.GetSubgraphItem(subgraph);
  GE_CHECK_NOTNULL(partitioned_call);
  GE_ASSERT(!node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH), "FftsPlus task is unsupported.");
  task = MakeShared<PartitionedCallNodeTask>(partitioned_call);
  GE_CHECK_NOTNULL(task);

  GELOGD("Done loading dynamic partitioned call: [%s]", node->GetName().c_str());
  return SUCCESS;
}

Status PartitionedCallNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[PartitionedCallPrepareTask] Start");
  GE_CHK_STATUS_RET(task.Init(context), "[Init][Task] failed for [%s].", context.GetNodeName());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[PartitionedCallPrepareTask] End");
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
