/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_EXECUTOR_EXECUTION_ENGINE_H_
#define GE_HYBRID_EXECUTOR_EXECUTOR_EXECUTION_ENGINE_H_

#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/node_executor/task_context.h"
#include "common/dump/dump_op.h"

namespace ge {
namespace hybrid {
class NodeDoneCallback {
 public:
  NodeDoneCallback(GraphExecutionContext *const graph_context, const std::shared_ptr<TaskContext> task_context);
  ~NodeDoneCallback() = default;
  Status OnNodeDone();
  static Status GetTaskDescInfo(TaskContext &context, const NodePtr node, std::vector<TaskDescInfo> &task_desc_info);

 private:
  Status PrepareConstInputs(const NodeItem &node_item) const;
  Status DumpDynamicNode();
  Status SaveDumpOpInfo();
  GraphExecutionContext *graph_context_;
  std::shared_ptr<TaskContext> context_;
  DumpOp dump_op_;
};

class ExecutionEngine {
 public:
  static Status ExecuteAsync(const NodeState &node_state,
                             const std::shared_ptr<TaskContext> &task_context,
                             GraphExecutionContext &execution_context,
                             const std::function<void()> &callback);

 private:
  static Status ValidateInputTensors(const NodeState &node_state, const TaskContext &task_context);
  static Status PropagateOutputs(const NodeItem &node_item, const TaskContext &task_context,
                                 const GraphExecutionContext &context);
  static Status DoExecuteAsync(const NodeState &node_state,
                               TaskContext &task_context,
                               GraphExecutionContext &context,
                               const std::function<void()> &callback);
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_EXECUTOR_EXECUTION_ENGINE_H_
