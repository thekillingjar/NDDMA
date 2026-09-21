/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_EXECUTOR_H_
#define GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_EXECUTOR_H_

#include "hybrid/node_executor/node_executor.h"
#include "hybrid/node_executor/rts/rts_node_task.h"

namespace ge {
namespace hybrid {
class IdentityNodeTask : public RtsNodeTask {
 public:
  Status Init(const HybridModel &model, const NodePtr &node) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;

 protected:
  Status DoCopyTensor(const TaskContext &context, const int32_t index) const;

 private:
  tagRtMemcpyKind kind_{RT_MEMCPY_DEVICE_TO_DEVICE};
};

class IdentityNNodeTask : public IdentityNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
};

class StartOfSequenceTask : public RtsNodeTask {
 public:
  Status Init(const HybridModel &model, const NodePtr &node) override;

  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;

 private:
  uint32_t model_id_ = 0U;
};

class NpuGetFloatStatusTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  ~NpuGetFloatStatusTask() override {
    GE_FREE_RT_LOG(args_);
  }

 private:
  void *args_{nullptr};
};

class NpuClearFloatStatusTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
};

class ProfilingTraceNodeTask : public RtsNodeTask {
 public:
  Status Init(const HybridModel &model, const NodePtr &node) override;

  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;

 private:
  std::vector<domi::TaskDef> task_defs_;
  uint32_t model_id_ = 0U;
};

class RtsNodeExecutor : public NodeExecutor {
 public:
  RtsNodeExecutor() noexcept : NodeExecutor() {}
  ~RtsNodeExecutor() override = default;
  Status PrepareTask(NodeTask &task, TaskContext &context) const override;
  Status LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const override;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_EXECUTOR_H_
