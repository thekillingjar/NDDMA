/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_TASK_H_
#define GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_TASK_H_

#include "hybrid/node_executor/node_executor.h"
#include "proto/task.pb.h"

namespace ge {
namespace hybrid {
class RtsNodeTask : public NodeTask {
 public:
  RtsNodeTask() = default;
  ~RtsNodeTask() override = default;
  Status Init(TaskContext &context) override {
    (void)context;
    return SUCCESS;
  }

  virtual Status Init(const HybridModel &model, const NodePtr &node) {
    (void)model;
    GELOGD("[%s] Done initialization successfully.", node->GetName().c_str());
    return SUCCESS;
  }

  Status UpdateArgs(TaskContext &context) override {
    GELOGD("[%s] Done update args successfully.", context.GetNodeName());
    return SUCCESS;
  }

  static Status GetScalarIndexValue(const TaskContext &task_ctx, const int32_t idx, int64_t &val);
 private:
  RtsNodeTask& operator=(const RtsNodeTask&) = delete;
  RtsNodeTask(const RtsNodeTask&) = delete;
};

class StreamActiveNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
};

class StreamSwitchNodeTask : public RtsNodeTask {
 public:
  Status Init(const HybridModel &model, const NodePtr &node) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;

 private:
  std::function<bool(int64_t, int64_t)> comp_func_{nullptr};
};

class StreamMergeNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
};

class PassThroughNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
};

class LabelSetNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
};

class LabelGotoNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
};

class LabelSwitchNodeTask : public RtsNodeTask {
 public:
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_NODE_EXECUTOR_RTS_RTS_NODE_TASK_H_
