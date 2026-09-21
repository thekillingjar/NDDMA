/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HYBRID_HCCL_NODE_EXECUTOR_H_
#define HYBRID_HCCL_NODE_EXECUTOR_H_
#include "hccl/base.h"
#include "common/opskernel/ge_task_info.h"
#include "graph/op_desc.h"
#include "graph/runtime_inference_context.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {

class HcclNodeTask : public NodeTask {
 public:
  HcclNodeTask() noexcept : NodeTask() {}

  ~HcclNodeTask() override {}

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  Status Init(TaskContext &context) override;

 private:
  static Status FillHcomOpInfo(const TaskContext &context, const OpDescPtr op_desc,
                               const std::vector<void *> &inputs,
                               const std::vector<void *> &outputs,
                               HcomOperation &hcom_op_info);
  static Status GetInputsOutPuts(const TaskContext &context, std::vector<void *> &inputs,
                                 std::vector<void *> &outputs);
  std::shared_ptr<DavinciModel> davinci_model_ = nullptr;
  std::mutex hccl_mutex_;
  std::condition_variable cond_;
};

class RdmaNodeTask : public NodeTask {
 public:
  RdmaNodeTask() = default;

  ~RdmaNodeTask() override {}

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  Status Init(TaskContext &context) override;

 private:
  Status SetAddrInfo(const TaskContext &context, const RuntimeInferenceContext &rt_ctx, const uint64_t * const data,
                     const size_t row_num, std::vector<HcomRemoteAccessAddrInfo> &addr_infos) const;
  Status GetOffsetTensor(const TaskContext &context, const RuntimeInferenceContext &rt_ctx,
                         const size_t row_num, GeTensorPtr &offset_tensor) const;
  Status ExtractTensor(const TaskContext &context, std::vector<HcomRemoteAccessAddrInfo> &addr_infos) const;
  std::pair<int64_t, int64_t> remote_index_;
  std::pair<int64_t, int64_t> offset_index_;
  int32_t local_index_ = 0;
  std::mutex hccl_mutex_;
  std::condition_variable cond_;
  bool skip_flag_ = false;
};


class AllToAllNodeTask : public NodeTask {
 public:
  AllToAllNodeTask() = default;

  ~AllToAllNodeTask() override = default;

  Status UpdateArgs(TaskContext &context) override {
    (void)context;
    return SUCCESS;
  }
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  Status Init(TaskContext &context) override {
    (void)context;
    return SUCCESS;
  }

 private:
  std::mutex hccl_mutex_;
  std::condition_variable cond_;
};

class HcclNodeExecutor : public NodeExecutor {
 public:
  Status LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const override;
  Status PrepareTask(NodeTask &task, TaskContext &context) const override;
  Status ExecuteTask(NodeTask &task, TaskContext &context, const std::function<void()> &callback) const override;
  Status Initialize() override;
  Status Finalize() override;
  HcclNodeExecutor() noexcept : NodeExecutor(), handle_(nullptr) {}
  ~HcclNodeExecutor() override = default;

 private:
  void *handle_;
};
}  // namespace hybrid
}  // namespace ge

#endif  // HYBRID_HCCL_NODE_EXECUTOR_H_
