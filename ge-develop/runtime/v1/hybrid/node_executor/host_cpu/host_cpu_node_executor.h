/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_KERNEL_HOST_CPU_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_HOST_CPU_NODE_EXECUTOR_H_

#include "hybrid/node_executor/node_executor.h"
#include "host_kernels/kernel.h"
#include "hybrid/node_executor/aicpu/aicpu_node_executor.h"

namespace ge {
namespace hybrid {

class HostAicpuNodeTask : public AicpuNodeTask {
 public:
  HostAicpuNodeTask(const NodeItem * const node_item, const domi::TaskDef &task_def)
      : AicpuNodeTask(node_item, task_def) {}
  ~HostAicpuNodeTask() override = default;

  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;

  Status UpdateArgs(TaskContext &context) override;

  Status UpdateExtInfo() override;

  void SetRunKernel(const std::function<uint32_t(void *)> run_cpu_kernel) { run_cpu_kernel_ = run_cpu_kernel; }

  Status SetHostExtInfo() const;

 private:
  Status Execute() const;

  std::function<uint32_t(void *)> run_cpu_kernel_ = nullptr;
};

class HostCpuNodeExecutor : public NodeExecutor {
 public:
  HostCpuNodeExecutor() noexcept : NodeExecutor() {}
  ~HostCpuNodeExecutor() override = default;
  Status PrepareTask(NodeTask &task, TaskContext &context) const override;

  Status LoadTask(const HybridModel &model,
                  const NodePtr &node,
                  std::shared_ptr<NodeTask> &task) const override;

 private:
  static Status ValidateTaskDef(const domi::TaskDef &task_def);
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_KERNEL_HOST_CPU_NODE_EXECUTOR_H_
