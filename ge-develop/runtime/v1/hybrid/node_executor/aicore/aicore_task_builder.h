/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_KERNEL_AICORE_TASK_BUILDER_H_
#define GE_HYBRID_KERNEL_AICORE_TASK_BUILDER_H_

#include <vector>
#include <string>
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/attr_utils.h"
#include "graph/op_kernel_bin.h"
#include "proto/task.pb.h"
#include "hybrid/model/hybrid_model.h"

namespace ge {
namespace hybrid {
class AiCoreNodeTask;

class AiCoreTaskBuilder {
 public:
  AiCoreTaskBuilder(const NodePtr &node, const std::vector<domi::TaskDef> &task_defs, const HybridModel &model,
                    AiCoreNodeTask &aicore_node_task);
  ~AiCoreTaskBuilder() = default;

  Status BuildTask();
  Status LoadAicpuTask();
  Status LoadFusedTask();

 private:
  bool ExpectAtomicAddrCleanTask() const;
  Status InitTaskDef();
  Status LoadAtomicWorkspace();

  NodePtr node_;
  OpDescPtr op_desc_;
  const std::vector<domi::TaskDef> &task_defs_;
  const HybridModel &model_;
  AiCoreNodeTask &aicore_node_task_;
  std::vector<domi::TaskDef> aicore_task_defs_;
  std::vector<const domi::TaskDef *> aicpu_task_defs_;  // aicpu executor references taskdef during execution.
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_KERNEL_AICORE_TASK_BUILDER_H_
