/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_RUNTIME_V2_KERNEL_TAGS_H
#define AIR_CXX_RUNTIME_V2_KERNEL_TAGS_H

#include <vector>
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task_type.h"
#include "core/execution_data.h"


namespace gert {
struct KernelTags {
  void Reset(size_t node_num, size_t thread_num = 0U);
  ExecTaskType GetByNode(const Node *kernel) {
    if (node_tags_[kernel->node_id] != ExecTaskType::MAX) {
      return node_tags_[kernel->node_id];
    }
    return UpdateTag(kernel);
  }

 private:
  ExecTaskType UpdateTag(const Node *kernel);

 private:
  size_t thread_num_ = 0U;
  std::vector<ExecTaskType> node_tags_;
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_KERNEL_TAGS_H
