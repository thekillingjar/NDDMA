/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_TASK_AICPU_C_C_TASK_BUILDER_H_
#define GE_SINGLE_OP_TASK_AICPU_C_C_TASK_BUILDER_H_

#include "graph/op_desc.h"
#include "aicpu_task_struct.h"
#include "single_op/single_op.h"
#include "single_op/single_op_model.h"
#include "runtime/mem.h"

namespace ge {
class AiCpuCCTaskBuilder {
 public:
  explicit AiCpuCCTaskBuilder(const OpDescPtr &op_desc, const domi::KernelDef &kernel_def);
  ~AiCpuCCTaskBuilder() = default;

  Status BuildTask(AiCpuCCTask &task, const uint64_t kernel_id,
                   const SingleOpModelParam &param) const;

 private:
  Status SetKernelArgs(AiCpuCCTask &task, const SingleOpModelParam &param) const;
  const OpDescPtr op_desc_;
  const domi::KernelDef &kernel_def_;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_TASK_AICPU_C_C_TASK_BUILDER_H_
