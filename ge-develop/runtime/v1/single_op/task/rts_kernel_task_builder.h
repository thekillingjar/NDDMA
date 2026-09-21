/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_TASK_RTS_KERNEL_TASK_BUILDER_H_
#define GE_SINGLE_OP_TASK_RTS_KERNEL_TASK_BUILDER_H_

#include "graph/op_desc.h"
#include "single_op/single_op.h"
#include "single_op/single_op_model.h"

namespace ge {
using GetOpDescFunc = std::function<ge::Status(uint32_t, std::shared_ptr<ge::OpDesc> &)>;
class RtsKernelTaskBuilder {
 public:
  static Status BuildMemcpyAsyncTask(const GetOpDescFunc &get_op_desc_func, const SingleOpModelParam &param,
                                     const domi::TaskDef &task_def, OpTask *&op_task, StreamResource &resource);

  static Status BuildNpuGetFloatStatusTask(const GetOpDescFunc &get_op_desc_func, const SingleOpModelParam &param,
                                           const domi::TaskDef &task_def, OpTask *&op_task, StreamResource &resource);
  static Status BuildNpuClearFloatStatusTask(const GetOpDescFunc &get_op_desc_func, const domi::TaskDef &task_def,
                                             OpTask *&op_task, StreamResource &resource);
  static Status BuildNpuGetFloatDebugStatusTask(const GetOpDescFunc &get_op_desc_func, const SingleOpModelParam &param,
                                           const domi::TaskDef &task_def, OpTask *&op_task, StreamResource &resource);
  static Status BuildNpuClearFloatDebugStatusTask(const GetOpDescFunc &get_op_desc_func, const domi::TaskDef &task_def,
                                             OpTask *&op_task, StreamResource &resource);
  static void UpdateCopyKind(const OpDescPtr &op_desc, rtMemcpyKind_t &kind);
};
}  // namespace ge
#endif  // GE_SINGLE_OP_TASK_RTS_KERNEL_TASK_BUILDER_H_
