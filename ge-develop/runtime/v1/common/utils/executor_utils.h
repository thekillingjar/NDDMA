/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_UTILS_EXECUTOR_UTILS_H_
#define GE_COMMON_UTILS_EXECUTOR_UTILS_H_

#include "graph/op_desc.h"
#include "single_op/task/op_task.h"
#include "hybrid/node_executor/task_context.h"
#include "framework/common/runtime_tensor_desc.h"

namespace ge {
class ExecutorUtils {
 public:
  static bool HasHostMemInput(const OpDescPtr &op_desc);
  static Status UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs,
                                       const OpTask &op_task,
                                       void *const io_base,
                                       const size_t io_size,
                                       vector<rtHostInputInfo_t> &host_inputs);
  static Status UpdateHostMemInputArgs(const hybrid::TaskContext &context,
                                       void *const io_addrs,
                                       const size_t io_addrs_size,
                                       const size_t host_mem_input_data_offset_in_args,
                                       vector<rtHostInputInfo_t> &host_inputs, const bool need_64b_aligned = false);
  static Status LoadAtomicWorkspace(const OpDescPtr &op_desc);
  static Status InitAtomicAddrCleanIndices(const OpDescPtr &op_desc, std::vector<int32_t> &atomic_output_indices,
                                           std::vector<int32_t> &atomic_workspace_indices);
  static bool GetOpIndex(const domi::TaskDef &task_def, uint32_t &op_index);

  static Status AssembleReuseBinaryArgs(const OpDescPtr &op_desc,
                                        optiling::utils::OpRunInfo &run_info,
                                        rtArgsEx_t &args_ex);
};
}

#endif  // GE_COMMON_UTILS_EXECUTOR_UTILS_H_
