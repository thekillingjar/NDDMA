/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_AICPU_UPDATE_KERNEL_H
#define AIR_CXX_AICPU_UPDATE_KERNEL_H

#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/continuous_vector.h"

namespace gert {
namespace kernel {
enum class MaxThreadSizeInputIndex {
  kOutputId,
  kNotLastOutSlice,
  kLastOutSlice
};

enum class ThreadParamInputIndex {
  kCutDim,
  kInputIndexes,
  kOutputIndexes,
  kNotLastInSlice,
  kNotLastOutSlice
};

enum class UpdateContextInputIndex {
  kFlushData,
  kCtxIds,
  kThreadDim
};

constexpr size_t MAX_OFFSET_NUM = 64UL;
struct AICpuThreadParam {
  size_t input_num{0U};
  size_t output_num{0U};
  size_t input_output_num{0U};
  uint32_t thread_dim{0U};
  uint64_t task_addr_offset[MAX_OFFSET_NUM]; // input + output
};

struct AICpuSubTaskFlush {
  void *task_base_addr_dev{nullptr};   // base address on the device
  void *args_base_addr_dev{nullptr};   // args address on the device
  size_t task_size{0UL};
  size_t arg_size{0UL};
  void *task_base_addr_host{nullptr};  // base address on the host
  void *args_base_addr_host{nullptr};  // args address on the host
};
}
}
#endif // AIR_CXX_AICPU_UPDATE_KERNEL_H
