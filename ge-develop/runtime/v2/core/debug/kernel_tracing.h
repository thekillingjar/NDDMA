/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_DEBUG_KERNEL_TRACING_H_
#define AIR_CXX_RUNTIME_V2_CORE_DEBUG_KERNEL_TRACING_H_
#include "exe_graph/runtime/extended_kernel_context.h"
#include "framework/runtime/subscriber/global_tracer.h"
namespace gert {
#define KERNEL_TRACE_ENABLE (GlobalTracer::GetInstance()->GetEnableFlags() != 0U)
#define KERNEL_TRACE_INNER(k_name, fmt, ...)                \
  do {                                                      \
    if (!KERNEL_TRACE_ENABLE) {                             \
      break;                                                \
    }                                                       \
    GELOGI("[KernelTrace][%s]" fmt, k_name, ##__VA_ARGS__); \
  } while (0)

#define KERNEL_TRACE(fmt, ...) \
  KERNEL_TRACE_INNER(reinterpret_cast<const ExtendedKernelContext *>(context)->GetKernelName(), fmt, ##__VA_ARGS__)
#define KERNEL_TRACE_NO_CTX(fmt, ...) KERNEL_TRACE_INNER("", fmt, ##__VA_ARGS__)

#define TRACE_STR_ALLOC_MEM "[MEM]Alloc memory at stream %" PRId64 ", block %p, address %p, size %zu"
#define KERNEL_TRACE_ALLOC_MEM(stream, block_addr, dev_addr, size) \
  KERNEL_TRACE(TRACE_STR_ALLOC_MEM, stream, block_addr, dev_addr, size)

#define TRACE_STR_FREE_MEM "[MEM]Free memory at stream %" PRId64 ", address %p"
#define KERNEL_TRACE_FREE_MEM(stream, dev_addr) KERNEL_TRACE(TRACE_STR_FREE_MEM, stream, dev_addr)
#define KERNEL_TRACE_WANDERING(src_stream, dst_stream, dev_addr)                                                   \
  KERNEL_TRACE("[MEM]Memory wandering from stream %" PRId64 " to %" PRId64 ", address %p", src_stream, dst_stream, \
               dev_addr)
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_CORE_DEBUG_KERNEL_TRACING_H_
