/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_KERNEL_LOG_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_KERNEL_LOG_H_
#include "framework/common/debug/ge_log.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "graph/ge_error_codes.h"

#define KLOGE(fmt, ...)                                                                                          \
  GELOGE(ge::FAILED, "[%s][%s]" fmt, reinterpret_cast<gert::ExtendedKernelContext *>(context)->GetKernelType(), \
         reinterpret_cast<ExtendedKernelContext *>(context)->GetKernelName(), ##__VA_ARGS__)
#define KLOGW(fmt, ...)                                                                              \
  GELOGW("[%s][%s]" fmt, reinterpret_cast<gert::ExtendedKernelContext *>(context)->GetKernelType(), \
         reinterpret_cast<ExtendedKernelContext *>(context)->GetKernelName(), ##__VA_ARGS__)
#define KLOGI(fmt, ...)                                                                              \
  GELOGI("[%s][%s]" fmt, reinterpret_cast<gert::ExtendedKernelContext *>(context)->GetKernelType(), \
         reinterpret_cast<ExtendedKernelContext *>(context)->GetKernelName(), ##__VA_ARGS__)
#define KLOGD(fmt, ...)                                                                              \
  GELOGD("[%s][%s]" fmt, reinterpret_cast<gert::ExtendedKernelContext *>(context)->GetKernelType(), \
         reinterpret_cast<ExtendedKernelContext *>(context)->GetKernelName(), ##__VA_ARGS__)

#define KERNEL_CHECK(exp, fmt, ...) \
  do {                              \
    if (!(exp)) {                   \
      KLOGE(fmt, ##__VA_ARGS__);    \
      return ge::GRAPH_FAILED;      \
    }                               \
  } while (false)

#define KERNEL_CHECK_SUCCESS(val) KERNEL_CHECK((val) == ge::GRAPH_SUCCESS, "Get error code %d", val)
#define KERNEL_CHECK_NOTNULL(val) KERNEL_CHECK((val) != nullptr, "Get nullptr %s", #val)

#endif  // AIR_CXX_RUNTIME_V2_KERNEL_KERNEL_LOG_H_
