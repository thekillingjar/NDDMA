/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_EXECUTOR_EXECUTOR_BASE_DEF_H_
#define AIR_CXX_RUNTIME_V2_CORE_EXECUTOR_EXECUTOR_BASE_DEF_H_
#include "exe_graph/runtime/base_type.h"
#include "exe_graph/runtime/kernel_run_context.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef size_t NodeIdentity;
typedef UINT32 (*KernelFunc)(KernelRunContext *context);
typedef struct {
  char head[8];
  NodeIdentity node_id;
  KernelFunc func;
  KernelRunContext context;
} Node;
#ifdef __cplusplus
}
#endif
#endif  // AIR_CXX_RUNTIME_V2_CORE_EXECUTOR_EXECUTOR_BASE_DEF_H_
