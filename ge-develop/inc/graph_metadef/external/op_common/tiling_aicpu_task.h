/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXTERNAL_OP_COMMON_TILING_AICPU_TASK_H_
#define EXTERNAL_OP_COMMON_TILING_AICPU_TASK_H_
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
struct TilingAicpuTask {
  gert::TilingContext *tilingContext;
  const char *opType;
  char reserve[64];
};
}  // namespace optiling

#endif