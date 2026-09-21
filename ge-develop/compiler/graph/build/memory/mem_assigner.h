/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_MEM_ASSIGNER_H_

#include "framework/common/ge_inner_error_codes.h"

namespace ge {
static const int64_t kInvalidOffset = -1;
static const int64_t MEM_ALIGN_SIZE = 512;

class MemAssigner {
 public:
  MemAssigner() = default;

  MemAssigner(const MemAssigner &) = delete;

  MemAssigner &operator=(const MemAssigner &) = delete;

  virtual ~MemAssigner() = default;

  virtual Status Assign() = 0;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_MEM_ASSIGNER_H_
