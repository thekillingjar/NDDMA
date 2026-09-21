/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXE_GRAPH_RUNTIME_ALLOCATOR_H_
#define METADEF_INC_EXE_GRAPH_RUNTIME_ALLOCATOR_H_
#include <string>
#include <tuple>
#include "exe_graph/runtime/tensor_data.h"

namespace gert {
enum class AllocatorUsage {
  kAllocNodeOutput,
  kAllocNodeWorkspace,
  kAllocNodeShapeBuffer,
  kEnd
};
struct AllocatorDesc {
  TensorPlacement placement;
  AllocatorUsage usage;
  bool operator<(const AllocatorDesc &other) const {
    return std::tie(placement, usage) < std::tie(other.placement, other.usage);
  }
  std::string GetKey() const {
    return "Allocator-" + std::to_string(placement);
  }
};
}
#endif  // METADEF_INC_EXE_GRAPH_RUNTIME_ALLOCATOR_H_
