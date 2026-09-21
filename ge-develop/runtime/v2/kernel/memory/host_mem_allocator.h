/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_HOST_MEM_ALLOCATOR_H
#define AIR_HOST_MEM_ALLOCATOR_H

#include <vector>
#include <iostream>
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "graph/utils/math_util.h"
#include "runtime/mem_allocator.h"
#include "ge/ge_allocator.h"
#include "single_stream_l2_allocator.h"

namespace gert {
namespace memory {
struct HostMemAllocator : public ge::Allocator {
  ge::MemBlock *Malloc(size_t size) override;
  void Free(ge::MemBlock *block) override;
};

class HostGertMemAllocator : public SingleStreamL2Allocator {
 public:
  explicit HostGertMemAllocator(HostMemAllocator *host_mem_allocator = nullptr)
      : SingleStreamL2Allocator(kOnHost, host_mem_allocator) {}
};
}  // namespace memory
}  // namespace gert
#endif  // AIR_HOST_MEM_ALLOCATOR_H
