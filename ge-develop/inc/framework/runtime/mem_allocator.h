/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_MEM_ALLOCATOR_H
#define AIR_MEM_ALLOCATOR_H

#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/allocator.h"
#include "common/ge_visibility.h"
#include "ge/ge_api_error_codes.h"
#include "runtime/mem.h"
#include "ge/ge_allocator.h"

namespace gert {
namespace memory {
struct MemSynchronizer {
  MemSynchronizer() = default;
  virtual ~MemSynchronizer() = default;
  // Wait until the memory is actually freed after task completed
  virtual ge::Status Synchronize() const = 0;
  virtual void Recycle() = 0;
};

class AllocatorManager {
 public:
  AllocatorManager() = default;
  virtual ~AllocatorManager() = default;
  virtual ge::Status Initialize(const std::vector<rtMemType_t> &memory_types) {
    (void)memory_types;
    return ge::SUCCESS;
  }
  virtual void Finalize() {};
  virtual void ReleaseResource(const uint32_t device_id = 0U) {
    (void)device_id;
  }
  virtual ge::Allocator *CreateAllocator(const uint32_t device_id, const rtMemType_t memory_type) {
    (void)device_id;
    (void)memory_type;
    return nullptr;
  }
};
}

class VISIBILITY_EXPORT Allocators {
 public:
  ge::Allocator *GetAllocator(const TensorPlacement &placement, const size_t &usage);
  ge::Status SetAllocator(const TensorPlacement &placement, const size_t &usage,
                          std::shared_ptr<ge::Allocator> &allocator);
 private:
  // todo remove usage
  std::shared_ptr<ge::Allocator> allocators[static_cast<size_t>(TensorPlacement::kTensorPlacementEnd)]
                                                  [static_cast<size_t>(AllocatorUsage::kEnd)];
};
}
#endif  // AIR_CXX_MEM_ALLOCATOR_H
