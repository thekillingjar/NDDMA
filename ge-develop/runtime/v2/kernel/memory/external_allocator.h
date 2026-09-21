/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_V2_KERNEL_MEMORY_EXTERNAL_ALLOCATOR_H
#define RUNTIME_V2_KERNEL_MEMORY_EXTERNAL_ALLOCATOR_H

#include <cstdint>
#include <array>
#include <memory>
#include "ge/ge_allocator.h"
#include "framework/common/debug/ge_log.h"
#include "framework/memory/allocator_desc.h"
#include "kernel/memory/util/object_allocator.h"
#include "kernel/memory/allocator/scalable_config.h"

namespace gert {
struct ExternalAllocator;
class ExternalMemBlock : public ge::MemBlock {
 public:
  ExternalMemBlock(ge::Allocator &allocator, void *addr, size_t block_size, void *block)
      : MemBlock(allocator, addr, block_size), block_(block) {}

 private:
  void* block_;
  friend struct ExternalAllocator;
};

struct ExternalAllocator : public ge::Allocator {
 public:
  explicit ExternalAllocator(ge::AllocatorDesc allocator_desc) : allocator_desc_(allocator_desc),
    external_mem_block_(ScalableConfig().span_prepared_count) {}

  ge::MemBlock *Malloc(size_t size) override;
  void Free(ge::MemBlock *block) override;
  ge::MemBlock *MallocAdvise(size_t size, void *addr) override;

 private:
  ge::AllocatorDesc allocator_desc_;
  ObjectAllocator<ExternalMemBlock> external_mem_block_;
  std::mutex external_allocator_mutex_;
};
}  // namespace gert

#endif
