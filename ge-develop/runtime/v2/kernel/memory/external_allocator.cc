/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "external_allocator.h"

namespace gert {
ge::MemBlock *ExternalAllocator::Malloc(size_t size) {
  const std::unique_lock<std::mutex> lk(external_allocator_mutex_);
  auto block = allocator_desc_.alloc_func(allocator_desc_.obj, size);
  if (block == nullptr) {
    GELOGE(ge::FAILED, "Failed to malloc memory by external allocator, size: %zu", size);
    return nullptr;
  }
  GELOGD("Malloc memory by external allocator success, block:%p, size: %zu", block, size);

  auto mem_block = new (external_mem_block_.Alloc()) ExternalMemBlock(*this,
      allocator_desc_.get_addr_from_block_func(block), size, block);
  GELOGD("Malloc MemBlock by GE, MemBlock:%p.", mem_block);
  return dynamic_cast<ge::MemBlock *>(mem_block);
}

void ExternalAllocator::Free(ge::MemBlock *block) {
  auto external_mem_block = dynamic_cast<ExternalMemBlock *>(block);
  if (external_mem_block == nullptr) {
    return;
  }
  // free external allocator block
  GELOGD("Free the block which malloc by external allocator, block:%p.", external_mem_block->block_);
  const std::unique_lock<std::mutex> lk(external_allocator_mutex_);
  allocator_desc_.free_func(allocator_desc_.obj, external_mem_block->block_);

  // free mem_block
  GELOGD("Free the memBlock which malloc by GE, memBlock:%p.", block);
  external_mem_block_.Free(*(external_mem_block));
}

ge::MemBlock *ExternalAllocator::MallocAdvise(size_t size, void *addr) {
  const std::unique_lock<std::mutex> lk(external_allocator_mutex_);
  if (allocator_desc_.alloc_advise_func == nullptr) {
    return Malloc(size);
  }
  auto block = allocator_desc_.alloc_advise_func(allocator_desc_.obj, size, addr);
  if (block == nullptr) {
    GELOGE(ge::FAILED, "Failed to malloc advise memory by external allocator, size: %zu", size);
    return nullptr;
  }
  GELOGD("Malloc advise memory by external allocator success, block:%p, size: %zu", block, size);

  auto mem_block = new (external_mem_block_.Alloc()) ExternalMemBlock(*this,
      allocator_desc_.get_addr_from_block_func(block), size, block);
  GELOGD("Malloc MemBlock by GE, MemBlock:%p.", mem_block);
  return dynamic_cast<ge::MemBlock *>(mem_block);
}
}  // namespace gert
