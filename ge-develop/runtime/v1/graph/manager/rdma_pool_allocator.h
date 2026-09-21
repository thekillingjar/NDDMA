/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_RDMA_POOL_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_RDMA_POOL_ALLOCATOR_H_

#include <iostream>
#include <mutex>
#include <unordered_map>

#include "graph/manager/block_memory.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/node.h"
#include "runtime/mem.h"

namespace ge {
class RdmaPoolAllocator {
 public:
  explicit RdmaPoolAllocator(const rtMemType_t memory_type);

  RdmaPoolAllocator(const RdmaPoolAllocator &) = delete;

  RdmaPoolAllocator &operator=(const RdmaPoolAllocator &) & = delete;

  ~RdmaPoolAllocator() = default;

  Status Initialize();
  void Finalize();

  Status InitMemory(const size_t mem_size);

  uint8_t *Malloc(const size_t size, const uint32_t device_id = 0U);

  Status Free(uint8_t *const memory_addr, const uint32_t device_id = 0U);

  Status GetBaseAddr(uint64_t &base_addr, uint64_t &mem_size) const;

  uint8_t *GetRdmaBaseAddr() const { return rdma_base_addr_; }

 private:
  void MergeBlocks(Block &dst, Block &src);

  rtMemType_t memory_type_;
  size_t rdma_mem_size_ = 0U;  // Total rdma memory size to be allocated.
  uint8_t *rdma_base_addr_ = nullptr;
  MemoryAllocator *memory_allocator_ = nullptr;
  BlockBin block_bin_;  // Save all rdma blocks.
  std::unordered_map<uint8_t *, Block *> allocated_blocks_;
  // lock around all operations
  mutable std::recursive_mutex mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_RDMA_POOL_ALLOCATOR_H_
