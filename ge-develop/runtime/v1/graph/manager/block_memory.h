/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_BLOCK_MEMORY_H_
#define GE_GRAPH_MANAGER_BLOCK_MEMORY_H_

#include <cstdint>
#include <set>
#include <functional>

namespace ge {
struct Block;
using Comparison = std::function<bool(const Block *, const Block *)>;
using BlockBin = std::set<Block *, Comparison>;

struct Block {
  uint32_t device_id;      // npu device id
  size_t size;             // block size in bytes
  BlockBin *bin;           // owning block bin
  uint8_t *ptr;            // memory address
  bool allocated;          // in-use flag
  Block *prev;             // prev block if split from a larger allocation
  Block *next;             // next block if split from a larger allocation

  Block(const uint32_t device, const size_t block_size, BlockBin *const block_bin, uint8_t *const mem_addr)
      : device_id(device),
        size(block_size),
        bin(block_bin),
        ptr(mem_addr),
        allocated(false),
        prev(nullptr),
        next(nullptr) {}

  // constructor for search key
  Block(const uint32_t device, const size_t block_size, uint8_t *const mem_addr)
      : Block(device, block_size, nullptr, mem_addr) {}

  bool IsSplit() const { return (prev != nullptr) || (next != nullptr); }
};
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_BLOCK_MEMORY_H_
