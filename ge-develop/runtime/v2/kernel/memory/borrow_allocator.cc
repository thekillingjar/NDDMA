/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "borrow_allocator.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
namespace memory {
MultiStreamMemBlock *BorrowAllocator::Alloc(size_t size) {
  // size_to_borrow_block存在的键值对一定是非空的
  auto iter = size_to_borrow_block.lower_bound(size);
  if (iter != size_to_borrow_block.end()) {
    auto &blocks = iter->second;
    auto block = *blocks.begin();
    blocks.erase(block);
    stream_to_borrow_block[block->GetBirthStreamId()].erase(block);
    if (blocks.empty()) {
      size_to_borrow_block.erase(size);
    }
    return block;
  }
  return nullptr;
}

void BorrowAllocator::Free(MultiStreamMemBlock *block) {
  const auto size = block->GetSize();
  const auto size_iter = size_to_borrow_block.find(size);
  if (size_iter == size_to_borrow_block.end()) {
    size_to_borrow_block.emplace(size, std::unordered_set<MultiStreamMemBlock *>{block});
  } else {
    size_iter->second.emplace(block);
  }
  const auto stream_id = block->GetBirthStreamId();
  const auto stream_iter = stream_to_borrow_block.find(stream_id);
  if (stream_iter == stream_to_borrow_block.end()) {
    stream_to_borrow_block.emplace(stream_id, std::unordered_set<MultiStreamMemBlock *>{block});
  } else {
    stream_iter->second.emplace(block);
  }
}

std::list<MultiStreamMemBlock *> BorrowAllocator::GetAndClearBorrowBlocks(int64_t stream_id) {
  std::list<MultiStreamMemBlock *> target_blocks;
  auto stream_iter = stream_to_borrow_block.find(stream_id);
  if (stream_iter == stream_to_borrow_block.end() || stream_iter->second.empty()) {
    return target_blocks;
  }
  for (const auto &block : stream_iter->second) {
    target_blocks.push_back(block);
    const auto block_size = block->GetSize();
    auto size_iter = size_to_borrow_block.find(block_size);
    if (size_iter != size_to_borrow_block.end()) {
      auto &blocks = size_iter->second;
      blocks.erase(block);
      if (blocks.empty()) {
        size_to_borrow_block.erase(block_size);
      }
    }
  }
  stream_to_borrow_block.erase(stream_id);
  return target_blocks;
}
}  // namespace memory
}  // namespace gert
