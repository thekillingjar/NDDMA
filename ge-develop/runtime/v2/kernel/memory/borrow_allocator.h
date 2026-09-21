/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BORROW_ALLOCATOR_H
#define AIR_CXX_BORROW_ALLOCATOR_H

#include <vector>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include "ge/ge_allocator.h"
#include "kernel/memory/multi_stream_mem_block.h"

namespace gert {
namespace memory {
class BorrowAllocator {
 public:
  MultiStreamMemBlock *Alloc(size_t size);

  // 将block存到size_to_borrow_block和stream_to_borrow_block中
  void Free(MultiStreamMemBlock *block);

  std::list<MultiStreamMemBlock *> GetAndClearBorrowBlocks(int64_t stream_id);

 private:
  std::map<size_t, std::unordered_set<MultiStreamMemBlock *>> size_to_borrow_block;
  std::unordered_map<int64_t, std::unordered_set<MultiStreamMemBlock *>> stream_to_borrow_block;
};
}  // namespace memory
}  // namespace gert

#endif  // AIR_CXX_BORROW_ALLOCATOR_H
