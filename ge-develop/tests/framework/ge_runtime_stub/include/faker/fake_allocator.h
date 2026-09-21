/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_SINGLE_OP_FAKE_ALLOCATOR_H
#define TESTS_SINGLE_OP_FAKE_ALLOCATOR_H
#include "ge/ge_allocator.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/runtime/device_memory_recorder.h"
#include <vector>
namespace ge {
class FakeAllocator : public ge::Allocator {
 public:
  ge::MemBlock *Malloc(size_t size) {
    auto addr = MakeUnique<uint8_t[]>(size);
    auto block = MakeUnique<MemBlock>(*this, (void *)addr.get(), size);
    addrs_.emplace_back(std::move(addr));
    blocks_.emplace_back(std::move(block));
    gert::DeviceMemoryRecorder::SetRecorder((void *)addr.get(), size);
    return blocks_.back().get();
  }

  void Free(MemBlock *block){
    size_t index = 0UL;
    for (size_t i = 0UL; i < addrs_.size(); ++i) {
      if (addrs_[i].get() == block->GetAddr()) {
        index = i;
        break;
      }
    }
    addrs_.erase(addrs_.begin() + index);
    blocks_.erase(blocks_.begin() + index);
  }

  std::vector<std::unique_ptr<uint8_t[]>> addrs_{};
  std::vector<std::unique_ptr<MemBlock>> blocks_{};
};
}  // namespace ge
#endif  // TESTS_SINGLE_OP_FAKE_ALLOCATOR_H
