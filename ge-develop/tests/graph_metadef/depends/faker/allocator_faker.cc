/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allocator_faker.h"
namespace gert {
  GertMemBlock *AllocatorFaker::Malloc(size_t size) {
    auto block = new (std::nothrow) GertMemBlockFaker(malloc(size));
    block_pool_.emplace_back(block);
    return block;
  }
  void AllocatorFaker::Free(GertMemBlock *block) {
    if (block != nullptr) {
      free(block->GetAddr());
      delete block;
    }
  }
  AllocatorFaker::~AllocatorFaker() {
    for (auto &block : block_pool_) {
      if (block != nullptr) {
        delete block;
        block = nullptr;
      }
    }
  }

  GertTensorData AllocatorFaker::MallocTensorData(size_t size) {
    return {size, kOnDeviceHbm, 0, Malloc(size)};
  }
  }
