/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_3525787DC06241A3B20F0D6DD3F1FF6F_H
#define INC_3525787DC06241A3B20F0D6DD3F1FF6F_H

#include <queue>
#include <functional>
#include <memory>
#include <utility>
#include "exe_graph/runtime/tensor_data.h"
#include "ge/ge_allocator.h"

namespace gert {
namespace memory {
class HostMem : public ge::MemBlock {
 public:
  HostMem(ge::Allocator &allocator, void *ptr, const std::function<void(void *p)> &func, const size_t block_size)
    : ge::MemBlock(allocator, ptr, block_size), ptr_(ptr), deleter_(func) {
  }
  void Free();
  ~HostMem() override {
    try {
      Free();
    } catch (...) {}
  }

 private:
  void *ptr_;
  const std::function<void(void *p)> deleter_;
};
}  // namespace memory
}  // namespace gert
#endif
