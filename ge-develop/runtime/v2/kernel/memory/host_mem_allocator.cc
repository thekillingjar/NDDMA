/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_mem_allocator.h"
#include "multi_stream_mem_block_helper.h"
#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "mem_block.h"
#include "core/utils/tensor_utils.h"

namespace gert {
namespace memory {
ge::MemBlock *HostMemAllocator::Malloc(size_t size) {
  auto host_addr_holder = ge::MakeUnique<uint8_t[]>(size);
  if (host_addr_holder == nullptr) {
    return nullptr;
  }
  auto *host_mem = new (std::nothrow) HostMem(
      *this, host_addr_holder.get(), [](void *p) { delete[] reinterpret_cast<uint8_t *>(p); }, size);
  if (host_mem == nullptr) {
    return nullptr;
  }
  host_addr_holder.release();
  return host_mem;
}

void HostMemAllocator::Free(ge::MemBlock *block) {
  if (block != nullptr) {
    (dynamic_cast<HostMem *>(block))->Free();
    delete block;
  }
}
}  // namespace memory
}  // namespace gert
