/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/host_mem_allocator.h"
#include "framework/common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"

namespace ge {
void *HostMemAllocator::Malloc(const std::shared_ptr<AlignedPtr> &aligned_ptr, const size_t size) {
  if (aligned_ptr == nullptr) {
    GELOGW("Insert a null aligned_ptr, size=%zu", size);
    if (size == 0U) {
      allocated_blocks_[nullptr] = { size, nullptr };
    }
    return nullptr;
  }
  GELOGD("allocate existed host memory succ, size=%zu", size);
  allocated_blocks_[aligned_ptr->Get()] = { size, aligned_ptr };
  return aligned_ptr->MutableGet();
}

void *HostMemAllocator::Malloc(const size_t size) {
  GELOGD("start to malloc host memory, size=%zu", size);
  const std::lock_guard<std::mutex> lock(mutex_);
  const std::shared_ptr<AlignedPtr> aligned_ptr = MakeShared<AlignedPtr>(size);
  if (aligned_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New AlignedPtr fail, size:%zu", size);
    GELOGE(INTERNAL_ERROR, "[Call][MakeShared] for AlignedPtr failed, size:%zu", size);
    return nullptr;
  }
  allocated_blocks_[aligned_ptr->Get()] = { size, aligned_ptr };
  GELOGD("allocate host memory succ, size=%zu", size);
  return aligned_ptr->MutableGet();
}

Status HostMemAllocator::Free(const void *const memory_addr) {
  if (memory_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param memory_addr is nullptr, check invalid");
    GELOGE(GE_GRAPH_FREE_FAILED, "[Check][Param] Invalid memory pointer");
    return GE_GRAPH_FREE_FAILED;
  }

  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = allocated_blocks_.find(memory_addr);
  if (it == allocated_blocks_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Memory_addr is not alloc before, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Invalid memory pointer:%p", memory_addr);
    return PARAM_INVALID;
  }
  it->second.second.reset();
  (void)allocated_blocks_.erase(it);

  return SUCCESS;
}

void HostMemAllocator::Clear() {
  for (auto &block : allocated_blocks_) {
    if (block.second.second != nullptr) {
      block.second.second.reset();
    }
  }
  allocated_blocks_.clear();
}
}  // namespace ge
