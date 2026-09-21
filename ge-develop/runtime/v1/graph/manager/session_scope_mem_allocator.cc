/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/session_scope_mem_allocator.h"

#include <set>
#include <string>
#include <utility>

#include "framework/common/debug/ge_log.h"
#include "graph/manager/mem_manager.h"

namespace ge {

SessionScopeMemAllocator::SessionScopeMemAllocator(const rtMemType_t memory_type,
                                                   MemoryAllocator *const memory_allocator)
    : memory_type_(memory_type), memory_allocator_(memory_allocator) {}

Status SessionScopeMemAllocator::Initialize(const uint32_t device_id) {
  (void)device_id;
  // when redo Initialize free old memory
  FreeAllMemory();
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  memory_allocator_ = &MemManager::Instance().MemInstance(memory_type_);
  return SUCCESS;
}

void SessionScopeMemAllocator::Finalize(const uint32_t device_id) {
  (void)device_id;
  FreeAllMemory();
}

uint8_t *SessionScopeMemAllocator::Malloc(size_t size, const uint64_t session_id, const uint32_t device_id) {
  if (memory_allocator_ == nullptr) {
    return nullptr;
  }
  GELOGI("Start malloc memory, size:%zu, session id:%" PRIu64 " device id:%u", size, session_id, device_id);
  const std::string purpose = "Memory for session scope";
  const auto ptr = memory_allocator_->MallocMemory(purpose, size, device_id);
  if (ptr == nullptr) {
    GELOGE(FAILED, "Malloc failed, no enough memory for size:%zu, session_id:%" PRIu64 " device_id:%u", size,
           session_id, device_id);
    return nullptr;
  }
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::shared_ptr<uint8_t> mem_ptr(ptr, [this](uint8_t *const p) { (void)memory_allocator_->FreeMemory(p); });
  allocated_memory_[session_id].emplace_back(size, mem_ptr);
  return ptr;
}

Status SessionScopeMemAllocator::Free(const uint64_t session_id, const uint32_t device_id) {
  GELOGI("Free session:%" PRIu64 " memory, device id:%u.", session_id, device_id);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = allocated_memory_.find(session_id);
  if (it == allocated_memory_.end()) {
    GELOGW("Invalid session_id");
    return PARAM_INVALID;
  }
  for (const auto &mem_info : it->second) {
    GELOGI("Free memory size:%zu.", mem_info.GetSize());
  }
  (void)allocated_memory_.erase(it);
  return SUCCESS;
}

void SessionScopeMemAllocator::FreeAllMemory() {
  GELOGI("Free all memory");
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &session_mem : allocated_memory_) {
    session_mem.second.clear();
  }
  allocated_memory_.clear();
}
}  // namespace ge
