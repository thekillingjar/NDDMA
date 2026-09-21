/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_SESSION_SCOPE_MEM_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_SESSION_SCOPE_MEM_ALLOCATOR_H_

#include <iostream>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "graph/node.h"
#include "runtime/mem.h"
#include "graph/manager/graph_mem_allocator.h"

namespace ge {
class SessionScopeMemoryInfo {
 public:
  SessionScopeMemoryInfo(const size_t size, const std::shared_ptr<uint8_t> &ptr)
      : size_(size), ptr_(ptr) {}
  SessionScopeMemoryInfo() = delete;
  virtual ~SessionScopeMemoryInfo() = default;

  SessionScopeMemoryInfo(const SessionScopeMemoryInfo &) = default;

  SessionScopeMemoryInfo &operator=(const SessionScopeMemoryInfo &) & = default;

  size_t GetSize() const {
    return size_;
  }

 private:
  size_t size_;
  std::shared_ptr<uint8_t> ptr_;
};

class SessionScopeMemAllocator {
 public:
  explicit SessionScopeMemAllocator(const rtMemType_t memory_type, MemoryAllocator *const memory_allocator = nullptr);

  SessionScopeMemAllocator(const SessionScopeMemAllocator &) = delete;

  SessionScopeMemAllocator &operator=(const SessionScopeMemAllocator &) & = delete;

  virtual ~SessionScopeMemAllocator() = default;

  /// @ingroup ge_graph
  /// @brief caching allocator init
  /// @param [in] device id
  /// @return Status of init
  Status Initialize(const uint32_t device_id = 0U);

  /// @ingroup ge_graph
  /// @brief memory allocator finalize, release all memory
  /// @return void
  void Finalize(const uint32_t device_id = 0U);

  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] size memory size
  /// @param [in] session_id session id
  /// @param [in] device id
  /// @return  memory address
  uint8_t *Malloc(size_t size, const uint64_t session_id, const uint32_t device_id = 0U);

  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] session_id session id
  /// @param [in] device_id device id
  /// @return Status result of function
  Status Free(const uint64_t session_id, const uint32_t device_id = 0U);

 private:
  void FreeAllMemory();

 private:
  rtMemType_t memory_type_;

  // device memory allocator
  MemoryAllocator *memory_allocator_;

  // lock around all operations
  mutable std::recursive_mutex mutex_;

  // allocated blocks by memory pointer
  std::unordered_map<uint64_t, std::vector<SessionScopeMemoryInfo>> allocated_memory_;
};
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_SESSION_SCOPE_MEM_ALLOCATOR_H_
