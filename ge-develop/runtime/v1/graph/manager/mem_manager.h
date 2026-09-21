/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_GRAPH_MEM_MANAGER_H_
#define GE_GRAPH_MANAGER_GRAPH_MEM_MANAGER_H_

#include <iostream>
#include <map>
#include <mutex>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/caching_allocator.h"
#include "graph/manager/host_mem_allocator.h"
#include "graph/manager/rdma_pool_allocator.h"
#include "graph/manager/host_mem_allocator.h"
#include "graph/manager/session_scope_mem_allocator.h"
#include "graph/node.h"
#include "runtime/mem.h"
#include "graph/manager/memory_manager.h"
#include "runtime/mem_allocator.h"
#include "base/err_msg.h"

namespace ge {
class MemManager : public MemoryManager {
 public:
  MemManager();
  ~MemManager() override;
  static MemManager &Instance();

  uint8_t *MallocMemory(const rtMemType_t memory_type,
                        const std::string &purpose,
                        const std::string &memory_key,
                        const size_t memory_size,
                        const uint32_t device_id) override;

  Status FreeMemory(const rtMemType_t memory_type, const std::string &memory_key,
		            const uint32_t device_id) override;

  uint8_t *GetMemoryBase(const rtMemType_t memory_type, const std::string &memory_key,
		                 const uint32_t device_id) override;

  uint8_t *GetMemoryAddr(const rtMemType_t memory_type, const std::string &memory_key,
		                 const uint32_t device_id) override;

  uint8_t *MallocMemory(const rtMemType_t memory_type, const std::string &purpose,
                        const size_t memory_size, const uint32_t device_id) override;

  Status FreeMemory(const rtMemType_t memory_type, void *const memory_addr, const uint32_t device_id) override;

  uint8_t *GetRdmaPoolMemory(const rtMemType_t memory_type, const size_t memory_size,
		                     const uint32_t device_id) override;
  uint8_t *GetHostPoolMemory(const rtMemType_t memory_type, const size_t memory_size) override;

  void FreeSessionMemory(const uint64_t session_id, const uint32_t device_id = 0U);

  MemoryAllocator &MemInstance(const rtMemType_t memory_type);
  CachingAllocator &CachingInstance(const rtMemType_t memory_type);
  RdmaPoolAllocator &RdmaPoolInstance(const rtMemType_t memory_type);
  HostMemAllocator &HostMemInstance(const rtMemType_t memory_type = RT_MEMORY_HBM);
  SessionScopeMemAllocator &SessionScopeMemInstance(const rtMemType_t memory_type);

  /// @ingroup ge_graph
  /// @brief memory allocator manager init
  /// @param [in] options user config params
  /// @return Status result of function
  Status Initialize(const std::vector<rtMemType_t> &memory_type);

  /// @ingroup ge_graph
  /// @brief memory allocator finalize
  /// @return void
  void Finalize() noexcept;

  Status RegisterAllocatorManager(gert::memory::AllocatorManager *const allocator_manager);

  void ReleaseResource(const uint32_t device_id = 0U);

 private:
  MemManager(const MemManager &) = delete;
  MemManager &operator=(const MemManager &) & = delete;

  /// @ingroup ge_graph
  /// @param [in] memory_type memory type
  /// @param [in] allocate_map memory allocator map
  /// @return Status result of function
  template <typename T>
  static Status InitAllocator(const std::vector<rtMemType_t> &memory_type, std::map<rtMemType_t, T *> &allocate_map) {
    T *allocator = nullptr;
    for (const rtMemType_t index : memory_type) {
      const auto it = allocate_map.find(index);
      if (it == allocate_map.end()) {
        allocator = new (std::nothrow) T(index);
        if (allocator != nullptr) {
          allocate_map[index] = allocator;
          GELOGI("Create Allocator memory type[%u] success.", index);
        } else {
          REPORT_INNER_ERR_MSG("E19999", "New MemoryAllocator fail, index:%u", index);
          GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Alloc Allocator failed.");
        }
      } else {
        allocator = it->second;
      }

      if (allocator == nullptr) {
        GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Create Allocator failed.");
        return ACL_ERROR_GE_MEMORY_ALLOCATION;
      } else {
        if (allocator->Initialize() != SUCCESS) {
          return ACL_ERROR_GE_INTERNAL_ERROR;
        }
      }
    }
    return SUCCESS;
  }

  /// @ingroup ge_graph
  /// @param [in] memory_type memory type
  /// @param [in] allocate_map memory allocator map
  /// @return Allocator ptr
  template <typename T>
  auto GetAllocator(const rtMemType_t memory_type, const std::map<rtMemType_t, T *> allocate_map) -> T& {
    const std::lock_guard<std::recursive_mutex> lock(allocator_mutex_);
    T *allocator = nullptr;
    const auto it = allocate_map.find(memory_type);
    if (it != allocate_map.end()) {
      allocator = it->second;
    }

    // Usually impossible
    if (allocator == nullptr) {
      GELOGW("can not get allocator, return default allocator, memory type is %u.", memory_type);
      static T default_allocator(RT_MEMORY_RESERVED);
      return default_allocator;
    }
    return *allocator;
  }

  template <typename T>
  void FinalizeAllocatorMap(std::map<rtMemType_t, T *> &allocate_map) const {
    for (auto &allocator : allocate_map) {
      if (allocator.second != nullptr) {
        allocator.second->Finalize();
        delete allocator.second;
        allocator.second = nullptr;
      }
    }
    allocate_map.clear();
  }

  std::map<rtMemType_t, MemoryAllocator *> memory_allocator_map_;
  std::map<rtMemType_t, CachingAllocator *> caching_allocator_map_;
  std::map<rtMemType_t, RdmaPoolAllocator *> rdma_allocator_map_;
  std::map<rtMemType_t, HostMemAllocator *> host_allocator_map_;
  std::map<rtMemType_t, SessionScopeMemAllocator *> session_scope_allocator_map_;
  std::recursive_mutex allocator_mutex_;
  std::vector<rtMemType_t> memory_type_;
  gert::memory::AllocatorManager *allocator_manager_ = nullptr;
  bool init_ = false;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_
