/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/mem_manager.h"

#include <string>

namespace ge {
MemManager::MemManager() : MemoryManager() {}

MemManager::~MemManager() { Finalize(); }

MemManager &MemManager::Instance() {
  static MemManager mem_manager;
  return mem_manager;
}

Status MemManager::Initialize(const std::vector<rtMemType_t> &memory_type) {
  const std::lock_guard<std::recursive_mutex> lock(allocator_mutex_);
  if (init_) {
    GELOGW("MemManager has been inited.");
    return SUCCESS;
  }

  GELOGI("MemManager Initialize.");
  if (allocator_manager_ != nullptr) {
    (void) allocator_manager_->Initialize(memory_type);
  }

  auto ret = InitAllocator(memory_type, memory_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create MemoryAllocator failed.");
    return ret;
  }

  ret = InitAllocator(memory_type, caching_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create CachingAllocator failed.");
    return ret;
  }

  ret = InitAllocator(memory_type, rdma_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create RdmaAllocator failed.");
    return ret;
  }

  ret = InitAllocator(memory_type, host_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create HostMemAllocator failed.");
    return ret;
  }

  ret = InitAllocator(memory_type, session_scope_allocator_map_);
  if (ret != SUCCESS) {
    GELOGE(ret, "Create HostMemAllocator failed.");
    return ret;
  }
  init_ = true;
  memory_type_ = memory_type;
  return SUCCESS;
}

uint8_t *MemManager::MallocMemory(const rtMemType_t memory_type, const std::string &purpose,
                                  const std::string &memory_key, const size_t memory_size, const uint32_t device_id) {
  return MemInstance(memory_type).MallocMemory(purpose, memory_key, memory_size, device_id);
}

Status MemManager::FreeMemory(const rtMemType_t memory_type, const std::string &memory_key, const uint32_t device_id) {
  return MemInstance(memory_type).FreeMemory(memory_key, device_id);
}

uint8_t *MemManager::GetMemoryBase(const rtMemType_t memory_type, const std::string &memory_key,
                                   const uint32_t device_id) {
  if (memory_type == RT_MEMORY_RDMA_HBM) {
    return RdmaPoolInstance(RT_MEMORY_HBM).GetRdmaBaseAddr();
  }

  return MemInstance(memory_type).GetMemoryAddr(memory_key, device_id);
}

uint8_t *MemManager::GetMemoryAddr(const rtMemType_t memory_type, const std::string &memory_key,
                                   const uint32_t device_id) {
  return MemInstance(memory_type).GetMemoryAddr(memory_key, device_id);
}

uint8_t *MemManager::MallocMemory(const rtMemType_t memory_type, const std::string &purpose,
                                  const size_t memory_size, const uint32_t device_id) {
  return MemInstance(memory_type).MallocMemory(purpose, memory_size, device_id);
}

Status MemManager::FreeMemory(const rtMemType_t memory_type, void *const memory_addr, const uint32_t device_id) {
  return MemInstance(memory_type).FreeMemory(memory_addr, device_id);
}

uint8_t *MemManager::GetRdmaPoolMemory(const rtMemType_t memory_type, const size_t memory_size,
                                       const uint32_t device_id) {
  return RdmaPoolInstance(memory_type).Malloc(memory_size, device_id);
}

uint8_t *MemManager::GetHostPoolMemory(const rtMemType_t memory_type, const size_t memory_size) {
  return static_cast<uint8_t *>(HostMemInstance(memory_type).Malloc(memory_size));
}

void MemManager::FreeSessionMemory(const uint64_t session_id, const uint32_t device_id) {
  for (const auto &memory_type : memory_type_) {
    (void)SessionScopeMemInstance(memory_type).Free(session_id, device_id);
  }
}

void MemManager::Finalize() noexcept {
  GELOGI("MemManager Finalize.");
  const std::lock_guard<std::recursive_mutex> lock(allocator_mutex_);
  if (!init_) {
    return;
  }
  // caching and rdma allocator use memory allocator, so finalize them first
  FinalizeAllocatorMap(session_scope_allocator_map_);
  FinalizeAllocatorMap(caching_allocator_map_);
  FinalizeAllocatorMap(rdma_allocator_map_);
  FinalizeAllocatorMap(host_allocator_map_);
  FinalizeAllocatorMap(memory_allocator_map_);
  init_ = false;
  memory_type_.clear();
  ReleaseResource();
  if (allocator_manager_ != nullptr) {
    allocator_manager_->Finalize();
  }
}

Status MemManager::RegisterAllocatorManager(gert::memory::AllocatorManager *const allocator_manager) {
  GELOGI("MemManager RegisterAllocatorManager.");
  const std::lock_guard<std::recursive_mutex> lock(allocator_mutex_);
  allocator_manager_ = allocator_manager;
  return ge::SUCCESS;
};

void MemManager::ReleaseResource(const uint32_t device_id) {
  GELOGI("MemManager ReleaseResource device id = %u.", device_id);
  const std::lock_guard<std::recursive_mutex> lock(allocator_mutex_);
  for (auto &it : memory_allocator_map_) {
    if (it.second != nullptr) {
      it.second->ReleaseResource(device_id);
    }
  }
  if (allocator_manager_ != nullptr) {
    allocator_manager_->ReleaseResource(device_id);
  }
}

MemoryAllocator &MemManager::MemInstance(const rtMemType_t memory_type) {
  return GetAllocator(memory_type, memory_allocator_map_);
}

CachingAllocator &MemManager::CachingInstance(const rtMemType_t memory_type) {
  return GetAllocator(memory_type, caching_allocator_map_);
}

RdmaPoolAllocator &MemManager::RdmaPoolInstance(const rtMemType_t memory_type) {
  return GetAllocator(memory_type, rdma_allocator_map_);
}

HostMemAllocator &MemManager::HostMemInstance(const rtMemType_t memory_type) {
  return GetAllocator(memory_type, host_allocator_map_);
}

SessionScopeMemAllocator &MemManager::SessionScopeMemInstance(const rtMemType_t memory_type) {
  return GetAllocator(memory_type, session_scope_allocator_map_);
}
}  // namespace ge
