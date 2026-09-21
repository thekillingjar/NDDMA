/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/graph_mem_allocator.h"
#include <string>
#include "common/math/math_util.h"
#include "graph/manager/mem_manager.h"

namespace ge {
Status MemoryAllocator::Initialize() {
  GELOGI("MemoryAllocator::Initialize.");

  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  // when redo Initialize free memory
  for (auto &it_map : deviceid_2_memory_bases_map_) {
    for (auto &it : it_map.second) {
      if (FreeMemory(it.second.memory_addr, it_map.first) != ge::SUCCESS) {
        GELOGW("Initialize: FreeMemory failed");
      }
    }
    it_map.second.clear();
  }
  deviceid_2_memory_bases_map_.clear();
  return SUCCESS;
}

void MemoryAllocator::Finalize() {
  // free memory
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &it_map : deviceid_2_memory_bases_map_) {
    for (auto &it : it_map.second) {
      if (FreeMemory(it.second.memory_addr, it_map.first) != ge::SUCCESS) {
        GELOGW("Finalize: FreeMemory failed");
      }
    }
    it_map.second.clear();
  }
  deviceid_2_memory_bases_map_.clear();
  device_to_allocator_.clear();
}

uint8_t *MemoryAllocator::MallocMemory(const std::string &purpose, const size_t memory_size,
                                       const uint32_t device_id) {
  void *memory_addr = nullptr;
  const auto allocator = GetAllocator(device_id);
  if (allocator != nullptr) {
    const auto block = allocator->Malloc(memory_size);
    if (block != nullptr) {
      memory_addr = block->GetAddr();
      if (memory_addr != nullptr) {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        mem_addr_to_block_addr_[memory_addr] = block;
      }
    }
  } else {
    if (rtMalloc(&memory_addr, memory_size, memory_type_, GE_MODULE_NAME_U16) != RT_ERROR_NONE) {
      memory_addr = nullptr;
    }
  }

  if (memory_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMalloc fail, purpose:%s, size:%zu, device_id:%u",
                      purpose.c_str(), memory_size, device_id);
    GELOGE(ge::INTERNAL_ERROR, "[Malloc][Memory] failed, device_id = %u, size= %" PRIu64,
           device_id, memory_size);

    return static_cast<uint8_t *>(memory_addr);
  }

  GELOGI("MemoryAllocator::MallocMemory device_id = %u, size= %" PRIu64 ".", device_id, memory_size);
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, ToMallocMemInfo(purpose, memory_addr, device_id, GE_MODULE_NAME_U16).c_str(),
                          memory_size);
  return static_cast<uint8_t *>(memory_addr);
}

Status MemoryAllocator::FreeMemory(void *memory_addr, const uint32_t device_id) {
  GELOGI("MemoryAllocator::FreeMemory device_id = %u.", device_id);
  const auto allocator = GetAllocator(device_id);
  if (allocator != nullptr) {
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto it = mem_addr_to_block_addr_.find(memory_addr);
    if (it != mem_addr_to_block_addr_.end()) {
      if (it->second != nullptr) {
        it->second->Free();
      }
      (void) mem_addr_to_block_addr_.erase(it);
      return ge::SUCCESS;
    } else {
      GELOGW("Can't Find block memory addr device_id = %u", device_id);
    }
  }
  GE_CHK_RT_RET(rtFree(memory_addr));
  memory_addr = nullptr;
  return ge::SUCCESS;
}

uint8_t *MemoryAllocator::MallocMemory(const std::string &purpose, const std::string &memory_key,
                                       const size_t memory_size, const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::map<string, MemoryInfo> memory_base_map;
  const auto it_map = deviceid_2_memory_bases_map_.find(device_id);
  if (it_map != deviceid_2_memory_bases_map_.end()) {
    memory_base_map = it_map->second;
    const auto it = it_map->second.find(memory_key);
    if (it != it_map->second.end()) {
      if (CheckInt32AddOverflow(it->second.memory_used_num, 1) == SUCCESS) {
        it->second.memory_used_num++;
      } else {
        return nullptr;
      }
      return it->second.memory_addr;
    }
  }
  uint8_t *const memory_addr = MallocMemory(purpose, memory_size, device_id);

  if (memory_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Malloc Memory fail, purpose:%s, memory_key:%s, memory_size:%zu, device_id:%u",
                      purpose.c_str(), memory_key.c_str(), memory_size, device_id);
    GELOGE(ge::INTERNAL_ERROR, "[Malloc][Memory] failed, memory_key[%s], size = %" PRIu64 ", device_id:%u.",
           memory_key.c_str(), memory_size, device_id);
    return nullptr;
  }

  MemoryInfo memory_info;
  memory_info.memory_addr = memory_addr;
  memory_info.memory_size = memory_size;
  memory_info.memory_used_num = 1;
  memory_info.device_id = device_id;
  memory_base_map[memory_key] = memory_info;
  deviceid_2_memory_bases_map_[device_id] = memory_base_map;
  mem_malloced_ = true;
  return memory_addr;
}

Status MemoryAllocator::FreeMemory(const std::string &memory_key, const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it_map = deviceid_2_memory_bases_map_.find(device_id);
  if (it_map == deviceid_2_memory_bases_map_.end()) {
    if (mem_malloced_) {
      GELOGW("MemoryAllocator::FreeMemory failed, device_id does not exist, device_id = %u.", device_id);
    }
    return ge::INTERNAL_ERROR;
  }
  const auto it = it_map->second.find(memory_key);
  if (it == it_map->second.end()) {
    if (mem_malloced_) {
      GELOGW("MemoryAllocator::FreeMemory failed,memory_key[%s] was does not exist, device_id = %u.", memory_key.c_str(),
             device_id);
    }
    return ge::INTERNAL_ERROR;
  }

  if (it->second.memory_used_num > 1) {
    GELOGD("MemoryAllocator::FreeMemory memory_key[%s] should not be released, reference count %d", memory_key.c_str(),
           it->second.memory_used_num);
    // reference count greater than 1 represnt that static memory is used by
    // someone else, reference count decrement
    it->second.memory_used_num--;
    return ge::SUCCESS;
  }

  if (FreeMemory(it->second.memory_addr, device_id) != ge::SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Free Memory fail, memory_key:%s, device_id:%u",
                      memory_key.c_str(), device_id);
    GELOGE(ge::INTERNAL_ERROR, "[Free][Memory] failed, memory_key[%s], device_id:%u",
           memory_key.c_str(), device_id);
    return ge::INTERNAL_ERROR;
  }

  GELOGI("MemoryAllocator::FreeMemory device_id = %u", it->second.device_id);

  (void)it_map->second.erase(it);
  return ge::SUCCESS;
}

uint8_t *MemoryAllocator::GetMemoryAddr(const std::string &memory_key, const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it_map = deviceid_2_memory_bases_map_.find(device_id);
  if (it_map == deviceid_2_memory_bases_map_.cend()) {
    GELOGW("MemoryAllocator::GetMemoryAddr failed, device_id does not exist, device_id = %u.", device_id);
    return nullptr;
  }
  const auto it = it_map->second.find(memory_key);
  if (it == it_map->second.end()) {
    GELOGW("MemoryAllocator::GetMemoryAddr failed, memory_key[%s] was does not exist, device_id = %u.", memory_key.c_str(),
           device_id);
    return nullptr;
  }

  return it->second.memory_addr;
}

void MemoryAllocator::SetAllocatorManager(gert::memory::AllocatorManager *const allocator_manager) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  allocator_manager_ = allocator_manager;
  GELOGI("SetAllocatorManager memory_type = %u.", memory_type_);
}

ge::Allocator *MemoryAllocator::GetAllocator(const uint32_t device_id) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (allocator_manager_ != nullptr) {
    const auto it = device_to_allocator_.find(device_id);
    if ((it != device_to_allocator_.end()) && (it->second != nullptr)) {
      return it->second;
    }

    const auto allocator = allocator_manager_->CreateAllocator(device_id, memory_type_);
    if (allocator != nullptr) {
      device_to_allocator_[device_id] = allocator;
      GELOGI("GetAllocator success memory_type = %u device_id =%u.", memory_type_, device_id);
      return allocator;
    }
  }
  return nullptr;
}

void MemoryAllocator::ReleaseResource(const uint32_t device_id) {
  GELOGI("ReleaseResource success memory_type = %u device_id =%u.", memory_type_, device_id);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  // free memory
  for (auto &it_map : deviceid_2_memory_bases_map_) {
    for (auto &it : it_map.second) {
      if (FreeMemory(it.second.memory_addr, it_map.first) != ge::SUCCESS) {
        GELOGW("Finalize: FreeMemory failed");
      }
    }
    it_map.second.clear();
  }
  device_to_allocator_.clear();
  mem_addr_to_block_addr_.clear();
}
}  // namespace ge
