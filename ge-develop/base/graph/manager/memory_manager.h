/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_COMMON_GRAPH_MANAGER_MEMORY_MANAGER_H_
#define GE_GRAPH_COMMON_GRAPH_MANAGER_MEMORY_MANAGER_H_

#include <string>
#include <sstream>
#include "ge/ge_api_types.h"
#include "runtime/mem.h"

namespace ge {
inline std::string ToMallocMemInfo(const std::string &purpose, const void *const ptr, const uint32_t device_id,
                                   const uint16_t module_id) {
  std::stringstream ss;
  ss << purpose << ", ptr=" << ptr << ", device_id=" << device_id << ", module_id=" << module_id;
  return ss.str();
}

class ExpandableMemoryAllocator {
 public:
  ExpandableMemoryAllocator() = default;
  virtual ~ExpandableMemoryAllocator() = default;

  ExpandableMemoryAllocator(const ExpandableMemoryAllocator &) = delete;
  ExpandableMemoryAllocator &operator=(const ExpandableMemoryAllocator &) & = delete;

  virtual uint8_t *MallocMemory(const std::string &purpose, int64_t memory_size, bool incremental = false) = 0;

  virtual Status FreeMemory() = 0;

  virtual bool IsSupportExpandableMemory() const = 0;

  virtual void SetReuse(bool reuse) = 0;

  virtual void SetSharePhyAllocator(bool share_phy_allocator) = 0;
};
using ExpandableMemoryAllocatorPtr = std::shared_ptr<ExpandableMemoryAllocator>;
class MemoryManager {
 public:
  MemoryManager() = default;
  virtual ~MemoryManager() = default;

  MemoryManager(const MemoryManager &) = delete;
  MemoryManager &operator=(const MemoryManager &) & = delete;

  virtual uint8_t *MallocMemory(rtMemType_t memory_type, const std::string &purpose, const std::string &memory_key,
                                size_t memory_size, uint32_t device_id) = 0;

  virtual Status FreeMemory(rtMemType_t memory_type, const std::string &memory_key, uint32_t device_id) = 0;

  virtual uint8_t *GetMemoryBase(rtMemType_t memory_type, const std::string &memory_key, uint32_t device_id) = 0;

  virtual uint8_t *GetMemoryAddr(rtMemType_t memory_type, const std::string &memory_key, uint32_t device_id) = 0;

  virtual uint8_t *MallocMemory(rtMemType_t memory_type, const std::string &purpose,
                                size_t memory_size, uint32_t device_id) = 0;

  virtual Status FreeMemory(rtMemType_t memory_type, void *const memory_addr, uint32_t device_id) = 0;

  virtual uint8_t *GetRdmaPoolMemory(rtMemType_t memory_type, size_t memory_size, uint32_t device_id) = 0;

  virtual uint8_t *GetHostPoolMemory(const rtMemType_t memory_type, const size_t memory_size) = 0;
};

}  // namespace ge

#endif  // GE_GRAPH_COMMON_GRAPH_MANAGER_MEMORY_MANAGER_H_
