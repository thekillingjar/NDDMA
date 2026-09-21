/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_

#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "framework/common/debug/ge_log.h"
#include "graph/node.h"
#include "runtime/mem.h"
#include "runtime/mem_allocator.h"

namespace ge {
struct MemoryInfo {
  uint8_t *memory_addr{nullptr};
  uint64_t memory_size{0U};
  int32_t memory_used_num{0};
  uint32_t device_id{0U};
};

class MemoryAllocator {
 public:
  explicit MemoryAllocator(const rtMemType_t memory_type, const bool mem_malloced = false)
      : memory_type_(memory_type), mem_malloced_(mem_malloced) {}

  virtual ~MemoryAllocator() = default;

  /// @ingroup ge_graph
  /// @brief memory allocator init
  /// @param [in] options user config params
  /// @return Status of init
  Status Initialize();

  /// @ingroup ge_graph
  /// @brief memory allocator finalize
  /// @return void
  void Finalize();

  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] purpose memory usage
  /// @param [in] size memory size
  /// @param [in] device_id device id
  /// @return  memory address
  uint8_t *MallocMemory(const std::string &purpose, const size_t memory_size, const uint32_t device_id = 0U);

  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] device_id device id
  /// @param [out] memory_ptr memory address ptr
  /// @return Status result of function
  Status FreeMemory(void *memory_addr, const uint32_t device_id = 0U);

  /// @ingroup ge_graph
  /// @brief malloc memory
  /// @param [in] purpose memory usage
  /// @param [in] memory_key memory key
  /// @param [in] size memory size
  /// @param [in] device_id device id
  /// @return memory address
  uint8_t *MallocMemory(const std::string &purpose, const std::string &memory_key, const size_t memory_size,
                        const uint32_t device_id = 0U);

  /// @ingroup ge_graph
  /// @brief free memory
  /// @param [in] memory_key memory key
  /// @param [in] device_id device id
  /// @return Status result of function
  Status FreeMemory(const std::string &memory_key, const uint32_t device_id = 0U);

  /// @ingroup ge_graph
  /// @brief get memory address
  /// @param [in] memory_key memory key
  /// @param [in] device_id device id
  /// @return memory address (must not free memory by it)
  uint8_t *GetMemoryAddr(const std::string &memory_key, const uint32_t device_id = 0U);

  void SetAllocatorManager(gert::memory::AllocatorManager *const allocator_manager);

  ge::Allocator *GetAllocator(const uint32_t device_id);

  void ReleaseResource(const uint32_t device_id = 0U);

 private:
  rtMemType_t memory_type_;
  bool mem_malloced_;
  gert::memory::AllocatorManager *allocator_manager_ = nullptr;
  std::map<uint32_t, map<std::string, MemoryInfo>> deviceid_2_memory_bases_map_;
  std::map<void *, ge::MemBlock *> mem_addr_to_block_addr_;
  std::map<uint32_t, ge::Allocator *> device_to_allocator_;
  std::recursive_mutex mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MEM_ALLOCATOR_H_
