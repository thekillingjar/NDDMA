/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_HOST_MEM_MANAGER_H_
#define GE_GRAPH_MANAGER_HOST_MEM_MANAGER_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "framework/common/util.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/tensor.h"
#include "runtime/mem.h"

namespace ge {
struct SharedMemInfo {
  std::string op_name;
  std::string shm_name;
  uint64_t mem_size;
  int32_t fd;
  uint8_t *device_address;
  std::shared_ptr<AlignedPtr> host_aligned_ptr;
  SharedMemInfo() : SharedMemInfo("", 0UL) {};
  SharedMemInfo(const std::string &name, const uint64_t size)
      : op_name(name), mem_size(size), fd(-1), device_address(nullptr), host_aligned_ptr(nullptr) {};
};
class SharedMemAllocator {
 public:
  SharedMemAllocator() = default;
  ~SharedMemAllocator() = default;

  static Status Allocate(SharedMemInfo &mem_info);
  static Status DeAllocate(const SharedMemInfo &mem_info);
};

class HostMemManager {
 public:
  HostMemManager() = default;
  ~HostMemManager() { Finalize(); }
  HostMemManager(const HostMemManager &) = delete;
  HostMemManager &operator=(const HostMemManager &) & = delete;

  static HostMemManager &Instance();
  Status Initialize();
  void Finalize() noexcept;
  Status MallocHostSharedMemory(SharedMemInfo &mem_info);
  bool QueryVarMemInfo(const std::string &op_name, SharedMemInfo &mem_info);

 private:
  static std::string OpNameToShmName(const std::string &op_name);
  std::unordered_map<std::string, SharedMemInfo> var_memory_base_map_;
  std::unique_ptr<SharedMemAllocator> allocator_;
  mutable std::recursive_mutex mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_HOST_MEM_MANAGER_H_
