/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_HOST_MEM_ALLOCATOR_H_
#define GE_GRAPH_MANAGER_HOST_MEM_ALLOCATOR_H_

#include <mutex>
#include <map>

#include "framework/common/ge_inner_error_codes.h"
#include "graph_metadef/graph/aligned_ptr.h"
#include "runtime/mem.h"

namespace ge {
class HostMemAllocator {
 public:
  explicit HostMemAllocator(const rtMemType_t mem_type) { (void)mem_type; }
  ~HostMemAllocator() = default;

  HostMemAllocator(const HostMemAllocator &) = delete;
  HostMemAllocator &operator=(const HostMemAllocator &) & = delete;

  Status Initialize() {
    Clear();
    return SUCCESS;
  }
  void Finalize() { Clear(); }

  void *Malloc(const std::shared_ptr<AlignedPtr> &aligned_ptr, const size_t size);
  void *Malloc(const size_t size);
  Status Free(const void *const memory_addr);

 private:
  void Clear();

  std::map<const void *, std::pair<size_t, std::shared_ptr<AlignedPtr>>> allocated_blocks_;
  // lock around all operations
  mutable std::mutex mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_HOST_MEM_ALLOCATOR_H_
