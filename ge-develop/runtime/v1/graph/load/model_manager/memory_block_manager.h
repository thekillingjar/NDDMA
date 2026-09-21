/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_MEMORY_BLOCK_MANAGER_H
#define AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_MEMORY_BLOCK_MANAGER_H
#include <cstdint>
#include <vector>
#include <string>
#include "runtime/mem.h"

namespace ge {
constexpr size_t kHugePagesize = 2U * 1024U * 1024U;
constexpr size_t kSmallPagesize = 4U * 1024U;
struct RtMemBlock {
  void *addr;
  size_t size;
  size_t used_size;
};

class MemoryBlockManager {
 public:
  explicit MemoryBlockManager(const uint32_t mem_type = RT_MEMORY_HBM, const size_t block_size = kHugePagesize) :
  mem_type_(mem_type), block_size_(block_size) {}
  virtual void *Malloc(const std::string &purpose, const size_t size);
  void Release();
  virtual ~MemoryBlockManager() = default;
 private:
  void *FindFreeMem(const size_t aligned_size);
 private:
  uint32_t mem_type_;
  size_t block_size_;
  std::vector<RtMemBlock> mem_blocks_;
};
}  // namespace ge
#endif  // AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_MEMORY_BLOCK_MANAGER_H
