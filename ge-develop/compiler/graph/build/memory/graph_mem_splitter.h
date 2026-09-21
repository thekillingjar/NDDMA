/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_SPLITTER_H_
#define GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_SPLITTER_H_

#include <map>
#include <vector>

#include "runtime/rt.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/build/memory/block_mem_assigner.h"

namespace ge {
constexpr size_t kSubMemoryMaximumSize = 200UL * 1024UL * 1024UL;

struct SubMemoryInfo {
  int64_t mem_type;
  int64_t mem_offset_base;
  int64_t mem_size;
  bool is_fixed_addr_prior;
};

class GraphMemSplitter {
 public:
  explicit GraphMemSplitter(std::vector<MemoryBlock *> mem_blocks, size_t split_size = kSubMemoryMaximumSize)
      : mem_blocks_(std::move(mem_blocks)), split_size_(split_size) {}

  GraphMemSplitter(const MemoryBlock &) = delete;

  GraphMemSplitter &operator=(const MemoryBlock &) = delete;

  ~GraphMemSplitter() {}

  void Split(const std::map<uint64_t, size_t> &mem_offset);

  void AddSubMemoryInfo(const SubMemoryInfo& sub_memory_info, bool io = false);

  void AddContinuousMemoryInfo(const std::map<uint64_t, size_t> &mem_offset);

  void AddIoMemoryInfo(const std::map<uint64_t, size_t> &mem_offset);

  void AddMemoryInfo(const std::map<uint64_t, size_t> &mem_offset, bool is_fixed_addr_prior = false, bool io = false);

  const std::vector<SubMemoryInfo> &GetSubMemInfo() const {
    return sub_memory_infos_;
  }
  std::vector<std::vector<int64_t>> GetSubMemOffsets() const;

 private:
  std::vector<MemoryBlock *> mem_blocks_;
  std::vector<SubMemoryInfo> sub_memory_infos_;
  size_t split_size_;
};

}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_SPLITTER_H_
