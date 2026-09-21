/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/graph_mem_splitter.h"

#include <string>

#include "graph/utils/type_utils.h"
#include "common/checker.h"

namespace ge {
void GraphMemSplitter::Split(const std::map<uint64_t, size_t> &mem_offset) {
  const MemoryBlock *pre_block = nullptr;
  bool continuous_memory = false;
  bool is_fixed_addr_prior = false;
  for (const auto block : mem_blocks_) {
    if ((block == nullptr) || block->child_block_ || block->is_zero_copy_ || (block->memory_type_ != RT_MEMORY_HBM)) {
      continue;
    }

    if (pre_block == nullptr) {
      pre_block = block;
      if (block->first_continuous_block_ && (!block->last_continuous_block_)) {
        continuous_memory = true;
      }

      is_fixed_addr_prior = (is_fixed_addr_prior || block->is_fixed_addr_prior_);
      continue;
    }
    auto split_size = block->HeadOffset() - pre_block->HeadOffset();
    // 连续内存block不能被拆分，当前block是连续内存首块时，需要在这里断开并和后续连续内存放在一起
    if (((split_size < split_size_) && (!block->first_continuous_block_)) || continuous_memory) {
      if (block->last_continuous_block_) {
        continuous_memory = false;
      }

      is_fixed_addr_prior = (is_fixed_addr_prior || block->is_fixed_addr_prior_);
      continue;
    }

    // 有的block可能同时有这两个属性，不需要和后续内存连续在一起
    if (block->first_continuous_block_ && (!block->last_continuous_block_)) {
      continuous_memory = true;
    }

    int64_t mem_offset_base = static_cast<int64_t>(pre_block->HeadOffset());
    AddSubMemoryInfo({static_cast<int64_t>(pre_block->memory_type_), mem_offset_base,
                      static_cast<int64_t>(block->HeadOffset() - mem_offset_base), is_fixed_addr_prior});
    pre_block = block;
    is_fixed_addr_prior = pre_block->is_fixed_addr_prior_;
  }

  // remaining blocks
  AddMemoryInfo(mem_offset, is_fixed_addr_prior);
}

void GraphMemSplitter::AddMemoryInfo(const std::map<uint64_t, size_t> &mem_offset, bool is_fixed_addr_prior, bool io) {
  const auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.cend()) {
    if (sub_memory_infos_.empty()) {
      // reuse part size < 200M
      AddSubMemoryInfo({RT_MEMORY_HBM, 0, static_cast<int64_t>(it->second), is_fixed_addr_prior});
    } else {
      int64_t mem_offset_base = sub_memory_infos_.back().mem_offset_base + sub_memory_infos_.back().mem_size;
      AddSubMemoryInfo(
          {RT_MEMORY_HBM, mem_offset_base, static_cast<int64_t>(it->second - mem_offset_base), is_fixed_addr_prior},
          io);
    }
  }
}

void GraphMemSplitter::AddSubMemoryInfo(const SubMemoryInfo &sub_memory_info, bool io) {
  if ((sub_memory_info.mem_size > 0) || io) {
    sub_memory_infos_.emplace_back(sub_memory_info);
  }
}

void GraphMemSplitter::AddContinuousMemoryInfo(const std::map<uint64_t, size_t> &mem_offset) {
  // 理论分析只有atomic内存，不涉及hccl内存
  AddMemoryInfo(mem_offset);
}

void GraphMemSplitter::AddIoMemoryInfo(const std::map<uint64_t, size_t> &mem_offset) {
  // fix标记需要通过sub_memory_infos_携带到执行态, 小于200M时，先合并前面的分段信息
  const auto it = mem_offset.find(RT_MEMORY_HBM);
  bool is_fixed_addr_prior = false;
  int64_t mem_size = 0;
  // total size <= 200M
  if ((it != mem_offset.cend()) && (it->second <= split_size_)) {
    for (auto &mem_info : sub_memory_infos_) {
      is_fixed_addr_prior = (is_fixed_addr_prior || mem_info.is_fixed_addr_prior);
      mem_size += mem_info.mem_size;
    }
    sub_memory_infos_.clear();
    AddSubMemoryInfo({RT_MEMORY_HBM, 0, mem_size, is_fixed_addr_prior});
  }

  // 零拷贝大小为0也添加一个信息
  AddMemoryInfo(mem_offset, false, true);
}

std::vector<std::vector<int64_t>> GraphMemSplitter::GetSubMemOffsets() const {
  std::vector<std::vector<int64_t>> sub_mem_offsets;
  uint32_t count = 0U;
  for (const auto &sub_mem_info : sub_memory_infos_) {
    std::vector<int64_t> sub_mem_offset;
    sub_mem_offset.emplace_back(sub_mem_info.mem_type);
    sub_mem_offset.emplace_back(sub_mem_info.mem_offset_base);
    sub_mem_offset.emplace_back(sub_mem_info.mem_size);
    sub_mem_offset.emplace_back(sub_mem_info.is_fixed_addr_prior);
    sub_mem_offsets.emplace_back(sub_mem_offset);
    GELOGI("SubMemOffsets[%u]: mem_type[%ld] mem_offset_base[%ld] mem_size[%ld] is_fixed_addr_prior[%d]",
           count, sub_mem_info.mem_type,
           sub_mem_info.mem_offset_base, sub_mem_info.mem_size, sub_mem_info.is_fixed_addr_prior);
    count++;
  }
  return sub_mem_offsets;
}
}  // namespace ge
