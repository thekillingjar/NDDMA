/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/dynamic_batch_mem_assigner.h"

#include <string>

#include "graph/utils/type_utils.h"
#include "common/checker.h"

namespace ge {
bool DynamicBatchBlockReuse(MemoryBlock &block) {
  return ((block.IsSameBatchLabel()) && ((block.reuse_mem_) || (block.need_same_offset_in_batch_)));
}

struct CompareSize {
  explicit CompareSize() {}

  bool operator() (const MemoryBlock *const left, const MemoryBlock *const right) const {
    if ((left != nullptr) && (right != nullptr)) {
      auto left_size = left->Size();
      auto right_size = right->Size();
      return (left_size > right_size);
    }
    return false;
  }
};

// |--------batch 2 block1-----------------------------------||--------batch 2 block2------|
// |----batch 1 block1----------------|                       |----batch 1 block2-----|
// |---batch 0 block1---||--batch 0 block2---|
// 小batch的block合并到最大batch的block，在不超过大batch的block的size情况下可以合并多个小batch的block
void AddBlockToMaxBatchBlock(std::vector<MemoryBlock *> &max_batch_blocks, std::vector<MemoryBlock *> &blocks) {
  CompareSize cmp;
  std::sort(max_batch_blocks.begin(), max_batch_blocks.end(), cmp);
  std::sort(blocks.begin(), blocks.end(), cmp);
  for (auto max_batch_block : max_batch_blocks) {
    max_batch_block->Reset();
    for (size_t i = 0U; i < blocks.size(); ++i) {
      auto block = blocks[i];
      if (block->child_block_) {
        continue;
      }
      block->Resize();
      if (!max_batch_block->AddBatchChildBlock(block)) {
        break;
      }
    }
  }
}

size_t GetBlocksSize(std::vector<MemoryBlock *> &blocks) {
  size_t block_size = 0U;
  for (auto block : blocks) {
    block->Resize();
    block_size += block->Size();
  }
  return block_size;
}

void CreateBlocks(std::vector<MemoryBlock *> &blocks, std::vector<std::vector<MemoryBlock *>> &new_blocks,
    bool use_extend_size_memory) {
  size_t block_size = 0U;
  std::vector<MemoryBlock *> split_blocks;
  for (auto block : blocks) {
    block->Resize();
    // 内存扩展模式才需要切分，其它整个batch一块就行
    if (use_extend_size_memory && ((block_size + block->Size()) > kMaxSplitSizeForDynamicBatch)) {
      new_blocks.emplace_back(std::move(split_blocks));
      block_size = 0U;
    }
    block_size += block->Size();
    split_blocks.emplace_back(block);
  }
  new_blocks.emplace_back(std::move(split_blocks));
}

void DynamicBatchMemAssigner::ResizeDynamicBatchBlocks() {
  use_extend_size_memory_ = ge::VarManager::IsGeUseExtendSizeMemory();
  struct DynamicBatchBlocsk {
    std::map<std::string, std::vector<MemoryBlock *>> dynamic_batch_blocks;
    std::map<std::string, std::vector<MemoryBlock *>> need_same_offset_blocks;
    std::map<std::string, std::vector<MemoryBlock *>> continuous_input_blocks;
  };

  std::map<uint64_t, DynamicBatchBlocsk> memory_type_to_batch_blocks;
  for (auto block : memory_blocks_) {
    if ((block == nullptr) || (block->child_block_) || (block->is_zero_copy_)) {
      continue;
    }

    // when memory is not reusable or not fixed between batches, it can't be reused by different branch
    if (DynamicBatchBlockReuse(*block)) {
      // Some op need to allocate the same address between branches,
      // for example, hccl ops in the multi-batch training scenario size is same in different batch.
      const auto memory_type = block->memory_type_;
      if (block->need_same_offset_in_batch_) {
        memory_type_to_batch_blocks[memory_type].need_same_offset_blocks[block->batch_label_].emplace_back(block);
      } else if (block->continuous_block_) {
        memory_type_to_batch_blocks[memory_type].continuous_input_blocks[block->batch_label_].emplace_back(block);
      } else {
        memory_type_to_batch_blocks[memory_type].dynamic_batch_blocks[block->batch_label_].emplace_back(block);
      }
    }
  }

  for (auto &memory_type_batch_blocks : memory_type_to_batch_blocks) {
    GELOGI("Process memory type:%lu", memory_type_batch_blocks.first);
    DoResizeDynamicBatchBlocks(memory_type_batch_blocks.second.dynamic_batch_blocks);
    DoResizeDynamicBatchBlocks(memory_type_batch_blocks.second.need_same_offset_blocks, true);
    DoResizeDynamicBatchBlocks(memory_type_batch_blocks.second.continuous_input_blocks, true);
  }
}

MemoryBlock *DynamicBatchMemAssigner::CreateContinuousBlock(std::vector<MemoryBlock *> &continuous_blocks) {
  GE_WARN_ASSERT(!continuous_blocks.empty());
  auto continuous_block = new (std::nothrow) MemoryBlock(reuse_strategy_, 0, -1, true);
  GE_ASSERT_TRUE(continuous_block != nullptr);

  // create blocks_store_ to assure blocks deleted finally
  blocks_store_.emplace_back(continuous_block);
  memory_blocks_.emplace_back(continuous_block);

  size_t continuous_block_size = 0U;
  size_t life_begin = 0U;
  size_t life_end = 0U;
  std::string batch_label;
  int64_t parent_node_stream_id = 0;
  uint64_t memory_type = RT_MEMORY_HBM;
  const ge::Node *parent_node = nullptr;
  for (auto block : continuous_blocks) {
    if ((block != nullptr) && (!block->NodeTypeIndexList().empty())) {
      if (parent_node == nullptr) {
        parent_node = block->NodeTypeIndexList().front().node_;
        life_begin = block->GetLifeBegin();
        batch_label = block->batch_label_;
        parent_node_stream_id = block->stream_id_;
        memory_type = block->memory_type_;
      }
      block->Resize();
      continuous_block_size += block->Size();
      continuous_block->AddChildBlock(block);
      life_end = std::max(life_end, block->GetLifeEnd(block->stream_id_));
    }
  }
  GE_ASSERT_TRUE(parent_node != nullptr);
  continuous_block->SetSize(continuous_block_size);
  continuous_block->stream_id_ = parent_node_stream_id;
  continuous_block->batch_label_ = batch_label;
  continuous_block->memory_type_ = memory_type;
  continuous_block->AddNodeTypeIndex({parent_node, kWorkspace, 0, false, life_begin,
                                      parent_node_stream_id, true},
                                     continuous_block_size, continuous_block_size, parent_node_stream_id);
  continuous_block->SetLifeTimeEnd(life_end, parent_node_stream_id);
  GELOGI("%s continuous block size:%zu memory_type:%lu", continuous_block->batch_label_.c_str(),
      continuous_block->Size(), memory_type);
  return continuous_block;
}

void DynamicBatchMemAssigner::DoResizeDynamicBatchBlocks(
    std::map<std::string, std::vector<MemoryBlock *>> &dynamic_batch_blocks, bool continuous) {
  size_t max_size = 0U;
  std::string max_batch_label;
  for (auto &batch_blocks : dynamic_batch_blocks) {
    auto size = GetBlocksSize(batch_blocks.second);
    if (size > max_size) {
      max_size = size;
      max_batch_label = batch_blocks.first;
    }
  }

  auto it_max_batch = dynamic_batch_blocks.find(max_batch_label);
  if (it_max_batch == dynamic_batch_blocks.end()) {
    return;
  }
  // |--------batch 2 continuous input blocks------|
  // |-----batch 1 continuous input blocks----|
  // |---batch 0 continuous input blocks--|
  // 所有连续内存部分合成一块大内存进行对齐，小batch复用大batch内存
  if (continuous) {
    auto max_continuous_block = CreateContinuousBlock(it_max_batch->second);
    for (auto &batch_blocks : dynamic_batch_blocks) {
      if (batch_blocks.first == max_batch_label) {
        continue;
      }
      auto continuous_block = CreateContinuousBlock(batch_blocks.second);
      if ((max_continuous_block != nullptr) && (continuous_block != nullptr)) {
        max_continuous_block->Reset();
        (void) max_continuous_block->AddBatchChildBlock(continuous_block);
      }
    }
    return;
  }

  // |--------400M block1-------||--------400M block2------|
  // |-- block1---||-- block2---||-- block3---|
  // 按照切分大小创建虚拟block
  std::vector<std::vector<MemoryBlock *>> new_blocks;
  CreateBlocks(it_max_batch->second, new_blocks, use_extend_size_memory_);
  std::vector<MemoryBlock *> max_blocks;
  for (auto &blocks : new_blocks) {
    auto block = CreateContinuousBlock(blocks);
    if (block != nullptr) {
      max_blocks.emplace_back(block);
    }
  }

  // |--------batch 2 block1------||--------batch 2 block2------|
  // |----batch 1 block1------|    |----batch 1 block2-----|
  // |---batch 0 block1---|        |--batch 0 block2---|
  // 非连续内存部分，每个block可以独立对齐
  for (auto &batch_blocks : dynamic_batch_blocks) {
    if (batch_blocks.first == max_batch_label) {
      continue;
    }
    AddBlockToMaxBatchBlock(max_blocks, batch_blocks.second);
  }
}
}  // namespace ge
