/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_CONTINUOUS_MEM_H
#define GE_GRAPH_BUILD_MEMORY_CONTINUOUS_MEM_H

#include <vector>
#include "graph/utils/graph_utils.h"
#include "ge_common/ge_api_types.h"

namespace ge {
enum class ContinuousMemScenario {
  kContinuousMemScenarioOutIn, // 需要连续输出的节点连接需要连续输入的节点
  kContinuousMemScenarioOut, // 连续输入
  kContinuousMemScenarioIn // 连续输出
};

class ContinuousMem {
 public:
  explicit ContinuousMem(const ContinuousMemScenario &scenario);
  ~ContinuousMem() = default;

  std::vector<std::string> ToString() const;

  [[nodiscard]] bool IsReuse() const {
    return can_reuse_;
  }
  void SetReuseFlag(bool reuse) {
    can_reuse_ = reuse;
  }
  [[nodiscard]] bool IsUseOneBlock() const {
    return use_one_block_;
  }
  void SetUseOneBlock(bool use_one_block) {
    use_one_block_ = use_one_block;
  }
  [[nodiscard]] const std::vector<NodeIndexIO> &GetContinuousNodeOut() const {
    return continuous_node_out_;
  }
  [[nodiscard]] const std::vector<class MemoryBlock *> &GetBlocks() const {
    return blocks_;
  }
  [[nodiscard]] size_t GetTotalSize() const {
    return total_size_;
  }
  [[nodiscard]] const std::vector<size_t> &GetAlignedSizes() const {
    return aligned_sizes_;
  }
  bool IsFirstNodeOut(const Node *const node, int32_t out_index) const {
    if (continuous_node_out_.empty()) {
      return false;
    }
    const auto first_node = continuous_node_out_.front();
    return (first_node.node_ptr_ == node) && (first_node.index_ == static_cast<uint32_t>(out_index));
  }

  Status PushBackNodeOut(const Node *const node, int32_t out_index);
  void PushBackBlock(class MemoryBlock *const block);
 private:
  bool can_reuse_ = true; // 连续输出-连续输入且集中清零场景不复用，其他可以复用
  bool use_one_block_ = true; // 连续输入且集中清零场景使用多个block，连续输入单独清零场景/连续输出场景/连续输出-连续输入场景使用1个block
  std::vector<NodeIndexIO> continuous_node_out_;
  std::vector<size_t> aligned_sizes_;
  size_t total_size_ = 0U;
  std::vector<class MemoryBlock *> blocks_;
};

/*
 * 管理连续内存
 */
class ContinuousMemMng {
 public:
  ContinuousMemMng() = default;
  ~ContinuousMemMng() = default;

  // 把需要连续的节点输出按照顺序排列，并建立索引方便查询
  Status Init(const ComputeGraphPtr &graph);

  // 目前只处理需要连续输出的节点对接需要连续输入的节点，但是保留对连续输入场景，连续输出场景的扩展性
  static bool IsTargetScenario(const Node *const node, ContinuousMemScenario &scenario);

  // 判断节点node的第out_index个输出是不是在本对象的管理范围内
  bool IsFound(const Node *const node, uint32_t out_index) const;

  /*
   * 对于任意输入node, out_index，如果没有找到则返回true
   * 判断某个输出是否需要分配内存，如果IsUseOneBlock()为true，首节点返回true，其他节点返回false；
   * 如果IsUseOneBlock()为false，所有节点都返回true
   */
  bool IsNeedAssignMemory(const Node *const node, uint32_t out_index) const;

  const ContinuousMem &GetContinuousMem(const ge::Node *const node, uint32_t out_index) const;

  const std::vector<ContinuousMem> &GetAllContinuousMem() const {
    return continuous_mem_;
  }
  Status PushBackBlock(const Node *const node, uint32_t out_index, class MemoryBlock *const block);
 private:
 Status SaveNodeOutInOrder(Node *const first_node, int32_t out_index, vector_bit_t &visited);
 Status SaveOneOut(const Node *const node, int32_t out_index);
 private:
  std::vector<ContinuousMem> continuous_mem_;
  // 用来快速索引某节点的输出是否在continuous_mem_中
  std::unordered_map<const OutDataAnchor *, size_t> node_out_to_index_;
};
}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_CONTINUOUS_MEM_H
