/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_type_list.h"
#include "graph/utils/op_type_utils.h"
#include "block_mem_assigner.h"
#include "compiler/graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "common/checker.h"

namespace ge {
namespace {
constexpr int64_t kAtomicCleanAllInput = -1;
std::vector<std::string> kNodeMemAttrStrs{"data", "concentrate_atomic"};
std::string GetNodeAttrStr(const NodeMemAttr &attr) {
  if (static_cast<size_t>(attr) < kNodeMemAttrStrs.size()) {
    return kNodeMemAttrStrs.at(static_cast<size_t>(attr));
  } else {
    return "unknown";
  }
}

bool IsNextNodeCleanInput(const Node *const node, const int32_t out_index) {
  const auto judge_func = [] (const Node *const cur_node, int32_t in_index) {
    (void)in_index;
    const auto op_desc = cur_node->GetOpDescBarePtr();
    if (op_desc != nullptr) {
      // 当前仅hcom算子会设置ATOMIC_ATTR_INPUT_INDEX，且为{-1}，表示对所有输入清零
      bool need_gentask_atomic = false;
      (void)ge::AttrUtils::GetBool(cur_node->GetOpDescBarePtr(), "need_gentask_atomic", need_gentask_atomic);
      if (!need_gentask_atomic) {
        std::vector<int64_t> atomic_input_index;
        (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_INPUT_INDEX, atomic_input_index);
        if ((!atomic_input_index.empty()) && (atomic_input_index.front() == kAtomicCleanAllInput)) {
          return true;
        }
      }
    }
    return false;
  };
  const auto out_anchor = node->GetOutDataAnchor(out_index);
  if (out_anchor != nullptr) {
    for (const auto peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
      if (peer_in_anchor != nullptr) {
        const auto next_node = peer_in_anchor->GetOwnerNodeBarePtr();
        GE_ASSERT_NOTNULL(next_node);
        if (MemLayoutConflictUtil::TraverseRefChain(next_node, peer_in_anchor->GetIdx(), judge_func)) {
          return true;
        }
      }
    }
  }
  return false;
}
}
std::vector<std::vector<bool>> NodeMemAttrConflictRegister::conflict_matrix_(static_cast<size_t>(NodeMemAttr::kMaxLen),
    std::vector<bool>(static_cast<size_t>(NodeMemAttr::kMaxLen), false));
REGISTER_NODE_MEM_ATTR_CONFLICT(NodeMemAttr::kData, NodeMemAttr::kConcentrateAtomic);

void BlockTypeList::swap(ge::BlockTypeList &other) noexcept {
  std::swap(all_nodes_mem_attrs_, other.all_nodes_mem_attrs_);
}

bool BlockTypeList::IsConflictWithBlock(const BlockTypeList &other) const {
  for (const auto &attr1 : all_nodes_mem_attrs_) {
    for (const auto &attr2 : other.all_nodes_mem_attrs_) {
      if (NodeMemAttrConflictRegister::HasConflict(attr1, attr2)) {
        return true;
      }
    }
  }
  return false;
}

bool BlockTypeList::IsConflictWithOneNode(const NodeTypeIndex &node_type_index) const {
  const auto node_mem_attrs = NodeMemAttrUtils::GetNodeMemAttrs(node_type_index);
  for (const auto &attr1 : all_nodes_mem_attrs_) {
    for (const auto &attr2 : node_mem_attrs) {
      if (NodeMemAttrConflictRegister::HasConflict(attr1, attr2)) {
        return true;
      }
    }
  }
  return false;
}

void BlockTypeList::WithAdded(const NodeTypeIndex &node_type_index) {
  const auto node_mem_attrs = NodeMemAttrUtils::GetNodeMemAttrs(node_type_index);
  all_nodes_mem_attrs_.insert(node_mem_attrs.cbegin(), node_mem_attrs.cend());
}

void BlockTypeList::WithDeleted(const MemoryBlock &memory_block, const NodeTypeIndex &node_type_index) {
  if (all_nodes_mem_attrs_.empty()) {
    return;
  }
  const auto node_mem_attrs = NodeMemAttrUtils::GetNodeMemAttrs(node_type_index);
  if (node_mem_attrs.empty()) {
    return;
  }
  all_nodes_mem_attrs_.clear();
  for (const auto &block_node : memory_block.NodeTypeIndexList()) {
    WithAdded(block_node);
  }
}

std::string BlockTypeList::ToString() const {
  std::stringstream ss;
  for (const auto &attr : all_nodes_mem_attrs_) {
    ss << GetNodeAttrStr(attr) << " ";
  }
  return ss.str();
}

NodeMemAttrVector NodeMemAttrUtils::GetNodeMemAttrs(const NodeTypeIndex &node_type_index) {
  NodeMemAttrVector attrs;
  if (OpTypeUtils::IsDataNode(node_type_index.node_->GetType())) {
    attrs.emplace_back(NodeMemAttr::kData);
  }

  if ((node_type_index.mem_type_ == kOutput) && IsConcentrateAtomic(node_type_index.node_,
                                                                    static_cast<int32_t>(node_type_index.index_))) {
    attrs.emplace_back(NodeMemAttr::kConcentrateAtomic);
  }
  return attrs;
}

bool NodeMemAttrUtils::IsConcentrateAtomic(const Node *const node, const int32_t out_index) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);

  // 先判断node的输出是否需要集中清零
  bool need_gentask_atomic = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), "need_gentask_atomic", need_gentask_atomic);
  if (!need_gentask_atomic) {
    std::vector<int64_t> atomic_output_index;
    (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
    for (auto &index : atomic_output_index) {
      if (static_cast<int32_t>(index) == out_index) {
        return true;
      }
    }
  }
  // 然后再判断下一个节点的输入是否需要集中清零
  return IsNextNodeCleanInput(node, out_index);
}

std::string NodeMemAttrUtils::GetAttrStr(const NodeTypeIndex &node_type_index) {
  std::stringstream ss;
  for (const auto &attr : GetNodeMemAttrs(node_type_index)) {
    ss << GetNodeAttrStr(attr) << " ";
  }
  return ss.str();
}
} // namespace ge