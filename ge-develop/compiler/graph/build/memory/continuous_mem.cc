/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "continuous_mem.h"
#include "common/checker.h"
#include "mem_reuse_strategy.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/ge_context.h"
namespace ge {
namespace {
constexpr const char *kCleanSeparately = "1";
Status SetVisited(vector_bit_t &visited, const Node *const node) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());

  const auto id = node->GetOpDescBarePtr()->GetId();
  GE_ASSERT_TRUE(static_cast<size_t>(id) < visited.size(), "vector size: %zu, node: %s, topo_id: %lld", visited.size(),
                 node->GetNamePtr(), id);
  visited[id] = true;
  return SUCCESS;
}

bool IsVisited(const vector_bit_t &visited, const Node *const node) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
  const auto id = node->GetOpDescBarePtr()->GetId();
  GE_ASSERT_TRUE(static_cast<size_t>(id) < visited.size(), "vector size: %zu, node: %s, topo_id: %lld", visited.size(),
                 node->GetNamePtr(), id);
  return visited.at(id);
}

Status GetRealPeerContinuousInputNode(const Node *const node, const int32_t out_index, Node *&continuous_in_node,
                                      int32_t &in_index) {
  GE_CHECK_NOTNULL(node);
  std::vector<Node *> dst_nodes;
  std::vector<int32_t> dst_indexes;
  GE_ASSERT_SUCCESS(MemReuseUtils::GetDstNodeThroughRefNode(node, out_index, dst_nodes, dst_indexes),
                    "node: %s, out_index: %d", node->GetNamePtr(), out_index);
  GE_ASSERT_TRUE(dst_indexes.size() == dst_nodes.size());
  continuous_in_node = nullptr;
  in_index = 0;
  // 一个输出如果连接到多个需要连续输入的节点，前面有pass会插入identity.
  for (size_t i = 0U; i < dst_nodes.size(); ++i) {
    if (MemLayoutConflictUtil::IsContinuousInput(dst_nodes.at(i))) {
      continuous_in_node = dst_nodes.at(i);
      in_index = dst_indexes.at(i);
      break;
    }
  }
  return SUCCESS;
}

/*
 * 找到首节点，入参是需要连续输入的节点
 *
 *                      hcom2
 *                     /    \
 *  a    hcom1    refnode    \      hcom3
 *   \  /    \   /            \    /    \
 *   hcom4    hcom5            hcom6     b
 *
 *  hcom1/hcom2/hcom3是需要连续输出的节点， hcom4/hcom5/hcom6是需要连续输入的节点
 *  函数入参无论是hcom4/5/6中的哪一个，都要找到a节点，及其输出index
 *  将遍历到的需要连续输入的节点设置为visited，避免重复处理
 */
Status FindFirstNode(const Node *const node, vector_bit_t &visited, Node *&first_node,
                     int32_t &out_index) {
  GE_ASSERT_NOTNULL(node);
  auto continuous_in_node = node;
  do {
    GE_ASSERT_SUCCESS(SetVisited(visited, continuous_in_node));
    GE_ASSERT_SUCCESS(MemReuseUtils::GetSrcNodeThroughRefNode(continuous_in_node, 0, first_node, out_index),
                      "continuous_in_node: %s, index: 0", continuous_in_node->GetNamePtr());
    if ((out_index == 0) || (!MemLayoutConflictUtil::IsContinuousOutput(first_node))) {
      return SUCCESS;
    }
    out_index = 0;
    continuous_in_node = nullptr;
    std::vector<Node *> dst_nodes;
    std::vector<int32_t> in_indexes;
    GE_ASSERT_SUCCESS(MemReuseUtils::GetDstNodeThroughRefNode(first_node, out_index, dst_nodes, in_indexes),
                      "first_node: %s, out_index: %d", first_node->GetNamePtr(), out_index);
    GE_ASSERT_TRUE(dst_nodes.size() == in_indexes.size());
    for (size_t i = 0U; i < dst_nodes.size(); ++i) {
      // 如果对端是需要连续输入的节点，但是输入index已经是0了，那么也没有必要继续遍历了
      if ((in_indexes.at(i) != 0) && (!IsVisited(visited, dst_nodes.at(i))) &&
          MemLayoutConflictUtil::IsContinuousInput(dst_nodes.at(i))) {
        continuous_in_node = dst_nodes.at(i);
        break;
      }
    }
  } while (continuous_in_node != nullptr);
  return SUCCESS;
}
}

/*
 * 连续输出-连续输入且集中清零场景不复用，其他可以复用
 * 连续输入且集中清零场景使用多个block，连续输入单独清零场景/连续输出场景/连续输出-连续输入场景使用1个block
 */
ContinuousMem::ContinuousMem(const ContinuousMemScenario &scenario) {
  std::string atomic_clean_policy;
  (void)GetContext().GetOption(ge::ATOMIC_CLEAN_POLICY, atomic_clean_policy);
  if (atomic_clean_policy != kCleanSeparately) {
    if (scenario == ContinuousMemScenario::kContinuousMemScenarioOutIn) {
      can_reuse_ = false;
    } else if (scenario == ContinuousMemScenario::kContinuousMemScenarioIn) {
      use_one_block_ = false;
    }
  }
}

/*
 * [ContinuousMem] can_reuse_:1, use_one_block_:1, total size:10240, node outputs num:2
 * [ContinuousMem] 4590(HcomAllReduce)[out:0, size: 5120]->4591(Add)[in:1][out:1]->4593(HcomAllGather)[in:4]
 * [ContinuousMem] 4590(HcomAllReduce)[out:1, size: 5120]->4592(HcomAllGather)[in:4]
 */
std::vector<std::string> ContinuousMem::ToString() const {
  std::vector<std::string> ret;
  std::stringstream ss0;
  ss0 << "[ContinuousMem] can_reuse_:" << can_reuse_ << ", use_one_block_:" << use_one_block_ << ", total size:"
     << total_size_ << ", node outputs num:" << continuous_node_out_.size();
  ret.emplace_back(ss0.str());

  for (size_t i = 0U; i < continuous_node_out_.size(); ++i) {
    const auto &node_out = continuous_node_out_.at(i);
    std::stringstream ss;
    ss << "[ContinuousMem] ";
    ss << node_out.node_ptr_->GetOpDescBarePtr()->GetId() << "(" << node_out.node_ptr_->GetType() << ")[out:"
       << node_out.index_ << ", size:" << aligned_sizes_.at(i) << "] -> ";
    bool suspend = true;
    for (const auto &peer_in : node_out.node_ptr_->GetOutDataAnchor(node_out.index_)->GetPeerInDataAnchorsPtr()) {
      const auto peer_node = peer_in->GetOwnerNodeBarePtr();
      if (peer_node != nullptr) {
        ss << peer_node->GetOpDescBarePtr()->GetId() << "(" << peer_node->GetType() << ")[in: "
           << peer_in->GetIdx() << "] ";
        suspend = false;
      }
    }
    if (suspend) {
      ss << "suspended out anchor";
    }
    ret.emplace_back(ss.str());
  }
  return ret;
}

Status ContinuousMem::PushBackNodeOut(const Node *const node, int32_t out_index) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(out_index));
  GE_ASSERT_NOTNULL(tensor_desc);
  int64_t size = 0;
  GE_ASSERT_SUCCESS(MemReuseUtils::GetTensorSize(*tensor_desc, size,
      MemReuseUtils::IsNeedSplitSize(node, static_cast<uint32_t>(out_index))),
      "node: %s, out_index: %u, get tensor size failed.", node->GetNamePtr(), out_index);
  continuous_node_out_.emplace_back(NodeIndexIO{node, static_cast<uint32_t>(out_index), kOut});
  auto mem_align_size = static_cast<size_t>(size);
  MemReuseUtils::AlignMemOffset(mem_align_size);
  aligned_sizes_.emplace_back(mem_align_size);
  total_size_ += mem_align_size;
  return SUCCESS;
}

void ContinuousMem::PushBackBlock(class MemoryBlock *const block) {
  blocks_.emplace_back(block);
}

Status ContinuousMemMng::Init(const ComputeGraphPtr &graph) {
  const auto all_nodes = graph->GetAllNodesPtr();
  vector_bit_t visited(all_nodes.size(), false);
  for (const auto node : all_nodes) {
    if (IsVisited(visited, node)) {
      continue;
    }
    /*
     * 目前只处理需要连续输出的节点对接需要连续输入的节点，但是保留对连续输入场景，连续输出场景的扩展性
     */
    ContinuousMemScenario scenario;
    if (!IsTargetScenario(node, scenario)) {
      continue;
    }
    continuous_mem_.emplace_back(ContinuousMem(scenario));
    Node *first_node = nullptr;
    int32_t out_index = 0U;
    GE_ASSERT_SUCCESS(FindFirstNode(node, visited, first_node, out_index), "save node output in order failed. "
                      "find first node failed, node: %s", node->GetNamePtr());
    if (IsVisited(visited, first_node)) {
      continue;
    }
    GE_ASSERT_SUCCESS(SetVisited(visited, first_node));
    GELOGI("[ContinuousMem] find first node: %s(%s, %lld), out_index: %d", first_node->GetNamePtr(),
           first_node->GetTypePtr(), first_node->GetOpDescBarePtr()->GetId(), out_index);
    GE_ASSERT_NOTNULL(first_node);
    GE_ASSERT_SUCCESS(SaveNodeOutInOrder(first_node, out_index, visited), "save node output in order failed. node: %s,"
                      " first_node:%s out_index: %d", node->GetNamePtr(), first_node->GetNamePtr(), out_index);
    if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
      const auto logs = continuous_mem_.back().ToString();
      for (const auto &log : logs) {
        GELOGI("%s", log.c_str());
      }
    }
  }
  return SUCCESS;
}

bool ContinuousMemMng::IsTargetScenario(const Node *const node, ContinuousMemScenario &scenario) {
  GE_ASSERT_NOTNULL(node);
  if (!MemLayoutConflictUtil::IsContinuousInput(node)) {
    return false;
  }

  for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
    const auto peer_anchor = in_anchor->GetPeerOutAnchor();
    OutDataAnchor *continuous_out_anchor = nullptr;
    if ((peer_anchor != nullptr) &&
        (MemLayoutConflictUtil::IsContinuousOutputThroughRefNode(peer_anchor.get(), false, continuous_out_anchor))) {
      (void)continuous_out_anchor;
      scenario = ContinuousMemScenario::kContinuousMemScenarioOutIn;
      return true;
    }
  }
  return false;
}

/*
 * 入参first_node是首节点a
 *
 *                      hcom2
 *                     /    \
 *  a    hcom1    refnode    \      hcom3
 *   \  /    \   /            \    /    \
 *   hcom4    hcom5            hcom6     b
 *
 *  hcom1/hcom2/hcom3是需要连续输出的节点， hcom4/hcom5/hcom6是需要连续输入的节点
 *  从左往右遍历，按顺序保存起来
 *  可能FindFirstNode已经遍历了部分并了visted标记，但不一定全，这里再从左到右设置一遍
 */
Status ContinuousMemMng::SaveNodeOutInOrder(Node *const first_node, int32_t out_index, vector_bit_t &visited) {
  GE_ASSERT_NOTNULL(first_node);
  Node *cur_node = first_node;
  auto cur_index = out_index;
  while(cur_node != nullptr) {
    OutDataAnchor *last_out_anchor = nullptr;
    if (MemLayoutConflictUtil::IsContinuousOutput(cur_node)) {
      const auto all_out_anchor = cur_node->GetAllOutDataAnchorsPtr();
      for (auto i = static_cast<size_t>(cur_index); i < all_out_anchor.size(); ++i) {
        GE_ASSERT_SUCCESS(SaveOneOut(cur_node, i), "save one output failed, cur_node: %s, i: %d, first_node: %s, "
            "out_index: %d", cur_node->GetNamePtr(), i, first_node->GetNamePtr(), out_index);
      }
      last_out_anchor = all_out_anchor.back();
      GE_ASSERT_NOTNULL(last_out_anchor);
    } else {
      GE_ASSERT_SUCCESS(SaveOneOut(cur_node, cur_index), "save one output failed, cur_node: %s, cur_index: %d, "
          "first_node: %s, out_index: %d", cur_node->GetNamePtr(), cur_index, first_node->GetNamePtr(), out_index);
      last_out_anchor = cur_node->GetOutDataAnchor(cur_index).get();
      GE_ASSERT_NOTNULL(last_out_anchor, "cur_node: %s, cur_index: %d, first_node: %s,"
          " out_index: %d", cur_node->GetNamePtr(), cur_index, first_node->GetNamePtr(), out_index);
    }
    Node *continuous_in_node = nullptr;
    int32_t in_index = 0;
    GE_ASSERT_SUCCESS(GetRealPeerContinuousInputNode(cur_node, last_out_anchor->GetIdx(), continuous_in_node,
                                                     in_index));
    if (continuous_in_node == nullptr) {
      break;
    }
    GE_ASSERT_SUCCESS(SetVisited(visited, continuous_in_node));
    cur_node = nullptr;
    for (uint32_t i = static_cast<uint32_t>(in_index) + 1U; i < continuous_in_node->GetAllInDataAnchorsSize(); ++i) {
      Node *src_node = nullptr;
      int32_t src_index = 0;
      GE_ASSERT_SUCCESS(MemReuseUtils::GetSrcNodeThroughRefNode(continuous_in_node, i, src_node, src_index),
                        "continuous_in_node: %s, i: 0", continuous_in_node->GetNamePtr(), i);
      if ((i + 1U) == continuous_in_node->GetAllInDataAnchorsSize()) {
        cur_node = src_node;
        cur_index = src_index;
      } else {
        GE_ASSERT_SUCCESS(SaveOneOut(src_node, src_index), "save one output failed, src_node: %s, src_index: %d, "
            "first_node: %s, out_index: %d", src_node->GetNamePtr(), src_index, first_node->GetNamePtr(), out_index);
      }
    }
  }
  return SUCCESS;
}

Status ContinuousMemMng::SaveOneOut(const ge::Node *const node, int32_t out_index) {
  GE_ASSERT_NOTNULL(node);
  const auto &out_anchor = node->GetOutDataAnchor(out_index);
  GE_ASSERT_NOTNULL(out_anchor);
  GE_ASSERT_TRUE(node_out_to_index_.find(out_anchor.get()) == node_out_to_index_.end(),
                 "node: %s, index: %u is already saved.", node->GetNamePtr(), out_index);
  node_out_to_index_[out_anchor.get()] = continuous_mem_.size() - 1U; // continuous_mem_必不为空
  GE_ASSERT_SUCCESS(continuous_mem_.back().PushBackNodeOut(node, out_index), "node: %s, out_index: %d",
                    node->GetNamePtr(), out_index);
  return SUCCESS;
}

bool ContinuousMemMng::IsFound(const ge::Node *const node, uint32_t out_index) const {
  GE_ASSERT_NOTNULL(node);
  const auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(out_index));
  GE_ASSERT_NOTNULL(out_anchor);
  return node_out_to_index_.find(out_anchor.get()) != node_out_to_index_.end();
}

bool ContinuousMemMng::IsNeedAssignMemory(const ge::Node *const node, uint32_t out_index) const {
  GE_ASSERT_NOTNULL(node);
  const auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(out_index));
  GE_ASSERT_NOTNULL(out_anchor);
  const auto iter = node_out_to_index_.find(out_anchor.get());
  if ((iter == node_out_to_index_.end()) || (iter->second >= continuous_mem_.size())) {
    return true;
  }
  const auto continuous_mem = continuous_mem_.at(iter->second);
  if (continuous_mem.IsFirstNodeOut(node, out_index)) {
    return true;
  }
  return !continuous_mem.IsUseOneBlock();
}

const ContinuousMem &ContinuousMemMng::GetContinuousMem(const ge::Node *const node, uint32_t out_index) const {
  static ContinuousMem dummy_mem(ContinuousMemScenario::kContinuousMemScenarioOutIn);
  if (node == nullptr) {
    return dummy_mem;
  }
  const auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(out_index));
  const auto iter = node_out_to_index_.find(out_anchor.get());
  if ((iter == node_out_to_index_.end()) || (iter->second >= continuous_mem_.size())) {
    return dummy_mem;
  }
  return continuous_mem_.at(iter->second);
}

Status ContinuousMemMng::PushBackBlock(const ge::Node *const node, uint32_t out_index,
                                       class ge::MemoryBlock *const block) {
  const auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(out_index));
  GE_ASSERT_NOTNULL(out_anchor);
  const auto iter = node_out_to_index_.find(out_anchor.get());
  GE_ASSERT_TRUE((iter != node_out_to_index_.end()) && (iter->second < continuous_mem_.size()));
  continuous_mem_[iter->second].PushBackBlock(block);
  return SUCCESS;
}
}  // namespace ge