/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/fusion_common/fusion_turbo_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/anchor.h"
#include "graph/utils/node_utils.h"

namespace fe {
const std::array<size_t, static_cast<size_t>(ge::DT_MAX + 1)> data_type_size = {
    4, // DT_FLOAT = 0,
    2, // DT_FLOAT16 = 1,
    1, // DT_INT8 = 2,
    4, // DT_INT32 = 3,
    1, // DT_UINT8 = 4,
    1, // DT_xxxx = 5,
    2, // DT_INT16 = 6,
    2, // DT_UINT16 = 7,
    4, // DT_UINT32 = 8,
    8, // DT_INT64 = 9,
    8, // DT_UINT64 = 10,
    8, // DT_DOUBLE = 11,
    1, // DT_BOOL = 12,
    8, // DT_STRING = 13,
    1, // DT_DUAL_SUB_INT8 = 14,
    1, // DT_DUAL_SUB_UINT8 = 15,
    8, // DT_COMPLEX64 = 16,
    16, // DT_COMPLEX128 = 17,
    1, // DT_QINT8 = 18,
    2, // DT_QINT16 = 19,
    4, // DT_QINT32 = 20,
    1, // DT_QUINT8 = 21,
    2, // DT_QUINT16 = 22,
    1, // DT_RESOURCE = 23,
    1, // DT_STRING_REF = 24,
    1, // DT_DUAL = 25,
    1, // DT_VARIANT = 26,
    2, // DT_BF16 = 27,
    1, // DT_UNDEFINED = 28,
    1, // DT_INT4 = 29,
    1, // DT_UINT1 = 30,
    1, // DT_INT2 = 31,
    4, // DT_UINT2 = 32,
    4, // DT_COMPLEX32 = 33,
    0, // DT_MAX = 34
};

ge::NodePtr FusionTurboUtils::GetConstInput(const ge::NodePtr &node, int32_t index) {
  ge::NodePtr ret = nullptr;

  auto in_anchor = node->GetInDataAnchor(index);

  FUSION_TURBO_NOTNULL(in_anchor, nullptr);
  const auto out_anchor = in_anchor->GetPeerOutAnchor();
  if (out_anchor == nullptr) {
    return nullptr;
  }

  const auto in_node = out_anchor->GetOwnerNode();
  if (in_node->GetType() == ge::CONSTANT) {
    ret = in_node;
  } else if (in_node->GetType() == ge::DATA) {
    const auto parent = ge::NodeUtils::GetParentInput(in_node);
    if ((parent != nullptr) && (parent->GetType() == ge::CONSTANT)) {
      ret = parent;
    }
  } else {
    // do nothing
  }

  return ret;
}

NodeIndex FusionTurboUtils::GetPeerOutPair(const ge::NodePtr &node, int32_t index) {
  NodeIndex ret;
  FUSION_TURBO_NOTNULL(node, ret);
  if (static_cast<uint32_t>(index) >= node->GetAllInDataAnchorsSize()) {
    return ret;
  }
  auto input_anchor = node->GetInDataAnchor(index);
  if (input_anchor == nullptr) {
    return ret;
  }
  auto peer_anchor = input_anchor->GetPeerOutAnchor();
  if (peer_anchor == nullptr) {
    return ret;
  }

  auto peer_anchor_index = peer_anchor->GetIdx();
  auto actual_node = peer_anchor->GetOwnerNode();
  ret.node = actual_node;
  ret.index = peer_anchor_index;
  return ret;
}

void Relations::AppendPeerInAllPairs(ThisIndex relation_index, const ge::NodePtr &node, int32_t index) {
  if (static_cast<uint32_t>(index) >= node->GetAllOutDataAnchorsSize()) {
    return;
  }
  auto output_anchor = node->GetOutDataAnchor(index);
  if (output_anchor == nullptr) {
    return;
  }

  auto peer_anchors = output_anchor->GetPeerInDataAnchors();
  if (peer_anchors.empty()) {
    return;
  }

  for (const auto &ele : peer_anchors) {
    NodeIndex temp(ele->GetOwnerNode(), ele->GetIdx());
    (void)out_relations[relation_index].emplace_back(temp);
  }
}

NodeIndex FusionTurboUtils::GetPeerInFirstPair(const ge::NodePtr &node, int32_t index) {
  NodeIndex ret;
  if (static_cast<uint32_t>(index) >= node->GetAllOutDataAnchorsSize()) {
    return ret;
  }

  auto output_anchor = node->GetOutDataAnchor(index);
  if (output_anchor == nullptr) {
    return ret;
  }
  auto peer_anchors = output_anchor->GetPeerInDataAnchors();
  if (peer_anchors.empty()) {
    return ret;
  }

  ret.index = peer_anchors.at(0)->GetIdx();
  ret.node = peer_anchors.at(0)->GetOwnerNode();
  return ret;
}

void Relations::PreProcessNodeIndices(ThisIndex index, const NodeIndices &node_indices) {
  for (const auto &node_index : node_indices) {
    PreProcessOneNodeIndex(index, node_index);
  }
}

void Relations::PreProcessOneNodeIndex(ThisIndex index, const NodeIndex &node_index) {
  if (node_index.node == nullptr) {
    return;
  }

  if (node_index.direction != PEER && node_index.direction != PEER_SINGLE) {
    (void)in_relations[index].emplace_back(node_index);
    (void)out_relations[index].emplace_back(node_index);
  } else {
    /* Update input's peer nodes */
    auto peer_out = FusionTurboUtils::GetPeerOutPair(node_index.node, node_index.index);
    if (peer_out.node != nullptr) {
      (void)in_relations[index].emplace_back(peer_out);
    } else {
      GELOGD("Peer input for %s %u is nullptr", node_index.node->GetName().c_str(), node_index.index);
    }

    /* Update output's peer nodes */
    if (node_index.direction == PEER) {
      AppendPeerInAllPairs(index, node_index.node, node_index.index);
    } else if (node_index.direction == PEER_SINGLE) {
      auto peer_in = FusionTurboUtils::GetPeerInFirstPair(node_index.node, node_index.index);
      if (peer_in.node == nullptr) {
        GELOGD("Peer output for %s %u is nullptr", node_index.node->GetName().c_str(), node_index.index);
      } else {
        (void)out_relations[index].emplace_back(peer_in);
      }
    }
  }
}

void Relations::PreProcess() {
  in_relations.clear();
  out_relations.clear();
  for (auto &relation : ori_relations) {
    for (auto &pair : relation.second) {
      PreProcessOneNodeIndex(relation.first, pair);
    }
  }
}

Relations::Relations() {}

Relations::Relations(const std::initializer_list<NodeIndex> &peer_indices) {
  int32_t index = 0;
  for (const auto &node_index : peer_indices) {
    NodeIndices temp = {node_index};
    std::ignore = ori_relations.emplace(std::make_pair(index, temp));
  }
  PreProcess();
}

Relations::Relations(const std::map<ThisIndex, NodeIndices> &relations_param) {
  ori_relations = relations_param;
  PreProcess();
}

Relations::Relations(std::map<ThisIndex, NodeIndices> &&relations_param) {
  ori_relations = std::move(relations_param);
  PreProcess();
}

Relations::Relations(const Relations &relations_param) {
  ori_relations = relations_param.ori_relations;
  in_relations = relations_param.in_relations;
  out_relations = relations_param.out_relations;
}

Relations::Relations(Relations &&relations_param) noexcept {
  ori_relations = std::move(relations_param.ori_relations);
  in_relations = std::move(relations_param.in_relations);
  out_relations = std::move(relations_param.out_relations);
}

Relations& Relations::operator=(const Relations &relations_param) {
  ori_relations = relations_param.ori_relations;
  in_relations = relations_param.in_relations;
  out_relations = relations_param.out_relations;
  return *this;
}

Relations& Relations::operator=(Relations &&relations_param) noexcept {
  ori_relations = std::move(relations_param.ori_relations);
  in_relations = std::move(relations_param.in_relations);
  out_relations = std::move(relations_param.out_relations);
  return *this;
}

Relations::Relations(ThisIndex this_index, const NodeIndex &peer_index) {
  std::ignore = Add(this_index, peer_index);
}

Relations::Relations(ThisIndex this_index, const NodeIndices &peer_indices) {
  std::ignore = Add(this_index, peer_indices);
}

Relations::Relations(ThisIndex this_index, NodeIndex &&peer_index) {
  std::ignore = Add(this_index, std::move(peer_index));
}

Relations::Relations(ThisIndex this_index, NodeIndices &&peer_indices) {
  std::ignore = Add(this_index, std::move(peer_indices));
}

Relations::Relations(
    const std::initializer_list<std::pair<ThisIndex, NodeIndex>> &peer_indices) {
  for (const auto &index_pair: peer_indices) {
    std::ignore = Add(index_pair.first, index_pair.second);
  }
}

Relations::Relations(
    const std::initializer_list<std::pair<ThisIndex, std::initializer_list<NodeIndex>>> &peer_indices_vec) {
  for (const auto &peer_indices: peer_indices_vec) {
    std::ignore = Add(peer_indices.first, peer_indices.second);
  }
}

Relations& Relations::Add(ThisIndex this_index, const NodeIndex &peer_index) {
  const auto iter = ori_relations.find(this_index);
  if (iter == ori_relations.end()) {
    NodeIndices temp = {peer_index};
    std::ignore = ori_relations.emplace(std::make_pair(this_index, temp));
  } else {
    (void)iter->second.emplace_back(peer_index);
  }
  PreProcessOneNodeIndex(this_index, peer_index);
  return *this;
}

Relations& Relations::Add(ThisIndex this_index, NodeIndex &&peer_index) {
  PreProcessOneNodeIndex(this_index, peer_index);
  const auto iter = ori_relations.find(this_index);
  if (iter == ori_relations.end()) {
    NodeIndices temp = {std::move(peer_index)};
    std::ignore = ori_relations.emplace(std::make_pair(this_index, std::move(temp)));
  } else {
    (void)iter->second.emplace_back(std::move(peer_index));
  }
  return *this;
}

Relations& Relations::Add(ThisIndex this_index, const std::initializer_list<NodeIndex> &peer_indices) {
  const auto iter = ori_relations.find(this_index);
  if (iter == ori_relations.end()) {
    std::ignore = ori_relations.emplace(std::make_pair(this_index, peer_indices));
  } else {
    for (const auto &peer_index : peer_indices) {
      (void)iter->second.emplace_back(peer_index);
    }
  }
  PreProcessNodeIndices(this_index, peer_indices);
  return *this;
}

Relations& Relations::Add(ThisIndex this_index, const NodeIndices &peer_indices) {
  const auto iter = ori_relations.find(this_index);
  if (iter == ori_relations.end()) {
    std::ignore = ori_relations.emplace(std::make_pair(this_index, peer_indices));
  } else {
    for (const auto &peer_index : peer_indices) {
      (void)iter->second.emplace_back(peer_index);
    }
  }
  PreProcessNodeIndices(this_index, peer_indices);
  return *this;
}

Relations& Relations::Add(ThisIndex this_index, NodeIndices &&peer_indices) {
  PreProcessNodeIndices(this_index, peer_indices);
  const auto iter = ori_relations.find(this_index);
  if (iter == ori_relations.end()) {
    std::ignore = ori_relations.emplace(std::make_pair(this_index, std::move(peer_indices)));
  } else {
    for (auto &&peer_index : peer_indices) {
      (void)iter->second.emplace_back(std::move(peer_index));
    }
  }
  return *this;
}

Relations& Relations::UpdatePeerIndex(ThisIndex this_index, NodeIndices &&peer_indices) {
  ori_relations[this_index] = std::move(peer_indices);
  PreProcess();
  return *this;
}

Relations& Relations::UpdatePeerIndex(ThisIndex this_index, const NodeIndices &peer_indices) {
  ori_relations[this_index] = peer_indices;
  PreProcess();
  return *this;
}

Relations& Relations::UpdatePeerIndex(const std::map<ThisIndex, NodeIndices> &peer_indices) {
  ori_relations = peer_indices;
  PreProcess();
  return *this;
}

Relations& Relations::UpdatePeerIndex(std::map<ThisIndex, NodeIndices> &&peer_indices) {
  ori_relations = std::move(peer_indices);
  PreProcess();
  return *this;
}

const std::map<ThisIndex, NodeIndices>& Relations::GetRelations() {
  return ori_relations;
}

const std::map<ThisIndex, NodeIndices>& Relations::GetInRelations() {
  return in_relations;
}

const std::map<ThisIndex, NodeIndices>& Relations::GetOutRelations() {
  return out_relations;
}
}
