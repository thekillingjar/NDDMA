/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph_comparer.h"

namespace gert {
bool ExeGraphComparer::GetAttr(const ge::NodePtr &const_node, ge::Buffer &buffer) {
  return ge::AttrUtils::GetZeroCopyBytes(const_node->GetOpDesc(), "value", buffer);
}
bool ExeGraphComparer::ExpectConnectTo(const ge::NodePtr &node, const std::string &peer_node_type, int32_t out_index,
                                       int32_t in_index) {
  auto out_anchor = node->GetOutDataAnchor(out_index);
  if (out_anchor == nullptr) {
    return false;
  }
  for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
    if (in_anchor->GetIdx() != in_index) {
      continue;
    }
    auto peer_node = in_anchor->GetOwnerNode();
    if (peer_node->GetType() == peer_node_type) {
      return true;
    }
  }
  return false;
}
bool ExeGraphComparer::ExpectConnectTo(const ge::NodePtr &node, const std::string &peer_node_type) {
  return ExpectConnectTo(node, peer_node_type, 0, 0);
}

bool ExeGraphComparer::ExpectConnectFrom(const ge::NodePtr &node, const std::string &from_node, int32_t out_index,
                                         int32_t in_index) {
  auto in_anchor = node->GetInDataAnchor(in_index);
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  if (out_anchor->GetIdx() != out_index) {
    return false;
  }
  auto out_node = out_anchor->GetOwnerNode();
  if (out_node == nullptr) {
    return false;
  }
  return out_node->GetType() == from_node;
}

// FastNode
bool ExeGraphComparer::GetAttr(const ge::FastNode *const_node, ge::Buffer &buffer) {
  return ge::AttrUtils::GetZeroCopyBytes(const_node->GetOpDescBarePtr(), "value", buffer);
}
bool ExeGraphComparer::ExpectConnectTo(const ge::FastNode *node, const std::string &peer_node_type, int32_t out_index,
                                       int32_t in_index) {
  for (const auto edge : node->GetOutEdgesRefByIndex(out_index)) {
    if (edge == nullptr) {
      continue;
    }
    if (edge->dst_input != in_index) {
      continue;
    }
    if (edge->dst->GetType() == peer_node_type) {
      return true;
    }
  }
  return false;
}
bool ExeGraphComparer::ExpectConnectTo(const ge::FastNode *node, const std::string &peer_node_type) {
  return ExpectConnectTo(node, peer_node_type, 0, 0);
}

bool ExeGraphComparer::ExpectConnectFrom(const ge::FastNode *node, const std::string &from_node, int32_t out_index,
                                         int32_t in_index) {
  auto edge = node->GetInDataEdgeByIndex(in_index);
  if (edge == nullptr || edge->src == nullptr) {
    return false;
  }
  if (edge->src_output != out_index) {
    return false;
  }
  return edge->src->GetType() == from_node;
}
}