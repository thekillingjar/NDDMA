/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/check_register.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
/*
 *   a       b           a                b
 *   /\      /\          /\               /\
 *  | concat1  | ==>    | identity identity |
 *  \          /        \      \  /        /
 *    concat2             \    concat1   /
 *                           \         /
 *                             concat2
 */
void ConnectToMultiContinuousInputNode(CheckFuncContext &context) {
  std::vector<OutDataAnchorPtr> out_data_anchors;
  for (const auto &node_index_io : context.all_nodes) {
    if (node_index_io.io_type_ == kOut) {
      out_data_anchors.emplace_back(node_index_io.node_->GetOutDataAnchor(node_index_io.index_));
    }
  }
  for (const auto &out_data_anchor : out_data_anchors) {
    const auto &peer_in_data_anchors = out_data_anchor->GetPeerInDataAnchors();
    size_t need_continuous_input_node_cnt = 0U;
    InDataAnchorPtr first_anchor = nullptr;
    for (const auto &peer_in_data_anchor : peer_in_data_anchors) {
      if (MemLayoutConflictUtil::IsContinuousInput(peer_in_data_anchor->GetOwnerNode())
          || MemLayoutConflictUtil::IsNoPaddingContinuousInput(peer_in_data_anchor->GetOwnerNode())) {
        ++need_continuous_input_node_cnt;
        if (need_continuous_input_node_cnt == 1U) {
          first_anchor = peer_in_data_anchor;
        }
      }
      // 一个节点的某个输出同时给多个需要连续输入的节点，除第一个外，连续输入节点对应的in_data_anchor需要插入identity
      if (need_continuous_input_node_cnt > 1U) {
        context.result.insert(peer_in_data_anchor);
        GELOGI("[MemConflict][Conflict] %s connect to multi nodes that need continuous inputs,"
               " need to insert identity before %s, first: %s", out_data_anchor->GetOwnerNodeBarePtr()->GetNamePtr(),
               peer_in_data_anchor->GetOwnerNodeBarePtr()->GetNamePtr(),
               first_anchor->GetOwnerNodeBarePtr()->GetNamePtr());
      }
    }
  }
}
/*
 *   a       b         a       b
 *   \      /          \      /
 * c  concat1  d ==> c  concat1  d
 *  \   |      /     |    |      |
 *    concat2        \  identity /
 *                    \   |     /
 *                      concat2
 */
void CascadeContinuousInputNodes(CheckFuncContext &context) {
  std::set<NodePtr> nodes_set;
  std::set<NodePtr> continuous_input_nodes;
  for (const auto &node_index_io : context.all_nodes) {
    if (nodes_set.insert(node_index_io.node_).second) {
      if (MemLayoutConflictUtil::IsContinuousInput(node_index_io.node_)
          || MemLayoutConflictUtil::IsNoPaddingContinuousInput(node_index_io.node_)) {
        continuous_input_nodes.insert(node_index_io.node_);
      }
    }
  }
  for (const auto &node : continuous_input_nodes) {
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        if (continuous_input_nodes.find(peer_in_data_anchor->GetOwnerNode()) != continuous_input_nodes.end()) {
          context.result.insert(peer_in_data_anchor);
          GELOGI("[MemConflict][Conflict] %s is connected with %s, they both need continuos inputs."
                 " need to insert identity.", node->GetNamePtr(), peer_in_data_anchor->GetOwnerNode()->GetNamePtr());
        }
      }
    }
  }
}

Status ContinuousInputAndContinuousInputChecker(CheckFuncContext &context) {
  ConnectToMultiContinuousInputNode(context);
  CascadeContinuousInputNodes(context);
  return SUCCESS;
}

REGISTER_FUNC(ANCHOR_ATTR_CONTINUOUS_INPUT, ANCHOR_ATTR_CONTINUOUS_INPUT,
              ContinuousInputAndContinuousInputChecker).CallAsSymbol();
REGISTER_FUNC(ANCHOR_ATTR_CONTINUOUS_INPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT,
              ContinuousInputAndContinuousInputChecker).CallAsSymbol();
}  // namespace ge
