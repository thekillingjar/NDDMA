/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "graph/utils/op_type_utils.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/check_register.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker_log.h"

namespace ge {
Status NoPaddingContinuousInputAndNoPaddingContinuousInputChecker(CheckFuncContext &context) {
  bool has_conflict = false;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::IsNoPaddingContinuousNodeConflict(context, has_conflict));
  if (!has_conflict) {
    GELOGI("[MemConflict] [%s, %s] not conflict",
           CheckerLog::ToStr(context.type_a[context.type_a_index]).c_str(),
           CheckerLog::ToStr(context.type_b[context.type_b_index]).c_str());
    return SUCCESS;
  }
  std::vector<OutDataAnchorPtr> out_data_anchors;
  for (const auto &node_index_io : context.all_nodes) {
    if (node_index_io.io_type_ == kOut) {
      out_data_anchors.emplace_back(node_index_io.node_->GetOutDataAnchor(node_index_io.index_));
    }
  }
  for (const auto &out_data_anchor : out_data_anchors) {
    const auto &peer_in_data_anchors = out_data_anchor->GetPeerInDataAnchors();
    size_t need_continuous_node_cnt = 0U;
    InDataAnchorPtr first_anchor = nullptr;
    for (const auto &peer_in_data_anchor : peer_in_data_anchors) {
      if (MemLayoutConflictUtil::IsNoPaddingContinuousInput(peer_in_data_anchor->GetOwnerNode())) {
        ++need_continuous_node_cnt;
        if (need_continuous_node_cnt == 1U) {
          first_anchor = peer_in_data_anchor;
        }
      }
      // 一个节点的某个输出同时给多个需要连续输入的节点，除第一个外，连续输入节点对应的in_data_anchor需要插入identity
      if (need_continuous_node_cnt > 1U) {
        context.result.insert(peer_in_data_anchor);
        GELOGI("[MemConflict][Conflict] %s connect to multi nodes that need nopadding continuous inputs,"
               " need to insert identity before %s, first: %s", CheckerLog::ToStr(out_data_anchor).c_str(),
               CheckerLog::ToStr(peer_in_data_anchor).c_str(), CheckerLog::ToStr(first_anchor).c_str());
      }
    }
  }
  return SUCCESS;
}

REGISTER_FUNC(ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT,
              NoPaddingContinuousInputAndNoPaddingContinuousInputChecker).CallAsSymbol();
}  // namespace ge
