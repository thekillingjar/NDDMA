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
#include "common/checker.h"

namespace ge {
Status ContinuousOutputAndContinuousOutputChecker(CheckFuncContext &context) {
  std::set<NodePtr> nodes_set;
  std::set<NodePtr> continuous_nodes;
  for (const auto &node_index_io : context.all_nodes) {
    if (nodes_set.insert(node_index_io.node_).second) {
      if (MemLayoutConflictUtil::IsContinuousOutput(node_index_io.node_)
          || MemLayoutConflictUtil::IsNoPaddingContinuousOutput(node_index_io.node_)) {
        continuous_nodes.insert(node_index_io.node_);
      }
    }
  }
  for (const auto &node : continuous_nodes) {
    for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
      const auto &peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_data_anchor);
        if (continuous_nodes.find(peer_out_data_anchor->GetOwnerNode()) != continuous_nodes.end()) {
          context.result.insert(peer_out_data_anchor);
          GELOGI("[MemConflict][Conflict] %s is connected with %s, they both need continuos outputs."
                 " need to insert identity.", node->GetNamePtr(), peer_out_data_anchor->GetOwnerNode()->GetNamePtr());
        }
    }
  }
  return SUCCESS;
}

REGISTER_FUNC(ANCHOR_ATTR_CONTINUOUS_OUTPUT, ANCHOR_ATTR_CONTINUOUS_OUTPUT,
              ContinuousOutputAndContinuousOutputChecker).CallAsSymbol();
REGISTER_FUNC(ANCHOR_ATTR_CONTINUOUS_OUTPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT,
              ContinuousOutputAndContinuousOutputChecker).CallAsSymbol();
}  // namespace ge
