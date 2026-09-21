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
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker_log.h"
#include "graph/utils/op_type_utils.h"
#include "common/checker.h"

namespace ge {
/*
 * case 1: need insert identity
 *   a
 *   |
 *   +--------+  b
 *   |         \ /
 * netoutput    c (need (NoPadding)Continuous input)
 *
 * case 2: do not insert identity
 * a   b
 *  \ /
 *   c (need (NoPadding)Continuous input, and output reference input)
 *   |
 * netoutput (user memory output)
 */
Status UserMemoryOutputAndContinuousInputChecker(CheckFuncContext &context) {
  auto netoutput_node_index_io = context.node_a;
  auto continuous_node_index_io = context.node_b;
  if (context.node_b.node_->GetType() == NETOUTPUT) {
    netoutput_node_index_io = context.node_b;
    continuous_node_index_io = context.node_a;
  }

  // todo: c间接连netoutput的场景
  const auto netoutput_in_anchor = netoutput_node_index_io.node_->GetInDataAnchor(netoutput_node_index_io.index_);
  GE_ASSERT_NOTNULL(netoutput_in_anchor);
  GE_ASSERT_NOTNULL(netoutput_in_anchor->GetPeerOutAnchor());
  if (netoutput_in_anchor->GetPeerOutAnchor()->GetOwnerNode() == continuous_node_index_io.node_) {
    GELOGI("[MemConflict] %s(need continuous input) is connected to %s, no need to insert identity.",
           continuous_node_index_io.ToString().c_str(), netoutput_node_index_io.ToString().c_str());
  } else {
    context.result.insert(netoutput_in_anchor);
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, netoutput_node_index_io);
  }
  return SUCCESS;
}

REGISTER_FUNC(ANCHOR_ATTR_USER_MEMORY_OUTPUT, ANCHOR_ATTR_CONTINUOUS_INPUT,
              UserMemoryOutputAndContinuousInputChecker);
REGISTER_FUNC(ANCHOR_ATTR_USER_MEMORY_OUTPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT,
              UserMemoryOutputAndContinuousInputChecker);
}  // namespace ge
