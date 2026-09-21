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

namespace ge {
Status ImmutableOutputAndNoPaddingContinuousInputChecker(CheckFuncContext &context) {
  auto immutable_node_index_io = context.node_a;
  auto continuous_node = context.node_b;
  if (MemLayoutConflictUtil::IsContainTargetType(context.type_b, ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT)) {
    immutable_node_index_io = context.node_b;
    continuous_node = context.node_a;
  }

  /*
   *  PhonyConcat 的所有输入是同一个anchor，不需要插入identity
   *            variable
   *               /\
   *  assign_slice0 assign_slice1 (inplace)
   *             \   /
   *          PhonyConcat
   */
  if (MemLayoutConflictUtil::AllRealInputsAreTheSameOutAnchor(continuous_node.node_)) {
    return SUCCESS;
  }

  context.result.insert(continuous_node.node_->GetInDataAnchor(continuous_node.index_));
  GE_MEM_LAYOUT_CONFLICT_LOGI(context, continuous_node);
  return SUCCESS;
}
REGISTER_FUNC(ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT,
              ImmutableOutputAndNoPaddingContinuousInputChecker);
}  // namespace ge
