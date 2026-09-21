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
/*
 *    a    b
 *     \   /
 *  PhonyConcat
 *       |
 *   PhonySplit
 *      /  \
 *     c    d
 */
bool IsContinuosInputNodeConnectContinuousOutNode(CheckFuncContext &context) {
  auto continuous_in = context.node_a;
  auto continuous_out = context.node_b;
  if (MemLayoutConflictUtil::IsContainTargetType(context.type_b, ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT)) {
    continuous_in = context.node_b;
    continuous_out = context.node_a;
  }
  if ((continuous_out.node_->GetInDataAnchor(0) != nullptr)) {
    if (continuous_out.node_->GetInDataAnchor(0)->GetPeerOutAnchor() == continuous_in.node_->GetOutDataAnchor(0)) {
      GELOGI("[MemConflict] node(%s) that require continuous input are directly connected to node((%s)) that require"
             " continuous output, skip.", continuous_in.ToString().c_str(), continuous_out.ToString().c_str());
      return true;
    }
  }
  return false;
}

Status NoPaddingContinuousInputAndNoPaddingContinuousOutputChecker(CheckFuncContext &context) {
  if (IsContinuosInputNodeConnectContinuousOutNode(context)) {
    return SUCCESS;
  }
  auto continuous_in = context.node_a;
  if (MemLayoutConflictUtil::IsContainTargetType(context.type_b, ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT)) {
    continuous_in = context.node_b;
  }
  context.result.insert(MemLayoutConflictUtil::GetAnchorFromIndexIo(continuous_in));
  GE_MEM_LAYOUT_CONFLICT_LOGI(context, continuous_in);
  return SUCCESS;
}

REGISTER_FUNC(ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT,
              NoPaddingContinuousInputAndNoPaddingContinuousOutputChecker);
}  // namespace ge
