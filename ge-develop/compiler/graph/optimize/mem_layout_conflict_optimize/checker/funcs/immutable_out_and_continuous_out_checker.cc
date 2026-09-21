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
#include "common/checker.h"
#include "graph/utils/op_type_utils.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/check_register.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker_log.h"

namespace ge {
/*
 *    var (need to insert identity before hcom)
 *     |
 *    hcom (need continuous output, and output reference input)
 *    / \
 *   a   b
 *
 *  为什么不直接插在var后面？如果var是单输出多引用连接到了ApplyMomentum等更新权重的算子上，直接插在var后面会导致var中数据无法被更新
 */
Status ImmutableOutputAndContinuousOutputChecker(CheckFuncContext &context) {
  bool done = false;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::AssignVarInsertIdentity(context,
                                                                   ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT, done));
  if (done) {
    return SUCCESS;
  }
  auto ref_node_out = context.node_a;
  if (MemLayoutConflictUtil::IsContainTargetType(context.type_a, ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT)) {
    ref_node_out = context.node_b;
  }
  const auto ref_node_in = MemLayoutConflictUtil::GetRefInput(ref_node_out, context.all_nodes);
  GE_ASSERT_NOTNULL(ref_node_in.node_);

  context.result.insert(ref_node_in.node_->GetInAnchor(ref_node_in.index_));
  GE_MEM_LAYOUT_CONFLICT_LOGI(context, ref_node_in);
  return SUCCESS;
}
REGISTER_FUNC(ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT, ANCHOR_ATTR_CONTINUOUS_OUTPUT,
              ImmutableOutputAndContinuousOutputChecker);
REGISTER_FUNC(ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT,
              ImmutableOutputAndContinuousOutputChecker);
}  // namespace ge
