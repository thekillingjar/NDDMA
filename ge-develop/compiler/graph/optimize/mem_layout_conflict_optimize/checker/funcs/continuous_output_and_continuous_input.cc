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
Status ContinuousOutputAndContinuousInputChecker(CheckFuncContext &context) {
  bool has_conflict = false;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::IsContinuousOutAndInConflict(context, has_conflict));
  if (!has_conflict) {
    GELOGI("[MemConflict] [%s, %s] not conflict",
           CheckerLog::ToStr(context.type_a[context.type_a_index]).c_str(),
           CheckerLog::ToStr(context.type_b[context.type_b_index]).c_str());
    return SUCCESS;
  }
  context.result.insert(MemLayoutConflictUtil::GetAnchorFromIndexIo(context.node_a));
  GE_MEM_LAYOUT_CONFLICT_LOGI(context, context.node_a);
  return SUCCESS;
}

REGISTER_FUNC(ANCHOR_ATTR_CONTINUOUS_INPUT, ANCHOR_ATTR_CONTINUOUS_OUTPUT,
              ContinuousOutputAndContinuousInputChecker);
}  // namespace ge
