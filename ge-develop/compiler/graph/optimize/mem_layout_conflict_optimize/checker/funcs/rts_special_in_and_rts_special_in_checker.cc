/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/mem.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/check_register.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker_log.h"

namespace ge {
Status RtsSpecialInputAndRtsSpecialInputChecker(CheckFuncContext &context) {
  // 能走到这个函数，说明a或者b一定是特殊类型
  if (MemLayoutConflictUtil::IsSameMemType(context)) {
    return SUCCESS;
  }

  const auto &node_index_io =
      ((context.node_a.node_->GetOpDesc()->GetId() < context.node_b.node_->GetOpDesc()->GetId()) &&
       (!MemLayoutConflictUtil::IsNodeOutRefFromInput(context.node_a, context.all_nodes)))
          ? context.node_a
          : context.node_b;
  context.result.insert(node_index_io.node_->GetInDataAnchor(static_cast<int32_t>(node_index_io.index_)));
  GE_MEM_LAYOUT_CONFLICT_LOGI(context, node_index_io);
  return SUCCESS;
}

REGISTER_FUNC(ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT,
              RtsSpecialInputAndRtsSpecialInputChecker);
}  // namespace ge
