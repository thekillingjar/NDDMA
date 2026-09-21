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
Status UserInputAndNotSupportedAddressRefreshOutputChecker(CheckFuncContext &context) {
  bool done = false;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::AssignVarInsertIdentity(context,
                                                                   ANCHOR_ATTR_USER_MEMORY_INPUT, done));
  if (done) {
    return SUCCESS;
  }
  auto not_refresh_node = context.node_a;
  auto user_in_node = context.node_b;
  if (MemLayoutConflictUtil::IsContainTargetType(context.type_b,
                                                 ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT)) {
    not_refresh_node = context.node_b;
    user_in_node = context.node_a;
  }
  if (context.graph_info.is_feature_map_refreshable ||
      MemLayoutConflictUtil::HasNotSupportPhysicalMemoryRefreshNode(context)) {
    context.result.insert(not_refresh_node.node_->GetOutAnchor(not_refresh_node.index_));
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, not_refresh_node);
  } else {
    context.result.insert(user_in_node.node_->GetOutAnchor(user_in_node.index_));
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, user_in_node);
  }
  return SUCCESS;
}
REGISTER_FUNC(ANCHOR_ATTR_USER_MEMORY_INPUT, ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              UserInputAndNotSupportedAddressRefreshOutputChecker);
}  // namespace ge
