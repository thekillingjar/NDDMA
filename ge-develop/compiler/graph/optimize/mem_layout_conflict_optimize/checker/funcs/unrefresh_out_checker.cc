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

namespace ge {
Status NotsupportedRefreshOutputChecker(CheckFuncContext &context) {
  if ((!context.graph_info.is_feature_map_refreshable) &&
      (!MemLayoutConflictUtil::HasNotSupportPhysicalMemoryRefreshNode(context))) {
    GELOGI("[MemConflict] feature map is not refreshable and does not have not-support-physical-memory-refresh node."
           " [%s][%s] and [%s][%s] do not need to insert identity.",
           context.node_a.node_->GetNamePtr(), CheckerLog::ToStr(context.type_a).c_str(),
           context.node_b.node_->GetNamePtr(), CheckerLog::ToStr(context.type_b).c_str());
    return SUCCESS;
  }
  OutDataAnchorPtr out_data_anchor;
  if (MemLayoutConflictUtil::IsContainTargetType(context.type_a,
                                                 ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT)) {
    out_data_anchor = context.node_a.node_->GetOutDataAnchor(context.node_a.index_);
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, context.node_a);
  } else {
    out_data_anchor = context.node_b.node_->GetOutDataAnchor(context.node_b.index_);
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, context.node_b);
  }
  context.result.insert(out_data_anchor);
  return SUCCESS;
}
REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT,
              NotsupportedRefreshOutputChecker);

REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              NotsupportedRefreshOutputChecker);

REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_CONTINUOUS_INPUT,
              NotsupportedRefreshOutputChecker);

REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_CONTINUOUS_OUTPUT,
              NotsupportedRefreshOutputChecker);

REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT,
              NotsupportedRefreshOutputChecker);

REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT,
              NotsupportedRefreshOutputChecker);

REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT,
              NotsupportedRefreshOutputChecker);

REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT,
              NotsupportedRefreshOutputChecker);

REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_NORMAL_INPUT,
              NotsupportedRefreshOutputChecker);

REGISTER_FUNC(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
              ANCHOR_ATTR_NORMAL_OUTPUT,
              NotsupportedRefreshOutputChecker);
}  // namespace ge
