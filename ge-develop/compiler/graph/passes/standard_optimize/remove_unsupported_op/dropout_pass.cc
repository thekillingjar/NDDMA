/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/remove_unsupported_op/dropout_pass.h"

#include <string>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/node_utils.h"

namespace ge {
///
/// run pass
/// @param [in] node node to be optimized
/// @return Status
///
Status DropOutPass::Run(NodePtr &node) {
  GELOGD("DropOutPass running");
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return FAILED;
  }
  if (node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param op_desc of node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] failed, param [opDesc] must not be null.");
    return PARAM_INVALID;
  }
  std::string op_type = node->GetOpDesc()->GetType();
  if (op_type == DROPOUT) {
    GELOGD("op type is dropout.");
    return IsolateAndDeleteNode(node, {0});
  }
  return SUCCESS;
}

REG_PASS_OPTION("DropOutPass").LEVELS(OoLevel::kO0);
}  // namespace ge
