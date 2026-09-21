/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/unused_const_pass.h"
#include <string>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
///
/// run pass
/// @param [in] node node to be deleted
/// @return Status
///
Status UnusedConstPass::Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return FAILED;
  }
  if (node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node's op_desc is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] failed, param [opDesc] must not be null.");
    return PARAM_INVALID;
  }

  std::string op_type = node->GetOpDesc()->GetType();
  if (op_type == UNUSEDCONST) {
    GELOGD("op type is unused const.");
    return IsolateAndDeleteNode(node, {-1});
  }
  return SUCCESS;
}

REG_PASS_OPTION("UnusedConstPass").LEVELS(OoLevel::kO1);
}  // namespace ge
