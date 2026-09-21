/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/remove_unsupported_op/print_op_pass.h"
#include <string>

namespace ge {
Status PrintOpPass::Run(ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  std::string ori_type;
  Status ret = GetOriginalType(node, ori_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OriginalType] of node:%s failed", node->GetName().c_str());
    return ret;
  }
  if (ori_type == "Print") {
    GELOGI("Delete node: %s, type: Print.", node->GetName().c_str());
    return IsolateAndDeleteNode(node, {});
  }
  return SUCCESS;
}

REG_PASS_OPTION("PrintOpPass").LEVELS(OoLevel::kO0);
}  // namespace ge
