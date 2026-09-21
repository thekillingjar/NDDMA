/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/addn_pass.h"

#include <vector>

namespace ge {
namespace {
const size_t kInputSizeSingle = 1;
}  // namespace

Status AddNPass::Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] param [node] must not be null.");
    return PARAM_INVALID;
  }

  if (node->GetType() == ADDN) {
    if (node->GetOpDesc() == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Param op_desc of node is nullptr, check invalid");
      GELOGE(PARAM_INVALID, "[Get][OpDesc] Param [node] op desc is null.");
      return PARAM_INVALID;
    }

    // AddN with single input can be optimized
    if (node->GetOpDesc()->GetInputsSize() == kInputSizeSingle) {
      GELOGD("AddNPass running");
      std::vector<int32_t> io_map = {PassUtils::GetUniqueInDataAnchorIndex(node)};
      return IsolateAndDeleteNode(node, io_map);
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("AddNPass").LEVELS(OoLevel::kO3);
}  // namespace ge
