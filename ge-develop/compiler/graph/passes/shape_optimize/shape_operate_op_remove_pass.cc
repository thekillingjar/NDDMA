/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/shape_operate_op_remove_pass.h"

namespace ge {
Status ShapeOperateOpRemovePass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
    bool to_be_deleted = false;
    GE_IF_BOOL_EXEC(!AttrUtils::GetBool(op_desc, ATTR_TO_BE_DELETED, to_be_deleted), to_be_deleted = false);
    GE_IF_BOOL_EXEC(to_be_deleted,
                    GE_CHK_STATUS_RET(graph->RemoveNode(node),
                                      "[Remove][Node] %s from graph:%s failed!", node->GetName().c_str(),
                                      graph->GetName().c_str()));
  }
  return SUCCESS;
}

REG_PASS_OPTION("ShapeOperateOpRemovePass").LEVELS(OoLevel::kO0);
}  // namespace ge
