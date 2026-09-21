/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/set_ffts_plus_attr_pass.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace {
// temporary attr.FE formal solotion will determine the attribute name.
const std::string ATTR_NAME_FFTS_PLUS_ENGINE = "_ffts_plus";
}

namespace ge {
Status SetFftsPlusAttrPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);

  if (graph->GetParentNode() == nullptr) {
    return SUCCESS;
  }

  const OpDescPtr &op_desc = graph->GetParentNode()->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (!op_desc->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH)) {
    return SUCCESS;
  }

  for (const NodePtr &node : graph->GetAllNodes()) {
    const auto tmp_op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(tmp_op_desc);
    (void)ge::AttrUtils::SetBool(tmp_op_desc, ATTR_NAME_FFTS_PLUS_ENGINE, true);

    if (tmp_op_desc->GetType() == NOOP) {
      (void)ge::AttrUtils::SetBool(tmp_op_desc, ATTR_NAME_NOTASK, true);
    }
  }

  return SUCCESS;
}

REG_PASS_OPTION("SetFftsPlusAttrPass").LEVELS(OoLevel::kO3);
}
