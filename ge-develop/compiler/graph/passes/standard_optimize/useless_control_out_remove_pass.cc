/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/useless_control_out_remove_pass.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
Status UselessControlOutRemovePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);

  if ((node->GetType() != CONSTANT) && (node->GetType() != CONSTANTOP)) {
    return SUCCESS;
  }
  GELOGD("UselessControlOutRemovePass running, node: %s.", node->GetName().c_str());

  // const has no control input
  if (node->GetInControlNodes().empty()) {
    if (node->GetOutDataNodes().empty()) {
      // It is an isolated const, just remove it.
      GELOGI("Delete isolated const: %s.", node->GetName().c_str());
      GE_CHK_STATUS_RET(IsolateAndDeleteNode(node, {}));
      AddNodeDeleted(node);
    } else {
      auto out_ctrl_anchor = node->GetOutControlAnchor();
      if (out_ctrl_anchor != nullptr && (out_ctrl_anchor->GetPeerAnchorsSize() > 0UL)) {
        GELOGI("Node: %s unlink all out control edge.", node->GetName().c_str());
        out_ctrl_anchor->UnlinkAll();
      }
    }
  }

  return SUCCESS;
}

REG_PASS_OPTION("UselessControlOutRemovePass").LEVELS(OoLevel::kO3);
}  // namespace ge
