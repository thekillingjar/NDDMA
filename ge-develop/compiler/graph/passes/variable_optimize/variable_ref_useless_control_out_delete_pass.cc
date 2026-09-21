/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/variable_optimize/variable_ref_useless_control_out_delete_pass.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
Status VariableRefUselessControlOutDeletePass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetDirectNode()) {
    if (!OpTypeUtils::IsVarLikeNode(node->GetType())) {
      continue;
    }
    std::string src_var_name;
    if (!AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, src_var_name)) {
      continue;
    }
    auto src_nodes = node->GetInDataNodes();
    if (src_nodes.empty()) {
      GELOGW("The variable ref name %s(ref %s) does not has a input node",
             node->GetName().c_str(), src_var_name.c_str());
      continue;
    }
    auto &src_node = src_nodes.at(0);
    auto controlled_nodes_vec = src_node->GetOutNodes();
    std::set<NodePtr> controlled_nodes{controlled_nodes_vec.begin(), controlled_nodes_vec.end()};

    auto out_control_anchor = node->GetOutControlAnchor();
    for (const auto &dst_node_anchor : out_control_anchor->GetPeerInControlAnchors()) {
      if (controlled_nodes.count(dst_node_anchor->GetOwnerNode()) > 0) {
        GELOGI("Unlink the duplicated control edge from variable ref %s to %s, prev node %s",
               node->GetName().c_str(),
               dst_node_anchor->GetOwnerNode()->GetName().c_str(),
               src_node->GetName().c_str());
        out_control_anchor->Unlink(dst_node_anchor);
      }
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("VariableRefUselessControlOutDeletePass").LEVELS(OoLevel::kO3);
}
