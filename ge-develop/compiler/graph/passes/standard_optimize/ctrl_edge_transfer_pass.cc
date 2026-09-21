/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/ctrl_edge_transfer_pass.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
/* Pass Explaination:
 *
 * After optimizing such as constant folding, it will form the following ctrl relationship
 * The sceno like this is unreasonable and when unknown shape, it will be error because
 * constant does not generate task. So when graph is stability, transfer the ctrl edge to
 * next op and guatantee the timing relationship
 *
 * A(ctrl edge)----constant------(ctrl edge)B or A(ctrl edge)----constant-----(data edge)B
 *
 * when after process, it will be like as follows:
 *
 *  A   constant
 *   \  /
 *    B
 */
Status CtrlEdgeTransferPass::Run(ge::ComputeGraphPtr graph) {
  GELOGD("CtrlEdgeTransferPass start running");
  GE_CHECK_NOTNULL(graph);

  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  if (!is_dynamic_shape) {
    return SUCCESS;
  }

  for (ge::NodePtr &n : graph->GetDirectNode()) {
    auto op_desc = n->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    auto op_type = op_desc->GetType();
    if (op_type == CONSTANT || op_type == CONSTANTOP) {
      if (n->GetInAllNodes().empty()) {
        GELOGD("[CtrlEdgeTransferPass] node [%s] in nodes is empty", n->GetName().c_str());
        continue;
      }

      GELOGD("start to tranfer ctrl edge for const node [%s]", n->GetName().c_str());

      for (auto &in_control_node : n->GetInControlNodes()) {
        GE_CHECK_NOTNULL(in_control_node);
        GE_CHK_GRAPH_STATUS_RET(ge::GraphUtils::RemoveEdge(in_control_node->GetOutControlAnchor(),
                                                           n->GetInControlAnchor()),
                                "[Remove][ControlEdge] between %s and %s failed",
                                in_control_node->GetName().c_str(), n->GetName().c_str());
        for (auto &out_node : n->GetOutNodes()) {
          if (out_node == nullptr) {
            continue;
          }
          GE_CHK_GRAPH_STATUS_RET(ge::GraphUtils::AddEdge(in_control_node->GetOutControlAnchor(),
                                                          out_node->GetInControlAnchor()),
                                  "[Add][ControlEdge] between %s and %s failed.",
                                  in_control_node->GetName().c_str(), out_node->GetName().c_str());
        }
      }
    }
  }
  GELOGD("CtrlEdgeTransferPass running success!");
  return SUCCESS;
}

REG_PASS_OPTION("CtrlEdgeTransferPass").LEVELS(OoLevel::kO3);
}  // namespace ge
