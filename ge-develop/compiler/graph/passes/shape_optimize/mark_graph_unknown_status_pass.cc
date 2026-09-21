/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/mark_graph_unknown_status_pass.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"

namespace ge {
namespace {
const char *const kOwnerGraphIsUnknown = "OwnerGraphIsUnknown";
const char *const kHostCpuEngineName = "DNN_VM_HOST_CPU";
}

Status MarkGraphUnknownStatusPass::Run(ComputeGraphPtr graph) {
  // todo 删除该pass，相关逻辑与图拆分逻辑归一，并移动到子图优化末尾
  GE_CHECK_NOTNULL(graph);
  bool is_unknown_shape = false;
  bool forced_unknown = false;
  bool is_graph_no_tiling = false;
  for (const auto &node : graph->GetDirectNode()) {
    auto desc = node->GetOpDesc();
    bool is_node_no_tiling = false;
    (void)AttrUtils::GetBool(desc, ATTR_NAME_OP_NO_TILING, is_node_no_tiling);
    if (is_node_no_tiling) {
      GELOGD("node [%s] is no tiling, mark known.", node->GetName().c_str());
      is_graph_no_tiling = is_graph_no_tiling || is_node_no_tiling;
      continue;
    }
    GE_CHK_GRAPH_STATUS_RET(ge::NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown_shape),
                            "[Get][ShapeStatus] of node[%s] failed!", node->GetName().c_str());
    if (desc->GetOpEngineName() == kHostCpuEngineName) {
      is_unknown_shape = true;
      GELOGD("Mark host cpu node %s unknown.", node->GetName().c_str());
    }
    if (is_unknown_shape) {
      break;
    }
    if (AttrUtils::GetBool(desc, ATTR_NAME_FORCE_UNKNOWN_SHAPE, forced_unknown) && forced_unknown) {
      GELOGD("node %s was marked as unknown shape.", node->GetName().c_str());
      is_unknown_shape = true;
      break;
    }
  }

  GE_CHK_BOOL_RET_STATUS(!(is_graph_no_tiling && is_unknown_shape), PARAM_INVALID,
                         "No tiling graph[%s] not support unknown shape node", graph->GetName().c_str());

  const auto &node = graph->GetParentNode();
  if (!is_unknown_shape && !is_graph_no_tiling && node != nullptr && node->GetType() == PARTITIONEDCALL) {
    GE_CHK_GRAPH_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown_shape),
                            "[Get][ShapeStatus] of node[%s] failed!", node->GetName().c_str());
  }

  for (const auto &sub_node : graph->GetDirectNode()) {
    GELOGD("Set OwnerGraphIsUnknown attr to node[%s]", sub_node->GetName().c_str());
    (void)AttrUtils::SetBool(sub_node->GetOpDesc(), kOwnerGraphIsUnknown, is_unknown_shape);
  }
  is_unknown_shape = is_unknown_shape || ge::GetContext().GetHostExecFlag();
  graph->SetGraphUnknownFlag(is_unknown_shape);
  GELOGD("mark graph [%s] unknown status success! value is %d", graph->GetName().c_str(), is_unknown_shape);
  return SUCCESS;
}
}  // namespace ge
