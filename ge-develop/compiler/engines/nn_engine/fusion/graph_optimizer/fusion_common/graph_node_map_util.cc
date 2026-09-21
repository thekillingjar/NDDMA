/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/fusion_common/graph_node_map_util.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/op_info_common.h"
#include "graph/compute_graph.h"

namespace fe {
Status GraphNodeMapUtil::CreatNodeOpTypeMap(NodeMapInfoPtr &node_map_info, ge::ComputeGraph &graph) {
  NodeTypeMapPtr node_map = nullptr;
  FE_MAKE_SHARED((node_map = std::make_shared<NodeTypeMap>()), return FAILED);

  string op_type;
  for (auto &node_ptr : graph.GetDirectNode()) {
    op_type = GetRealNodeType(node_ptr->GetOpDesc());
    GraphPassUtil::AddNodeToNodeTypeMap(node_map, op_type, node_ptr);
  }
  FE_MAKE_SHARED((node_map_info = std::make_shared<NodeMapInfo>()), return FAILED);
  *node_map_info = {0, node_map};

  for (auto &iter : *node_map) {
    FE_LOGD("type: %s, node_num: %zu", iter.first.c_str(), iter.second.size());
  }
  return SUCCESS;
}

void GraphNodeMapUtil::DelNodeFromOpTypeMap(NodeMapInfoPtr &node_map_info, ge::NodePtr &node_ptr) {
  if (node_map_info == nullptr || node_ptr == nullptr) {
    return;
  }
  NodeTypeMapPtr node_type_map = node_map_info->node_type_map;
  string real_op_type = GetRealNodeType(node_ptr->GetOpDesc());
  GraphPassUtil::RemoveNodeFromNodeTypeMap(node_type_map, real_op_type, node_ptr);
}

Status GraphNodeMapUtil::SetOpTypeMapToGraph(NodeMapInfoPtr &node_map_info, ge::ComputeGraph &graph) {
  if (!graph.SetExtAttr("NodeMapInfo", node_map_info)) {
    REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][SetOpTypeMap] set NodeTypeMap failed");
    return FAILED;
  }
  return SUCCESS;
}

Status GraphNodeMapUtil::ClearOpTypeMapToGraph(ge::ComputeGraph &graph) {
  NodeMapInfoPtr node_map_info = nullptr;
  node_map_info = graph.TryGetExtAttr("NodeMapInfo", node_map_info);
  if (node_map_info != nullptr) {
    node_map_info = nullptr;
    if (!graph.SetExtAttr("NodeMapInfo", node_map_info)) {
      REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][clrOpTypeMap] set NodeTypeMap null failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GraphNodeMapUtil::ReCreateNodeTypeMapInGraph(ge::ComputeGraph &graph) {
  NodeMapInfoPtr node_map_info = nullptr;
  node_map_info = graph.TryGetExtAttr("NodeMapInfo", node_map_info);
  if (node_map_info == nullptr) {
    FE_LOGW("Cannot retrieve node_type_map from the graph.");
    return SUCCESS;
  }

  string op_type;
  NodeTypeMapPtr node_type_map_ptr = node_map_info->node_type_map;
  FE_CHECK(node_type_map_ptr == nullptr, FE_LOGW("Cannot retrieve node_type_map from the graph."),
           return SUCCESS);
  node_type_map_ptr->clear();
  for (auto &node_ptr : graph.GetDirectNode()) {
    op_type = GetRealNodeType(node_ptr->GetOpDesc());
    GraphPassUtil::AddNodeToNodeTypeMap(node_type_map_ptr, op_type, node_ptr);
  }
  return SUCCESS;
}

Status GraphNodeMapUtil::CreatAndSetOpTypeMap(NodeMapInfoPtr &node_map_info, ge::ComputeGraph &graph) {
  NodeMapInfoPtr tmp_node_map_info = nullptr;
  if (CreatNodeOpTypeMap(tmp_node_map_info, graph) == FAILED) {
    return FAILED;
  }

  if (SetOpTypeMapToGraph(tmp_node_map_info, graph) == FAILED) {
    return FAILED;
  }

  node_map_info = tmp_node_map_info;
  return SUCCESS;
}
}  // namespace fe
