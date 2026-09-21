/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_FUSION_COMMON_GRAPH_NODE_MAP_UTIL_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_FUSION_COMMON_GRAPH_NODE_MAP_UTIL_H_

#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "graph/compute_graph.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"

namespace fe {
class GraphNodeMapUtil {
 public:
  static Status CreatNodeOpTypeMap(NodeMapInfoPtr &node_map_info, ge::ComputeGraph &graph);

  static void DelNodeFromOpTypeMap(NodeMapInfoPtr &node_map_info, ge::NodePtr &node_ptr);

  static Status SetOpTypeMapToGraph(NodeMapInfoPtr &node_map_info, ge::ComputeGraph &graph);

  static Status GetOpTypeMapToGraph(NodeMapInfoPtr &node_type_map, ge::ComputeGraph &graph);

  static Status GetOpTypeMapToGraph(NodeMapInfoPtr &node_type_map, const ge::ComputeGraph &graph);

  static Status ClearOpTypeMapToGraph(ge::ComputeGraph &graph);

  static Status ReCreateNodeTypeMapInGraph(ge::ComputeGraph &graph);

  static Status CreatAndSetOpTypeMap(NodeMapInfoPtr &node_map_info, ge::ComputeGraph &graph);
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_FUSION_COMMON_GRAPH_NODE_MAP_UTIL_H_
