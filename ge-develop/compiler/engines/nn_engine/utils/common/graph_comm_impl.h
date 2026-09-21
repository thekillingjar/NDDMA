/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_GRAPH_COMMON_IMPL_H_
#define FUSION_ENGINE_UTILS_COMMON_GRAPH_COMMON_IMPL_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common/aicore_util_types.h"
#include "graph/compute_graph.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
using ScopeNodeMap = std::map<int64_t, std::vector<ge::NodePtr>>;

class GraphCommImpl {
 public:
  GraphCommImpl();
  virtual ~GraphCommImpl();
  GraphCommImpl(const GraphCommImpl &in) = delete;
  GraphCommImpl &operator=(const GraphCommImpl &in) = delete;

  void AddFusionSrc(const uint32_t &src_op_id, const ge::AnchorPtr &src_anchor, const int32_t &fusion_src_index,
                    const int32_t &fusion_dst_index, std::vector<FusionOpSrc> &exist_fusion_src_list) const;

  ge::NodePtr FindNodeInFusNodeList(const string &node_name, const vector<ge::NodePtr> &fus_nodelist) const;

  Status GetAllInEdgeList(const ge::NodePtr &node, std::vector<std::pair<ge::AnchorPtr, ge::AnchorPtr>> &in_edge_pair,
                          const int32_t &edge_type, const std::unordered_set<ge::NodePtr> &fus_node_set,
                          bool is_tuning_mode) const;

  Status GetAllOutEdgeList(const ge::NodePtr &node,
                           std::vector<std::pair<ge::AnchorPtr, ge::AnchorPtr>> &out_edge_pair,
                           const int32_t &edge_type, const std::unordered_set<ge::NodePtr> &fus_node_set) const;

  Status MergeFusionNodeInputCtrlEdgeList(const ge::NodePtr &fus_node,
                                          const vector<FusionDataFlow> &fus_input_ctrl_edge_list) const;

  Status MergeFusionNodeOutputCtrlEdgeList(const ge::NodePtr &fus_node,
                                           const vector<FusionDataFlow> &fus_output_ctrl_edge_list) const;

  void PutEdgeToFusionDataFlowVec(const std::pair<ge::AnchorPtr, ge::AnchorPtr> &edge,
                                  const ge::AnchorPtr &fus_node_anchor, const ge::AnchorPtr &edge_node_anchor,
                                  std::vector<FusionDataFlow> &fus_edge_list) const;

  void SaveFusionNode(const uint32_t &scopeid, const ge::NodePtr &node, ScopeNodeMap &fus_node_map) const;

  bool IsInfusNodeList(const ge::NodePtr &node, const std::unordered_set<ge::NodePtr> &fus_nodelist) const;

  Status AddEdge(const ge::AnchorPtr &src_anchor, const ge::NodePtr &dst_node,
                 const ge::AnchorPtr &old_dst_anchor) const;
};

}  // namespace fe
#endif  // FUSION_ENGINE_UTILS_COMMON_GRAPH_COMMON_IMPL_H_
