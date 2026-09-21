/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_GRAPH_FE_GRAPH_UTILS_H_
#define FUSION_ENGINE_UTILS_COMMON_GRAPH_FE_GRAPH_UTILS_H_
#include <string>
#include <vector>
#include <algorithm>
#include "common/fe_utils.h"
#include "common/op_info_common.h"
#include "common/util/op_info_util.h"
#include "common/fe_context_utils.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ref_relation.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"

namespace fe {
struct RelationUpdateInfo {
  ge::Format primary_format = ge::FORMAT_RESERVED;
  int32_t sub_format = 0;
  ge::GeShape shape;
  ge::DataType data_type = ge::DT_UNDEFINED;
  string attr_name;
  int attr_value = -1;

  RelationUpdateInfo(const ge::Format &update_primary_format, const int32_t &update_sub_format,
                     const ge::GeShape &update_shape, string update_attr_name, int update_attr_value) {
    primary_format = update_primary_format;
    sub_format = update_sub_format;
    shape = update_shape;
    attr_name = update_attr_name;
    attr_value = update_attr_value;
  };

  RelationUpdateInfo(const ge::DataType &update_data_type, string update_attr_name, int update_attr_value) {
    data_type = update_data_type;
    attr_name = update_attr_name;
    attr_value = update_attr_value;
  };
};

class FeGraphUtils {
 public:
  static void DumpSubGraphAndOnnx(const ge::ComputeGraph &graph, const std::string &suffix);

  static void DumpGraphAndOnnx(const ge::ComputeGraph &graph, const std::string &suffix);
  static void DumpGraph(const ge::ComputeGraph &graph, const std::string &suffix);

  static void IsNodeSpecificType(const std::unordered_set<string> &types, ge::NodePtr &node,
                                 bool &matched);
  static void ProcessPartitionedCall(const std::string &name, std::string &type, ge::NodePtr &parent_node,
                                     ge::NodePtr &really_parent_node, ge::NodePtr &node);

  static bool IsPeerOutConst(const ge::Node *node, const int &anchor_index, ge::NodePtr &peer_out_node);

  static bool IsPeerOutWeight(ge::Node *node, const int &anchor_index, ge::NodePtr &peer_out_node);
  
  static bool IsMainGraphData(const ge::OpDescPtr &op_desc_ptr);

  static bool IsMainGraphNetOutput(const ge::OpDescPtr &op_desc_ptr);

  static bool IsSubGraphData(const ge::OpDescPtr &op_desc_ptr);

  static bool IsSubGraphNetOutput(const ge::OpDescPtr &op_desc);

  static bool IsSubGraphDataOrNetOutput(const ge::OpDescPtr &op_desc_ptr);

  static bool IsNotSubGraphDataAndNetOutput(const ge::OpDescPtr &op_desc_ptr);

  static Status GetPreOutAnchorOfSubData(const ge::NodePtr &data_node_ptr,
                                         ge::OutDataAnchorPtr &pre_out_data_anchor_ptr);

  static Status GetPreSubNetoutputInAnchor(std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections,
                                           std::vector<ge::InDataAnchorPtr> &vec_netoutput_in_ahchor);

  static Status GetNextInAnchorsOfSubNetOutput(const ge::NodePtr &net_output_node_ptr, const int &input_index,
                                               std::vector<ge::InDataAnchorPtr> &next_in_data_anchors);

  static Status GetNextSubDatasOutAnchors(std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections,
                                          std::vector<ge::OutDataAnchorPtr> &out_data_anchors);

  static Status UpdateFormatOfRelatedEdges(const std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections,
                                           const RelationUpdateInfo &relation_update_info_a);

  static bool CheckRelatedEdgesOriginShape(const std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections);

  static void GetGraphIdFromAttr(const ge::ComputeGraph &graph, string &graph_id);

  static bool CheckTypeOnRootGraph(const std::unordered_set<string> &types, ge::NodePtr &parent_node);

  static Status GetAoeTypeFromRootGraph(ge::ComputeGraph& graph, std::string &aoe_type);

  static void FindPeerOpType(const ge::NodePtr &node, const bool is_input, std::string &peer_op_type);

  static void GetPrecisionModeFromGraph(const ge::ComputeGraph& graph, fe::PrecisionMode &precision_mode);
};
}  // namespace fe

#endif  // FUSION_ENGINE_UTILS_COMMON_GRAPH_FE_GRAPH_UTILS_H_
