/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/label/partitioned_call_op_label_maker.h"

#include "framework/common/util.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace ge {
constexpr int32_t kSubGraphIndex = 0;

/**
 * @ingroup ge
 * @brief Make label node to functional call.
 * @param [in/out] label_index: serial id for whole graph.
 * @return: 0 for success / others for fail
 */
Status PartitionedCallOpLabelMaker::Run(uint32_t &label_index) {
  (void)label_index;
  GE_CHECK_NOTNULL(parent_node_);
  GE_CHECK_NOTNULL(parent_graph_);

  OpDescPtr call_desc = parent_node_->GetOpDesc();
  GE_CHECK_NOTNULL(call_desc);

  std::string sub_graph_name = call_desc->GetSubgraphInstanceName(kSubGraphIndex);
  if (sub_graph_name.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) subgraph_index:%d name is empty, check invalid",
                       call_desc->GetName().c_str(), call_desc->GetType().c_str(), kSubGraphIndex);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Node:%s has no subgraph name.", sub_graph_name.c_str());
    return FAILED;
  }

  ComputeGraphPtr sub_graph = parent_graph_->GetSubgraph(sub_graph_name);
  if (sub_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) subgraph_name:%s does not exist in parent_graph, check invalid",
                       call_desc->GetName().c_str(), call_desc->GetType().c_str(),
                       sub_graph_name.c_str());
    GELOGE(INTERNAL_ERROR, "[Get][SubGraph] Node:%s has no subgraph.", sub_graph_name.c_str());
    return FAILED;
  }

  const std::string stream_active_name = parent_node_->GetName() + "/StreamActive"; // rtStreamActive
  NodePtr stream_active = AddStreamActive(sub_graph, stream_active_name);
  if (stream_active == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add StreamActive node in graph:%s fail",
                      sub_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][StreamActive] in Subgraph:%s failed.", sub_graph->GetName().c_str());
    return FAILED;
  }

  for (auto &node : sub_graph->GetDirectNode()) {
    if (node->GetType() == NETOUTPUT) {
      auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      (void)AttrUtils::SetBool(op_desc, ATTR_NAME_SUBGRAPH_END_NODE, true);
    }
  }

  return SUCCESS;
}

REGISTER_LABEL_MAKER(PARTITIONEDCALL, PartitionedCallOpLabelMaker);
REGISTER_LABEL_MAKER(STATEFULPARTITIONEDCALL, PartitionedCallOpLabelMaker);
}  // namespace ge

