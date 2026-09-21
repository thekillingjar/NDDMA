/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/update_desc/subgraph/sub_netoutput_format_dtype_update.h"

namespace fe {
Status SubNetOutputFormatDtypeUpdate::UpdateTensorDesc(ge::NodePtr node_ptr) {
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto owner_graph = node_ptr->GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(owner_graph);
  auto parent = owner_graph->GetParentNode();
  if (parent == nullptr || parent->GetOpDesc()->HasAttr(ge::ATTR_INSERT_BY_MBATCH)) {
    FE_LOGD("Parent node of %s has attr ATTR_INSERT_BY_MBATCH.", node_ptr->GetName().c_str());
    return SUCCESS;
  }

  string graph_name = owner_graph->GetName();
  string node_name = node_ptr->GetName();

  for (const auto &in_data_anchor_ptr : node_ptr->GetAllInDataAnchors()) {
    FE_CHECK_NOTNULL(in_data_anchor_ptr);
    int index = in_data_anchor_ptr->GetIdx();
    // find the reflections
    ge::RefCell key(node_ptr->GetName(), node_ptr, ge::NODE_IN, index);
    FE_LOGD("LookUpKey: the %s %d of Graph[%s] Op[%s].", STR_INPUT_LOWERCASE.c_str(), index, graph_name.c_str(),
            node_name.c_str());
    std::unordered_set<ge::RefCell, ge::RefCellHash> reflections;
    auto status = reflection_builder_ptr_->LookUpRefRelations(key, reflections);
    if (status != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdTensorDesc] Failed to look up relations for Node [%s]: %s %d.",
                      node_name.c_str(), STR_INPUT_LOWERCASE.c_str(), index);
      continue;
    }

    if (!FeGraphUtils::CheckRelatedEdgesOriginShape(reflections)) {
      FE_LOGD("Graph[%s] Op[%s]: check relations for %s tensor %u unsuccessful.", graph_name.c_str(), node_name.c_str(),
              STR_INPUT_LOWERCASE.c_str(), 0);
      continue;
    }
    // 1. update the format and shape
    UpdateFormat(node_ptr, index, true);

    // 2. update the dtype and the related edges
    if (UpdateDtype(node_ptr, in_data_anchor_ptr) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdTensorDesc] Node[%s]: Failed to update dtype of input %d.",
                      node_name.c_str(), index);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status SubNetOutputFormatDtypeUpdate::UpdateDtype(ge::NodePtr node_ptr, const ge::InDataAnchorPtr &in_data_anchor_ptr) {
  auto owner_graph = node_ptr->GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(owner_graph);
  string graph_name = owner_graph->GetName();
  string node_name = node_ptr->GetName();
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  int index = in_data_anchor_ptr->GetIdx();
  ge::GeTensorDesc input_desc = op_desc_ptr->GetInputDesc(index);
  // 1. if the dtype has been update, return
  if (ge::AttrUtils::HasAttr(input_desc, ATTR_NAME_DTYPE_IS_UPDATED)) {
    FE_LOGD(
        "Graph[%s] NetOutput[%s]: the dtype of the input %d has been "
        "updated, return.",
        graph_name.c_str(), node_name.c_str(), index);
    return SUCCESS;
  }

  // 2. get the previous node
  ge::OutDataAnchorPtr peer_out_data_anchor = in_data_anchor_ptr->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(peer_out_data_anchor);
  ge::NodePtr pre_node_ptr = peer_out_data_anchor->GetOwnerNode();
  FE_CHECK_NOTNULL(pre_node_ptr);
  ge::OpDescPtr pre_op_desc_ptr = pre_node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(pre_op_desc_ptr);
  ge::GeTensorDesc pre_op_output_desc = pre_op_desc_ptr->GetOutputDesc(peer_out_data_anchor->GetIdx());

  // 3. update the dtype of the input of netoutput
  ge::DataType cur_dtype = input_desc.GetDataType();
  ge::DataType new_dtype = pre_op_output_desc.GetDataType();
  if (cur_dtype != new_dtype) {
    FE_LOGD(
        "Graph[%s] NetOutput[%s]: update the input %d, cur_data_type=[%s], "
        "newDataType=[%s].",
        graph_name.c_str(), node_name.c_str(), index, ge::TypeUtils::DataTypeToSerialString(cur_dtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(new_dtype).c_str());
    input_desc.SetDataType(new_dtype);
    (void)op_desc_ptr->UpdateInputDesc(index, input_desc);
  }

  // 4. update the related edges
  if (UpdateDtypeOfRelatedEdges(input_desc, node_ptr, ge::NODE_IN, index) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOptJdgInst][UpdFmtAndDtype][UpdDtype] Graph[%s] NetOutput[%s]: Failed to update the related edges of the "
        "input %d.",
        graph_name.c_str(), node_name.c_str(), index);
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
