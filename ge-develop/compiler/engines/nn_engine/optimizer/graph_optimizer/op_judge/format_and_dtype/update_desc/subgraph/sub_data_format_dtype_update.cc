/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/update_desc/subgraph/sub_data_format_dtype_update.h"

namespace fe {
Status SubDataFormatDtypeUpdate::UpdateTensorDesc(ge::NodePtr node_ptr) {
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto owner_graph = node_ptr->GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(owner_graph);

  string graph_name = owner_graph->GetName();
  string node_name = node_ptr->GetName();

  // find the reflections
  ge::RefCell key(node_ptr->GetName(), node_ptr, ge::NODE_OUT, 0);
  FE_LOGD("LookUpKey: the %s %d of Graph[%s] Op[%s].", STR_OUTPUT_LOWERCASE.c_str(), 0, graph_name.c_str(),
          node_name.c_str());
  std::unordered_set<ge::RefCell, ge::RefCellHash> reflections;

  FE_CHECK_NOTNULL(reflection_builder_ptr_);
  auto status = reflection_builder_ptr_->LookUpRefRelations(key, reflections);
  if (status != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdTensorDesc] Node[%s]: failed to look up relations for %s 0.",
                    node_name.c_str(), STR_OUTPUT_LOWERCASE.c_str());
    return FAILED;
  }

  if (!FeGraphUtils::CheckRelatedEdgesOriginShape(reflections)) {
    FE_LOGD("Graph[%s] Op[%s]: check relations for %s tensor %u unsuccessful.", graph_name.c_str(), node_name.c_str(),
            STR_OUTPUT_LOWERCASE.c_str(), 0);
    return SUCCESS;
  }

  // 1. update the format and shape
  UpdateFormat(node_ptr, 0, false);

  // 2. update the data_type
  if (UpdateDtype(node_ptr) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdTensorDesc] Failed to update dtype for Graph[%s], Node[%s].",
                    graph_name.c_str(), node_name.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status SubDataFormatDtypeUpdate::UpdateDtype(ge::NodePtr node_ptr) {
  string graph_name = node_ptr->GetOwnerComputeGraph()->GetName();
  string node_name = node_ptr->GetName();
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();

  // 1. get the previous node
  ge::OutDataAnchorPtr pre_out_data_anchor_ptr;
  if (FeGraphUtils::GetPreOutAnchorOfSubData(node_ptr, pre_out_data_anchor_ptr) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOptJdgInst][UpdFmtAndDtype][UpdDtype] Failed to get the previous out_anchor for Graph[%s] Data[%s].",
        graph_name.c_str(), node_name.c_str());
    return FAILED;
  }

  FE_CHECK_NOTNULL(pre_out_data_anchor_ptr);
  ge::NodePtr pre_node_ptr = pre_out_data_anchor_ptr->GetOwnerNode();
  FE_CHECK_NOTNULL(pre_node_ptr);
  ge::OpDescPtr pre_op_desc_ptr = pre_node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(pre_op_desc_ptr);
  int pre_output_idx = pre_out_data_anchor_ptr->GetIdx();
  ge::GeTensorDesc pre_op_output_desc = pre_op_desc_ptr->GetOutputDesc(pre_output_idx);

  // 2. update the dtype of the output of data
  int index = 0;
  ge::GeTensorDesc output_desc = op_desc_ptr->GetOutputDesc(index);
  ge::DataType cur_dtype = output_desc.GetDataType();
  ge::DataType new_dtype = pre_op_output_desc.GetDataType();
  if (cur_dtype != new_dtype) {
    FE_LOGD(
        "Graph[%s] Data[%s]: update the output, cur_data_type=[%s], "
        "newDataType=[%s].",
        graph_name.c_str(), node_name.c_str(), ge::TypeUtils::DataTypeToSerialString(cur_dtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(new_dtype).c_str());
    output_desc.SetDataType(new_dtype);
    (void)op_desc_ptr->UpdateOutputDesc(index, output_desc);
  }

  // 3. update the related edges
  if (UpdateDtypeOfRelatedEdges(output_desc, node_ptr, ge::NODE_OUT, index) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdDtype] Graph[%s] Data[%s]: Failed to update related edges.",
                    graph_name.c_str(), node_name.c_str());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
