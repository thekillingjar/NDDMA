/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/update_desc/subgraph/sub_graph_format_dtype_update.h"

namespace fe {
void SubGraphFormatDtypeUpdate::UpdateFormat(ge::NodePtr node_ptr, const int &index, const bool &is_input) {
  auto owner_graph = node_ptr->GetOwnerComputeGraph();
  FE_CHECK(owner_graph == nullptr,
           FE_LOGW("Owner graph of node %s is null.", node_ptr->GetName().c_str()), return);
  string graph_name = owner_graph->GetName();
  string node_name = node_ptr->GetName();
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();

  ge::GeTensorDesc tensor_desc = is_input ? op_desc_ptr->GetInputDesc(index) : op_desc_ptr->GetOutputDesc(index);
  auto cur_format = tensor_desc.GetFormat();
  ge::Format cur_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(cur_format));
  auto &cur_shape = tensor_desc.MutableShape();
  if (cur_primary_format != tensor_desc.GetOriginFormat()) {
    string input_or_output = is_input ? STR_INPUT_LOWERCASE : STR_OUTPUT_LOWERCASE;
    auto cur_sub_format = ge::GetSubFormat(cur_format);
    auto new_format = static_cast<ge::Format>(ge::GetFormatFromSub(tensor_desc.GetOriginFormat(), cur_sub_format));
    FE_LOGD(
        "Graph[%s] Op[%s]: update the %s %d, cur_format=[%s], "
        "curShape=[%s], new_format=[%s], new_shape=[%s].",
        graph_name.c_str(), node_name.c_str(), input_or_output.c_str(), index,
        ge::TypeUtils::FormatToSerialString(cur_format).c_str(), GetShapeDims(cur_shape).c_str(),
        ge::TypeUtils::FormatToSerialString(new_format).c_str(), GetShapeDims(tensor_desc.GetOriginShape()).c_str());

    tensor_desc.SetFormat(static_cast<ge::Format>(new_format));
    tensor_desc.SetShape(tensor_desc.GetOriginShape());
    (void)op_desc_ptr->UpdateInputDesc(index, tensor_desc);
  }
}

Status SubGraphFormatDtypeUpdate::UpdateDtypeOfRelatedEdges(const ge::GeTensorDesc &tensor_desc,
                                                            const ge::NodePtr &node_ptr,
                                                            const ge::InOutFlag &in_out_flag, const int &index) {
  auto owner_graph = node_ptr->GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(owner_graph);
  string graph_name = owner_graph->GetName();
  string node_name = node_ptr->GetName();

  // 1. find the reflections
  ge::RefCell key(node_ptr->GetName(), node_ptr, in_out_flag, index);
  string input_or_output_str = in_out_flag == ge::NODE_IN ? STR_INPUT_LOWERCASE : STR_OUTPUT_LOWERCASE;
  FE_LOGD("LookUpKey: the %s %d of Graph[%s] Op[%s].", input_or_output_str.c_str(), index, graph_name.c_str(),
          node_name.c_str());
  std::unordered_set<ge::RefCell, ge::RefCellHash> reflections;
  auto status = reflection_builder_ptr_->LookUpRefRelations(key, reflections);
  if (status != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdDtype] Node[%s]: failed to look up reference relations for %s %d.",
                    node_name.c_str(), IS_INPUT_TO_STRING(in_out_flag == ge::NODE_IN), index);
    return FAILED;
  }

  // 2. update all related edges
  RelationUpdateInfo relation_update_info = {tensor_desc.GetDataType(), ATTR_NAME_DTYPE_IS_UPDATED, 1};
  if (UpdateDtypeOfRelatedEdges(reflections, relation_update_info) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdDtype] Failed to update related edges for Graph[%s], Node[%s].",
                    graph_name.c_str(), node_name.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status SubGraphFormatDtypeUpdate::UpdateDtypeOfRelatedEdges(
    const std::unordered_set<ge::RefCell, ge::RefCellHash> &reflections,
    const RelationUpdateInfo &relation_update_info) const {
  for (const auto &cell : reflections) {
    ge::NodePtr node_ptr = cell.node;
    FE_CHECK_NOTNULL(node_ptr);
    ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);
    auto owner_graph = node_ptr->GetOwnerComputeGraph();
    FE_CHECK_NOTNULL(owner_graph);
    string graph_name = owner_graph->GetName();
    string node_name = node_ptr->GetName();

    string input_or_output = cell.in_out == ge::NODE_IN ? STR_INPUT_LOWERCASE : STR_OUTPUT_LOWERCASE;
    FE_LOGD("Relations: the %s %d of Graph[%s] Op[%s].", input_or_output.c_str(), cell.in_out_idx, graph_name.c_str(),
            node_name.c_str());

    // 1. get the input or output desc
    auto index = cell.in_out_idx;
    auto desc = (cell.in_out == ge::NODE_IN ? op_desc_ptr->GetInputDesc(static_cast<uint32_t>(index))
                                            : op_desc_ptr->GetOutputDesc(static_cast<uint32_t>(index)));

    // 2. set the new dtype
    ge::DataType cur_dtype = desc.GetDataType();
    ge::DataType new_dtype = relation_update_info.data_type;
    if (cur_dtype != new_dtype) {
      desc.SetDataType(new_dtype);
      FE_LOGD(
          "Graph[%s] Op[%s]: update the %s %d desc, "
          "curDtype=[%s], new_dtype=[%s].",
          graph_name.c_str(), node_name.c_str(), input_or_output.c_str(), index,
          ge::TypeUtils::DataTypeToSerialString(cur_dtype).c_str(),
          ge::TypeUtils::DataTypeToSerialString(new_dtype).c_str());
    }

    // 3. set the attribute for the tensor desc
    if (!relation_update_info.attr_name.empty()) {
      FE_LOGD(
          "Graph[%s] Op[%s]: update the %s %d desc, the value "
          "of the attribute %s is %d.",
          graph_name.c_str(), node_name.c_str(), input_or_output.c_str(), index, relation_update_info.attr_name.c_str(),
          relation_update_info.attr_value);
      (void)ge::AttrUtils::SetInt(desc, relation_update_info.attr_name, relation_update_info.attr_value);
    }

    // 4. update the tensor desc
    if (cell.in_out == ge::NODE_IN) {
      (void)op_desc_ptr->UpdateInputDesc(static_cast<uint32_t>(index), desc);
    } else {
      (void)op_desc_ptr->UpdateOutputDesc(static_cast<uint32_t>(index), desc);
    }
  }
  return SUCCESS;
}
}  // namespace fe
