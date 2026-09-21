/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/node_optimize_pass_base.h"
#include <map>
#include <string>
#include <vector>
#include "common/fe_utils.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"
#include "common/op_info_common.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace fe {
vector<FusionPattern *> NodeOptimizePassBase::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern(GetPatternName());
  FE_CHECK(pattern == nullptr,
           REPORT_FE_ERROR("[GraphOpt][NdOpti][DefPtn] Failed to create a new object."), return patterns);
  vector<string> node_types = GetNodeTypes();
  pattern->AddOpDesc(GetPatternName(), node_types).SetOutput(GetPatternName());
  patterns.push_back(pattern);
  return patterns;
}

Status NodeOptimizePassBase::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  ge::NodePtr node_ptr = GetNodeFromMapping(GetPatternName(), mapping);
  FE_CHECK(node_ptr == nullptr, REPORT_FE_ERROR("[GraphOpt][NdOpti][Fus] nodePtr is null, fusion failed."),
           return PARAM_INVALID);
  return DoFusion(graph, node_ptr, fusion_nodes);
}

int64_t NodeOptimizePassBase::GetDimAttrValue(const ge::OpDescPtr &op_desc_ptr, const string &dim_attr,
                                              const bool &is_input) const {
  ge::GeTensorDesc tensor_desc = is_input ? op_desc_ptr->GetInputDesc(0) : op_desc_ptr->GetOutputDesc(0);
  int64_t dim_attr_value = 0;
  (void)ge::AttrUtils::GetInt(op_desc_ptr, dim_attr, dim_attr_value);
  if (dim_attr_value < 0) {
    dim_attr_value += tensor_desc.GetOriginShape().GetDimNum();
  }
  return dim_attr_value;
}

Status NodeOptimizePassBase::InsertNode(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst,
                                        ge::NodePtr &new_node, ge::DataType quant_data_type) {
  FE_CHECK_NOTNULL(src);
  FE_CHECK_NOTNULL(dst);
  ge::NodePtr src_node = src->GetOwnerNode();
  FE_CHECK_NOTNULL(src_node);
  ge::NodePtr dst_node = dst->GetOwnerNode();
  FE_CHECK_NOTNULL(dst_node);

  if (new_node->GetOpDesc()->UpdateInputDesc(0, src_node->GetOpDesc()->GetOutputDesc(src->GetIdx())) != SUCCESS) {
    FE_LOGI("%s update of input_desc was unsuccessful.", new_node->GetName().c_str());
    return FAILED;
  }
  if (new_node->GetOpDesc()->UpdateOutputDesc(0, dst_node->GetOpDesc()->GetInputDesc(dst->GetIdx())) != SUCCESS) {
    FE_LOGI("%s update of output_desc was unsuccessful.", new_node->GetName().c_str());
    return FAILED;
  }
  if (new_node->GetType() == QUANT) {
    new_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(quant_data_type);
    new_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(quant_data_type);
  }
  if (ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS) {
    FE_LOGI("Remove %s input0 edge unsuccessful.", dst_node->GetName().c_str());
    return FAILED;
  }
  if (ge::GraphUtils::AddEdge(src, new_node->GetInDataAnchor(0)) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][NdOpti][InsNd] Adding edge between node %s and node %s failed.",
                    src_node->GetName().c_str(), new_node->GetName().c_str());
    return FAILED;
  }

  if (ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), dst) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][NdOpti][InsNd] Adding edge between node %s and node %s failed.",
                    new_node->GetName().c_str(), dst_node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status NodeOptimizePassBase::CreateStridedRead(ge::NodePtr next_node,
                                               std::shared_ptr<ge::OpDesc> &strided_read_opdesc) {
  FE_MAKE_SHARED(
      strided_read_opdesc = std::make_shared<ge::OpDesc>(next_node->GetName() + "_" + STRIDEDREAD, STRIDEDREAD),
      return FAILED);
  FE_CHECK_NOTNULL(strided_read_opdesc);
  ge::GeTensorDesc input_desc;
  ge::GeTensorDesc output_desc;
  if (strided_read_opdesc->AddInputDesc("x", input_desc) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][NdOpti][CrtStrdRead] Failed to add input_desc for %s.",
                    strided_read_opdesc->GetName().c_str());
    return FAILED;
  }
  if (strided_read_opdesc->AddOutputDesc("y", output_desc) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][NdOpti][CrtStrdRead] Failed to add output_desc: %s.",
                    strided_read_opdesc->GetName().c_str());
    return FAILED;
  }
  (void)ge::AttrUtils::SetInt(strided_read_opdesc, STRIDE_ATTR_AXIS, 1);
  return SUCCESS;
}

Status NodeOptimizePassBase::CreateStridedWrite(ge::NodePtr prev_node,
                                                std::shared_ptr<ge::OpDesc> &strided_write_opdesc) {
  FE_MAKE_SHARED(
      strided_write_opdesc = std::make_shared<ge::OpDesc>(prev_node->GetName() + "_" + STRIDEDWRITE, STRIDEDWRITE),
      return FAILED);
  FE_CHECK_NOTNULL(strided_write_opdesc);
  std::vector<std::string> datadump_origin_op_names;
  if (!ge::AttrUtils::SetListStr(strided_write_opdesc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                 datadump_origin_op_names)) {
    REPORT_FE_ERROR("[GraphOpt][NdOpti][CrtStrdWrt] Failed to set the _datadump_original_op_names attribute.");
    return FAILED;
  }
  ge::GeTensorDesc input_desc;
  ge::GeTensorDesc output_desc;
  if (strided_write_opdesc->AddInputDesc("x", input_desc) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][NdOpti][CrtStrdWrt] Failed to add input_desc: %s.",
                    strided_write_opdesc->GetName().c_str());
    return FAILED;
  }
  if (strided_write_opdesc->AddOutputDesc("y", output_desc) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][NdOpti][CrtStrdWrt] Failed to add output_desc: %s.",
                    strided_write_opdesc->GetName().c_str());
    return FAILED;
  }
  (void)ge::AttrUtils::SetInt(strided_write_opdesc, STRIDE_ATTR_AXIS, 1);
  return SUCCESS;
}

void NodeOptimizePassBase::SetGeAttrForConcat(const ge::OpDescPtr &concat_op_desc_ptr,
                                              const size_t &dim_index) const {
  (void)ge::AttrUtils::SetBool(concat_op_desc_ptr, ge::ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetBool(concat_op_desc_ptr, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetBool(concat_op_desc_ptr, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetInt(concat_op_desc_ptr, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, dim_index);
}

Status NodeOptimizePassBase::JudgeOp(ge::NodePtr node) const {
  FE_CHECK(node == nullptr, FE_LOGI("Node is nullptr."), return FAILED);
  ge::InDataAnchorPtr src_in_data = node->GetInDataAnchor(0);
  FE_CHECK_NOTNULL(src_in_data);
  ge::OutDataAnchorPtr src = src_in_data->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(src);
  ge::OutDataAnchorPtr dst_out_data = node->GetOutDataAnchor(0);
  FE_CHECK_NOTNULL(dst_out_data);
  if (dst_out_data->GetPeerInDataAnchors().empty()) {
    return FAILED;
  }
  ge::InDataAnchorPtr dst = dst_out_data->GetPeerInDataAnchors().at(0);
  ge::NodePtr src_node = src->GetOwnerNode();
  FE_CHECK_NOTNULL(src_node);
  ge::NodePtr dst_node = dst->GetOwnerNode();
  FE_CHECK_NOTNULL(dst_node);

  if (node->GetOpDesc()->UpdateInputDesc(0, src_node->GetOpDesc()->GetOutputDesc(src->GetIdx())) != SUCCESS) {
    FE_LOGI("%s update of input_desc was unsuccessful.", node->GetName().c_str());
    return FAILED;
  }
  if (node->GetOpDesc()->UpdateOutputDesc(0, dst_node->GetOpDesc()->GetInputDesc(dst->GetIdx())) != SUCCESS) {
    FE_LOGI("%s update of output_desc was unsuccessful.", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

void NodeOptimizePassBase::SetGeAttrForSplit(const ge::OpDescPtr &split_op_desc_ptr,
                                             const size_t &dim_index) const {
  (void)ge::AttrUtils::SetBool(split_op_desc_ptr, ge::ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetBool(split_op_desc_ptr, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetBool(split_op_desc_ptr, ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)ge::AttrUtils::SetInt(split_op_desc_ptr, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, dim_index);
}

Status NodeOptimizePassBase::GetNC1HWC0Shape(ge::GeTensorDescPtr tensor_desc,
                                             const ge::DataType &quant_data_type) const {
  FE_CHECK_NOTNULL(tensor_desc);
  ge::Format origin_format = tensor_desc->GetOriginFormat();
  ge::GeShape origin_shape = tensor_desc->GetOriginShape();
  vector<int64_t> old_dim_vec = origin_shape.GetDims();
  if (old_dim_vec.empty()) {
    FE_LOGW("oldDimVec is empty!");
    return SUCCESS;
  }
  ge::DataType data_type = tensor_desc->GetOriginDataType();
  if (quant_data_type == ge::DT_INT8 || quant_data_type == ge::DT_INT4) {
    data_type = quant_data_type;
    tensor_desc->SetDataType(quant_data_type);
  }
  ge::Format new_format = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0,
      GetC0BitByDataType(tensor_desc->GetDataType())));
  tensor_desc->SetFormat(new_format);
  ge::GeShape new_shape;
  ShapeAndFormat output_shape_and_format_info = {origin_shape, new_shape, origin_format, new_format, data_type};
  (void)GetShapeAccordingToFormat(output_shape_and_format_info);
  tensor_desc->SetShape(new_shape);

  if ((tensor_desc->GetOriginDataType() == ge::DT_FLOAT || tensor_desc->GetOriginDataType() == ge::DT_FLOAT16) &&
      quant_data_type != ge::DT_INT8 && quant_data_type != ge::DT_INT4) {
    tensor_desc->SetDataType(ge::DT_FLOAT16);
  }
  return SUCCESS;
}

bool NodeOptimizePassBase::is_single_out_and_ref(const ge::NodePtr &node_ptr) const {
  ge::Node::Vistor<ge::OutDataAnchorPtr> all_out_data_anchors = node_ptr->GetAllOutDataAnchors();
  FE_LOGD("Node[%s]: all_out_data_anchors.size=[%zu].",
          node_ptr->GetName().c_str(),
          all_out_data_anchors.size());

  if (all_out_data_anchors.size() == 1) {
    ge::OutDataAnchorPtr out_data_anchor = node_ptr->GetOutDataAnchor(0);
    FE_CHECK_NOTNULL(out_data_anchor);
    FE_LOGD("Node[%s]: out_data_anchor->GetPeerInDataAnchors().size=[%zu].",
            node_ptr->GetName().c_str(),
            out_data_anchor->GetPeerInDataAnchors().size());
    return out_data_anchor->GetPeerInDataAnchors().size() == 1;
  }
  return false;
}
}  // namespace fe
