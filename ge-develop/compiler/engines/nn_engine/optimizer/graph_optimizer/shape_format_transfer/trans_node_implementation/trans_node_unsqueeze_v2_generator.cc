/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_unsqueeze_v2_generator.h"
#include <sstream>
#include "common/fe_inner_error_codes.h"
#include "common/util/op_info_util.h"
#include "graph/utils/tensor_utils.h"
#include "expand_dimension.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"

namespace fe {
TransNodeUnsqueezeV2Generator::TransNodeUnsqueezeV2Generator(FEOpsKernelInfoStorePtr fe_ops_store_ptr,
                                                             TransInfoPtr trans_info_ptr)
    : TransNodeBaseGenerator(fe_ops_store_ptr, trans_info_ptr) {}

TransNodeUnsqueezeV2Generator::~TransNodeUnsqueezeV2Generator() {}

Status TransNodeUnsqueezeV2Generator::AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) {
  trans_info_ptr_ = trans_info_ptr;
  auto src_out_shape_dim = trans_info_ptr->src_out_shape.GetDimNum();
  auto dst_in_shape_dim = trans_info_ptr->dst_in_shape.GetDimNum();
  /*
   * Now we support 1,2,3D to 4D and 1,2,3,4D to 5D. UnsqueezeV2 op requires
   * programmer to elucidate how to shape will change.
   * e.g. To unsqueeze 2D(NH) -> 4D(NCHW), we will add dimension c,
   * w to 2D op and shape of c,w is 1.
   * So 2D(NH) becomes 4D(n,1(c),h,1(w).
   */
  auto dst_in_primary_format = trans_info_ptr->dst_in_primary_format;
  bool is_dst_shape_dim_expected = false;
  // support 3D format
  if (dst_in_primary_format == ge::FORMAT_NDC1HWC0 || Is3DFormat(dst_in_primary_format)) {
    is_dst_shape_dim_expected = src_out_shape_dim <= NCHW_DIMENSION_NUM;
  } else {
    is_dst_shape_dim_expected = src_out_shape_dim <= LOW_DIMENSION_NUM_THD;
  }
  if (src_out_shape_dim < dst_in_shape_dim && is_dst_shape_dim_expected && src_out_shape_dim > 0) {
    /* 1,2,3D -> NCHW */
    ge::GeShape new_shape;
    ExpandDimension(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_out_primary_format,
                    trans_info_ptr_->dst_reshape_type, trans_info_ptr_->src_out_shape, new_shape);
    FE_LOGD("After applying unsqueeze, the new shape is %s.", GetShapeDims(new_shape).c_str());
    FE_LOGD("Source node is %s, destination node is %s.", trans_info_ptr_->src_op_desc->GetName().c_str(),
            trans_info_ptr_->dst_op_desc->GetName().c_str());
    return AddOpAndNode(fused_graph, new_shape, trans_info_ptr_->src_out_primary_format,
                        trans_info_ptr_->src_out_sub_format, trans_info_ptr->src_out_data_type);
  } else {
    FE_LOGI("[GraphOpt][Trans][Unsqueeze] cannot unsqueeze from node [%s] to node [%s].",
            trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str());
    return SUCCESS;
  }
}

void TransNodeUnsqueezeV2Generator::GetNewShapeRange(vector<std::pair<int64_t, int64_t>> &new_range) const {
  ge::GeShape low_shape;
  ge::GeShape high_shape;
  for (auto &i : trans_info_ptr_->src_out_range) {
    low_shape.AppendDim(i.first);
    high_shape.AppendDim(i.second);
  }
  ExpandDimension(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_out_primary_format,
                  trans_info_ptr_->dst_reshape_type, low_shape);
  ExpandDimension(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_out_primary_format,
                  trans_info_ptr_->dst_reshape_type, high_shape);
  for (size_t i = 0; i < low_shape.GetDimNum(); i++) {
    new_range.emplace_back(std::make_pair(low_shape.GetDim(i), high_shape.GetDim(i)));
  }
}

Status TransNodeUnsqueezeV2Generator::SetAttr(const ge::OpDescPtr &op_desc_ptr) const {
  int64_t reshape_type_mask = 0;
  if (!transformer::ExpandDimension::GenerateReshapeType(trans_info_ptr_->src_out_original_format,
                                                         trans_info_ptr_->src_out_primary_format,
                                                         trans_info_ptr_->src_out_original_shape.GetDimNum(),
                                                         trans_info_ptr_->dst_reshape_type,
                                                         reshape_type_mask)) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Unsqueeze] Op[name=%s,type=%s]: generate reshape type mask failed.",
                    op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return FAILED;
  }

  std::vector<int32_t> axis;
  for (size_t i = 0; i < 8UL; i++) {
    int64_t bit_value = (static_cast<uint64_t>(reshape_type_mask) >> i) & 1;
    if (bit_value == 1) {
      axis.emplace_back(i);
    }
  }

  if (!ge::AttrUtils::SetListInt(op_desc_ptr, AXIS_ATTR_NAME, axis)) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Unsqueeze] Failed to set unsqueeze op [%s] axis!", op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TransNodeUnsqueezeV2Generator::SetTensorDescInfo(ge::OpDescPtr &op_desc_ptr) const {
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto input_tensor_0 = op_desc_ptr->MutableInputDesc(0);
  FE_CHECK_NOTNULL(input_tensor_0);

  input_tensor_0->SetOriginFormat(trans_info_ptr_->src_out_original_format);
  input_tensor_0->SetOriginShape(trans_info_ptr_->src_out_original_shape);

  auto output_tensor_0 = op_desc_ptr->MutableOutputDesc(0);
  FE_CHECK_NOTNULL(output_tensor_0);

  output_tensor_0->SetOriginFormat(trans_info_ptr_->src_out_original_format);
  output_tensor_0->SetOriginShape(trans_info_ptr_->src_out_original_shape);

  return SUCCESS;
}

Status TransNodeUnsqueezeV2Generator::AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::GeShape &shape,
                                                   const ge::Format &primary_format, const int32_t &sub_format,
                                                   const ge::DataType &dtype) {
  ge::OpDescPtr op_desc_ptr = CreateBasicOpDescForTransNode(UNSQUEEZE_V2);
  FE_CHECK_NOTNULL(op_desc_ptr);

  FE_LOGD("Create [%s] node between [%s] and [%s] successfully!", UNSQUEEZE_V2.c_str(),
          trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str());

  auto input_format = static_cast<ge::Format>(
      ge::GetFormatFromSub(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_out_sub_format));
  if (op_desc_ptr->AddInputDesc(UNSQUEEZE_V2_INPUT_NAME, ge::GeTensorDesc(trans_info_ptr_->src_out_shape, input_format,
                                                                          trans_info_ptr_->src_out_data_type)) !=
      SUCCESS) {
    FE_LOGD("CreateReshapeOp: Unable to add input description for op [UNSQUEEZE_V2].");
    return FAILED;
  }

  auto output_format = static_cast<ge::Format>(ge::GetFormatFromSub(primary_format, sub_format));
  if (op_desc_ptr->AddOutputDesc(UNSQUEEZE_V2_OUTPUT_NAME, ge::GeTensorDesc(shape, output_format, dtype)) != SUCCESS) {
    FE_LOGD("CreateReshapeOp: op [UNSQUEEZE_V2]: add output desc unsuccessful.");
    return FAILED;
  }

  /* The output shape of unsqueeze depends on its attr value which name is "axes". */
  if (SetAttr(op_desc_ptr) != SUCCESS) {
    FE_LOGD("CreateReshapeOp: op [UNSQUEEZE_V2]: Unable to set attribute axes.");
    return FAILED;
  }

  (void)SetTensorDescInfo(op_desc_ptr);

  // insert new op need add attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES
  // for data dump
  std::vector<std::string> original_names;
  if (!ge::AttrUtils::SetListStr(op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names)) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Unsqueeze] Failed to set op[%s] attribute ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  SetTensorRealDimCountAndNewShape(op_desc_ptr, {trans_info_ptr_->src_out_shape}, shape);
  vector<std::pair<int64_t, int64_t>> dst_in_range;
  GetNewShapeRange(dst_in_range);
  SetNewShapeRange(op_desc_ptr, trans_info_ptr_->src_out_range, dst_in_range);
  SetNodeParallelGroupId(op_desc_ptr, trans_info_ptr_->src_op_desc, trans_info_ptr_->dst_op_desc);
  if (AddEdgesAndFreshTransInfo(fused_graph, op_desc_ptr) != SUCCESS) {
    FE_LOGD("Add edge unsuccessful!");
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
