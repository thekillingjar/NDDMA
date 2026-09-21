/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_squeeze_v2_generator.h"
#include <sstream>
#include "common/fe_inner_error_codes.h"
#include "common/util/op_info_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"

namespace fe {
TransNodeSqueezeV2Generator::TransNodeSqueezeV2Generator(FEOpsKernelInfoStorePtr fe_ops_store_ptr,
                                                         TransInfoPtr trans_info_ptr)
    : TransNodeBaseGenerator(fe_ops_store_ptr, trans_info_ptr) {}

TransNodeSqueezeV2Generator::~TransNodeSqueezeV2Generator() {}

Status TransNodeSqueezeV2Generator::AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) {
  trans_info_ptr_ = trans_info_ptr;
  auto src_out_shape_dim = trans_info_ptr->src_out_shape.GetDimNum();
  auto dst_in_shape_dim = trans_info_ptr->dst_in_shape.GetDimNum();
  auto src_out_primary_format = trans_info_ptr->src_out_primary_format;
  auto iter = std::find(FE_ORIGIN_FORMAT_VECTOR.begin(), FE_ORIGIN_FORMAT_VECTOR.end(), src_out_primary_format);
  /*
   * Now we only support 4D to 1,2,3D. squeezeV2 op requires
   * programmer to elucidate how to shape will change.
   * e.g. To squeeze 4D(NCHW) -> 2D(NH) , we will remove dimension c,w to 2D op.
   * So 4D(NCHW) becomes 2D(NH).
   */
  bool is_dst_shape_dim_expected = false;
  if (Is3DFormat(src_out_primary_format)) {
    is_dst_shape_dim_expected = dst_in_shape_dim <= NCHW_DIMENSION_NUM;
  } else {
    is_dst_shape_dim_expected = dst_in_shape_dim <= LOW_DIMENSION_NUM_THD;
  }
  if (src_out_shape_dim > dst_in_shape_dim && is_dst_shape_dim_expected && dst_in_shape_dim > 0 &&
      iter != FE_ORIGIN_FORMAT_VECTOR.end()) {
    /* NCHW -> 1,2,3D */
    ge::GeShape new_shape = trans_info_ptr_->dst_in_shape;
    string new_shapestr = GetShapeDims(new_shape);
    FE_LOGD("The new shape after squeezing is %s.", new_shapestr.c_str());
    FE_LOGD("Source node is %s, dst node is %s.", trans_info_ptr_->src_op_desc->GetName().c_str(),
            trans_info_ptr_->dst_op_desc->GetName().c_str());
    return AddOpAndNode(fused_graph, new_shape, trans_info_ptr_->src_out_primary_format,
                        trans_info_ptr_->src_out_sub_format, trans_info_ptr->src_out_data_type);
  } else {
    FE_LOGI("[GraphOpt][Trans][Squeeze] cannot perform squeeze op from node [%s] to node [%s].",
            trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str());
    return SUCCESS;
  }
}

Status TransNodeSqueezeV2Generator::SetTensorDescInfo(ge::OpDescPtr &op_desc_ptr) const {
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

Status TransNodeSqueezeV2Generator::SetAttr(ge::OpDescPtr &op_desc_ptr) const {
  std::vector<int32_t> axis;
  int64_t reshape_type_mask = 0;
  (void)ge::AttrUtils::GetInt(trans_info_ptr_->src_out_tensor_desc_ptr, ge::ATTR_NAME_RESHAPE_TYPE_MASK,
                              reshape_type_mask);
  for (size_t i = 0; i < 8UL; i++) {
    int64_t bit_value = (reshape_type_mask >> i) & 1;
    if (bit_value == 1) {
      axis.emplace_back(i);
    }
  }
  // 2. set attr axis info
  if (!ge::AttrUtils::SetListInt(op_desc_ptr, AXIS_ATTR_NAME, axis)) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Squeeze] Failed to set squeeze op [%s] axis!", op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TransNodeSqueezeV2Generator::AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::GeShape &shape,
                                                 const ge::Format &primary_format, const int32_t &sub_format,
                                                 const ge::DataType &dtype) {
  ge::OpDescPtr op_desc_ptr = CreateBasicOpDescForTransNode(SQUEEZE_V2);
  FE_CHECK_NOTNULL(op_desc_ptr);

  FE_LOGD("Create [%s] node between [%s] and [%s] successfully!", SQUEEZE_V2.c_str(),
          trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str());

  auto input_format = static_cast<ge::Format>(
      ge::GetFormatFromSub(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_out_sub_format));
  if (op_desc_ptr->AddInputDesc(SQUEEZE_V2_INPUT_NAME, ge::GeTensorDesc(trans_info_ptr_->src_out_shape, input_format,
                                                                        trans_info_ptr_->src_out_data_type)) !=
      SUCCESS) {
    FE_LOGD("Create squeeze op: op [SQUEEZE_V2]: add input desc unsuccessful.");
    return FAILED;
  }

  auto output_format = static_cast<ge::Format>(ge::GetFormatFromSub(primary_format, sub_format));
  if (op_desc_ptr->AddOutputDesc(SQUEEZE_V2_OUTPUT_NAME, ge::GeTensorDesc(shape, output_format, dtype)) != SUCCESS) {
    FE_LOGD("Create squeeze op: op [SQUEEZE_V2]: add output desc unsuccessful.");
    return FAILED;
  }

  /* The output shape of squeeze depends on its attr value which name is "axis". */
  if (SetAttr(op_desc_ptr) != SUCCESS) {
    FE_LOGD("CreateReshapeOp: op [SQUEEZE_V2]: Set attr axis unsuccessful.");
    return FAILED;
  }

  (void)SetTensorDescInfo(op_desc_ptr);

  // insert new op need add attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES
  // for data dump
  std::vector<std::string> original_names;
  if (!ge::AttrUtils::SetListStr(op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names)) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Squeeze] Setting op[%s] attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES failed.",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  SetTensorRealDimCountAndNewShape(op_desc_ptr, {trans_info_ptr_->src_out_shape}, shape);
  SetNewShapeRange(op_desc_ptr, trans_info_ptr_->src_out_range, trans_info_ptr_->dst_in_range);
  SetNodeParallelGroupId(op_desc_ptr, trans_info_ptr_->src_op_desc, trans_info_ptr_->dst_op_desc);
  if (AddEdgesAndFreshTransInfo(fused_graph, op_desc_ptr) != SUCCESS) {
    FE_LOGD("Add edge unsuccessful!");
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
