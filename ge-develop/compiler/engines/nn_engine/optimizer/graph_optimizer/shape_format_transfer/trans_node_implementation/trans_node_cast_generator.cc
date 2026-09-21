/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_cast_generator.h"
#include "common/fe_inner_error_codes.h"
#include "common/util/op_info_util.h"
#include "expand_dimension.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"

namespace fe {
TransNodeCastGenerator::TransNodeCastGenerator(FEOpsKernelInfoStorePtr fe_ops_store_ptr, TransInfoPtr trans_info_ptr)
    : TransNodeBaseGenerator(fe_ops_store_ptr, trans_info_ptr) {}

TransNodeCastGenerator::~TransNodeCastGenerator() {}

Status TransNodeCastGenerator::AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) {
  trans_info_ptr_ = trans_info_ptr;
  if (trans_info_ptr->src_out_data_type == trans_info_ptr->dst_in_data_type) {
    return SUCCESS;
  }

  /* After Inserting Cast Op, we will insert TransData op. So for Cast op, its
   input and output format can just
   * be the same as the output format of its source node.
   e.g. SOURCE(5HD-5HD, Fp16) -> CAST(5HD-5HD, FP16 to FP32) ->
   DESTINATION(4D-4D, FP32) After inserting TransData Op, it will become:
   SOURCE(5HD-5HD, Fp16) -> TRANSDATA(5HD to 4D, FP32) -> CAST(5HD-5HD, FP16 to
   FP32) -> DESTINATION(4D-4D, FP32)
   */
  return AddOpAndNode(fused_graph, trans_info_ptr->src_out_shape, trans_info_ptr->src_out_primary_format,
                      trans_info_ptr->src_out_sub_format, trans_info_ptr->dst_in_data_type);
}


Status TransNodeCastGenerator::AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::GeShape &shape,
                                            const ge::Format &primary_format, const int32_t &sub_format,
                                            const ge::DataType &dtype) {
  TransInfoPtr trans_info_ptr = trans_info_ptr_;
  ge::OpDescPtr op_desc_ptr = CreateBasicOpDescForTransNode(CAST);
  FE_CHECK_NOTNULL(op_desc_ptr);

  FE_LOGD("Create [%s] node between [%s] and [%s] successfully!", CAST.c_str(),
          trans_info_ptr->src_op_desc->GetName().c_str(), trans_info_ptr->dst_op_desc->GetName().c_str());
  (void)ge::AttrUtils::SetInt(op_desc_ptr, CAST_ATTR_SRCT, static_cast<int64_t>(trans_info_ptr->src_out_data_type));
  (void)ge::AttrUtils::SetInt(op_desc_ptr, CAST_ATTR_DSTT, static_cast<int64_t>(dtype));
  (void)ge::AttrUtils::SetInt(op_desc_ptr, CAST_ATTR_DST_TYPE, static_cast<int64_t>(dtype));
  (void)ge::AttrUtils::SetBool(op_desc_ptr, CAST_ATTR_TRUNCATE, false);

  int32_t input_c0_bit = GetC0BitValFromC0(trans_info_ptr_->src_out_c0_format);
  ge::Format input_format = static_cast<ge::Format>(ge::GetFormatFromSubAndC0(trans_info_ptr_->src_out_primary_format,
      trans_info_ptr_->src_out_sub_format, input_c0_bit));
  if (op_desc_ptr->AddInputDesc(CAST_INPUT_NAME, ge::GeTensorDesc(trans_info_ptr_->src_out_shape, input_format,
                                                                  trans_info_ptr->src_out_data_type)) != SUCCESS) {
    FE_LOGD("CreateReshapeOp by cast: op [RESHAPE] add input desc unsuccessful.");
    return FAILED;
  }

  auto output_format = static_cast<ge::Format>(ge::GetFormatFromSubAndC0(primary_format, sub_format, input_c0_bit));
  if (op_desc_ptr->AddOutputDesc(CAST_OUTPUT_NAME, ge::GeTensorDesc(shape, output_format, dtype)) != SUCCESS) {
    FE_LOGD("CreateReshapeOp by cast: op [RESHAPE] add output description unsuccessful.");
    return FAILED;
  }
  (void)SetTensorDescInfo(op_desc_ptr);

  // insert new op need add attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES
  // for data dump
  std::vector<std::string> original_names;
  (void)ge::AttrUtils::SetListStr(op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  std::unordered_set<ge::Format> reshape_formats = {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0_C04, ge::FORMAT_NDC1HWC0};
  if (reshape_formats.count(primary_format) > 0) {
    auto input_desc_ptr = op_desc_ptr->MutableInputDesc(0);
    FE_CHECK_NOTNULL(input_desc_ptr);
    auto output_desc_ptr = op_desc_ptr->MutableOutputDesc(0);
    FE_CHECK_NOTNULL(output_desc_ptr);
    int64_t input_reshape_type_mask =
            transformer::ExpandDimension::GenerateReshapeType(input_desc_ptr->GetOriginFormat(),
                                                              input_desc_ptr->GetFormat(),
                                                              input_desc_ptr->GetOriginShape().GetDimNum(),
                                                              trans_info_ptr->src_reshape_type);
    (void) ge::AttrUtils::SetInt(input_desc_ptr, ge::ATTR_NAME_RESHAPE_TYPE_MASK, input_reshape_type_mask);
    (void) ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_RESHAPE_TYPE_MASK, input_reshape_type_mask);
  }
  SetTensorRealDimCountAndNewShape(op_desc_ptr, {trans_info_ptr_->src_out_shape}, shape);
  SetNewShapeRange(op_desc_ptr, trans_info_ptr_->src_out_range, trans_info_ptr_->src_out_range);
  SetNodeParallelGroupId(op_desc_ptr, trans_info_ptr_->src_op_desc, trans_info_ptr_->dst_op_desc);
  if (AddEdgesAndFreshTransInfo(fused_graph, op_desc_ptr) != SUCCESS) {
    FE_LOGD("Add edge unsuccessful!");
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
