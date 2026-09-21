/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transdata_generator.h"
#include <memory>
#include <string>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "graph/utils/tensor_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "expand_dimension.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"

namespace fe {
namespace {
static const std::string kStageAddOpAndNd = "[GraphOptJdgInst][ShapeTrans][AddOpAndNd]";
static const trans_data_map kTransdataMap = {
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NHWC, kStrategyIdExtDeft), ge::FORMAT_NCHW},
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_CHWN, kStrategyIdExtDeft), ge::FORMAT_NCHW},
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), ge::FORMAT_NCHW},
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NCHW, kStrategyIdExtThree), ge::FORMAT_HWCN},
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_HWCN, kStrategyIdExtDeft), ge::FORMAT_NCHW},
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), ge::FORMAT_NCHW},
};

bool IsNeedToUpdateOutputShapeByOriginalShape(TransInfoPtr trans_info_ptr) {
  bool is_common_format = false;
  for (auto iter : FE_ORIGIN_FORMAT_VECTOR) {
    if (iter == trans_info_ptr->src_out_primary_format) {
      is_common_format = true;
      break;
    }
  }
  if (is_common_format) {
    return false;
  } else if ((trans_info_ptr->src_out_primary_format == ge::FORMAT_NC1HWC0 ||
              trans_info_ptr->src_out_primary_format == ge::FORMAT_NC1HWC0_C04) &&
             (trans_info_ptr->src_out_original_shape.GetDimNum() > DIM_DEFAULT_SIZE ||
              trans_info_ptr->src_out_original_shape.GetDimNum() == 0)) {
    /* Src out format is 5HD and it's original shape is already 5HD or empty,
     * we don't need to calculate the new shape by original shape. */
    return false;
  } else {
    return true;
  }
}
} // namespace

TransNodeTransdataGenerator::TransNodeTransdataGenerator(FEOpsKernelInfoStorePtr fe_ops_store_ptr,
                                                         TransInfoPtr trans_info_ptr)
    : TransNodeBaseGenerator(fe_ops_store_ptr, trans_info_ptr) {}
TransNodeTransdataGenerator::~TransNodeTransdataGenerator() {}

Status TransNodeTransdataGenerator::AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) {
  /* The two anchors of inserting TransData op should be same in data type.
   * If it is 4D to 5HD, 4D format should be NCHW.
   * Thas requires we do Permute and Cast before TransData. */
  trans_info_ptr_ = trans_info_ptr;

  TransformDimsWithFormat(true);
  ge::Format out_format_new_node = trans_info_ptr->dst_in_primary_format;
  auto out_sub_format = trans_info_ptr->dst_in_sub_format;

  auto trans_data = kTransdataMap.find(trans_info_ptr->strategy_id);
  if (trans_data != kTransdataMap.end()) {
    out_format_new_node = trans_data->second;
    out_sub_format = 0;
  } else if (trans_info_ptr->src_out_primary_format == ge::FORMAT_FRACTAL_NZ) {
    /* To ensure that Format Nz will only be tranformed to
     * its original format */
    /* For Nz to NCHW or NHWC or HWCN, we set the output
     * format as ND, to let the trans-nodes support more scenario and
     * let two trans-nodes of ND->Nz and Nz->ND merge themselves.
     * Because NHWC->NZ and NZ->HWCN will not merge automatically althought
     * their shape is the same */
    out_format_new_node = ge::FORMAT_ND;
    out_sub_format = 0;
  } else if (trans_info_ptr->dst_in_primary_format == ge::FORMAT_FRACTAL_NZ) {
    /* For NCHW or NHWC or HWCN to Nz, we set the input format of trans-nodes
     * as ND, to let the trans-nodes support more scenario and
     * let two trans-nodes of ND->Nz and Nz->ND merge themselves.
     * Because NHWC->NZ and NZ->HWCN will not merge automatically althought
     * their shape is the same */
    trans_info_ptr->src_out_primary_format = ge::FORMAT_ND;
  } else if (trans_info_ptr->src_out_primary_format == ge::FORMAT_C1HWNCoC0) {
    out_format_new_node = ge::FORMAT_HWCN;
    out_sub_format = 0;
  }

  return AddOpAndNode(fused_graph, ge::GeShape(), out_format_new_node, out_sub_format,
                      trans_info_ptr->src_out_data_type);
}

/* Get input and output shape of op transdata. */
Status TransNodeTransdataGenerator::GetShapeOfTransdata(const ge::OpDescPtr op_desc_ptr, ge::GeShape &new_in_shape,
                                                        ge::GeShape &new_out_shape,
                                                        vector<std::pair<int64_t, int64_t>> &new_in_range,
                                                        vector<std::pair<int64_t, int64_t>> &new_out_range,
                                                        const ge::Format &output_format,
                                                        const ge::DataType &output_dtype) {
  /* CCE transdata node will be ignored in our progress,
     but GE constant folding depends on transdata's shape,
     to resolve this, set imply_type as TBE. */
  ge::GeShape temp_src_out_shape;
  ge::Format temp_src_out_format;
  int64_t group_value = GROUPS_DEFAULT_VALUE;
  if (trans_info_ptr_->src_out_primary_format == ge::FORMAT_FRACTAL_Z) {
    /* Update shape of input of transdata base on its imply type.
     * For Fragz, we will change the shape based on its
     * father's original shape */
    temp_src_out_shape = trans_info_ptr_->src_out_original_shape;
    temp_src_out_format = trans_info_ptr_->src_out_original_format;
    ExpandDimension(temp_src_out_format, trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_reshape_type,
                    temp_src_out_shape);
    group_value = trans_info_ptr_->src_out_sub_format;
  } else {
    temp_src_out_shape = trans_info_ptr_->src_out_shape;
    temp_src_out_format = trans_info_ptr_->src_out_primary_format;
  }

  CalcShapeExtraAttr extra_attr = {trans_info_ptr_->hidden_size, trans_info_ptr_->input_size,
                                   trans_info_ptr_->state_size};
  int32_t c0_bit = GetC0BitValFromC0(trans_info_ptr_->src_out_c0_format);
  ge::Format c0_format = static_cast<ge::Format>(ge::GetFormatFromC0(trans_info_ptr_->src_out_primary_format, c0_bit));
  ShapeAndFormat input_shape_and_format_info = {temp_src_out_shape, new_in_shape, temp_src_out_format,
                                                c0_format,
                                                trans_info_ptr_->src_out_data_type,
                                                group_value,
                                                extra_attr};
  (void)GetShapeAccordingToFormat(input_shape_and_format_info);
  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr)) {
    new_in_range = trans_info_ptr_->src_out_range;
  }
  /* If 5D to 4D we use source node's original shape as final
   * 4D shape, and do transpose if necessary.
   * If source node's original shape is 5D, than we just
   * use formula {C = C1 * C0, N,H,W remain the same as 5D} to get the new 4D
   * shape */
  if (IsNeedToUpdateOutputShapeByOriginalShape(trans_info_ptr_) &&
      !trans_info_ptr_->src_out_shape.IsUnknownDimNum()) {
    /* Update output shape of transdata based on original shape and
     * original format of its father node */
    temp_src_out_shape = trans_info_ptr_->src_out_original_shape;
    temp_src_out_format = trans_info_ptr_->src_out_original_format;
    ExpandDimension(temp_src_out_format, trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_reshape_type,
                    temp_src_out_shape);
  } else {
    /* Update output shape of transdata based on its input. */
    temp_src_out_shape = new_in_shape;
    temp_src_out_format = trans_info_ptr_->src_out_primary_format;
  }
  int64_t dst_group_value = trans_info_ptr_->dst_in_sub_format;
  c0_bit = GetC0BitValFromC0(trans_info_ptr_->dst_in_c0_format);
  c0_format = static_cast<ge::Format>(ge::GetFormatFromC0(output_format, c0_bit));
  ShapeAndFormat output_shape_and_format_info = {temp_src_out_shape, new_out_shape, temp_src_out_format, c0_format,
                                                 output_dtype,       dst_group_value,     extra_attr};
  (void)GetShapeAccordingToFormat(output_shape_and_format_info);
  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr)) {
    new_out_range = trans_info_ptr_->dst_in_range;
  }
  return SUCCESS;
}

void TransNodeTransdataGenerator::SetAttr(const TransInfoPtr &trans_info_ptr, const ge::Format &primary_format,
                                          const int32_t &sub_format, ge::OpDescPtr &op_desc_ptr) const {
  (void)ge::AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_INPUT_FORMAT,
                              static_cast<int64_t>(trans_info_ptr->src_out_primary_format));
  (void)ge::AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_OUTPUT_FORMAT, static_cast<int64_t>(primary_format));
  string src_format = ge::TypeUtils::FormatToSerialString(trans_info_ptr->src_out_primary_format);
  string dst_format = ge::TypeUtils::FormatToSerialString(primary_format);
  (void)ge::AttrUtils::SetStr(op_desc_ptr, ATTR_NAME_SRC_FORMAT, src_format);
  (void)ge::AttrUtils::SetStr(op_desc_ptr, ATTR_NAME_DST_FORMAT, dst_format);

  // src
  if ((std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(),
                 trans_info_ptr->src_out_primary_format) != FE_GROUP_RELA_FORMAT_VECTOR.end()) ||
      (std::find(FE_C04_FORMAT_VECTOR.begin(), FE_C04_FORMAT_VECTOR.end(),
                 trans_info_ptr->src_out_primary_format) != FE_C04_FORMAT_VECTOR.end())) {
    FE_LOGD("Transdata[%s]: src_out_primary_format=%s, src_out_sub_format=%d.", op_desc_ptr->GetName().c_str(),
            FormatToStr(trans_info_ptr->src_out_primary_format).c_str(), trans_info_ptr->src_out_sub_format);
    FE_LOGD("Transdata[%s]: dst_out_primary_format=%s, dst_out_sub_format=%d.", op_desc_ptr->GetName().c_str(),
            FormatToStr(primary_format).c_str(), sub_format);
    int64_t group = trans_info_ptr->src_out_sub_format;
    if (group < GROUPS_DEFAULT_VALUE) {
      group = GROUPS_DEFAULT_VALUE;
    }

    (void)ge::AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_GROUPS, group);
    FE_LOGD("Set groups attribute[%ld] for transdata node[%s].", group, op_desc_ptr->GetName().c_str());
  }

  // dst
  if ((std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(), primary_format) !=
       FE_GROUP_RELA_FORMAT_VECTOR.end()) ||
      (std::find(FE_C04_FORMAT_VECTOR.begin(), FE_C04_FORMAT_VECTOR.end(), primary_format) !=
       FE_C04_FORMAT_VECTOR.end())) {
    FE_LOGD("Transdata[%s]: src_out_primary_format=%s, src_out_sub_format=%d.", op_desc_ptr->GetName().c_str(),
            FormatToStr(trans_info_ptr->src_out_primary_format).c_str(), trans_info_ptr->src_out_sub_format);
    FE_LOGD("Transdata[%s]: dst_out_primary_format=%s, dst_out_sub_format=%d.", op_desc_ptr->GetName().c_str(),
            FormatToStr(primary_format).c_str(), sub_format);
    int64_t group = sub_format;
    if (group < GROUPS_DEFAULT_VALUE) {
      group = GROUPS_DEFAULT_VALUE;
    }

    (void)ge::AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_GROUPS, group);
    FE_LOGD("Set groups attribute[%ld] for transdata node[%s].", group, op_desc_ptr->GetName().c_str());
  }
  // is src or dst op is dump able and format has been changed, this transdata can not be merged or fused
  bool is_src_dump_able = IsDumpableOp(trans_info_ptr->src_op_desc) &&
                          trans_info_ptr->insertion_mode == INSERT_TRANS_NODE_BY_ORIGINAL_FORMAT_END &&
                          (trans_info_ptr->src_out_original_format != trans_info_ptr->src_out_primary_format);
  bool is_dst_dump_able = IsDumpableOp(trans_info_ptr->dst_op_desc) &&
                          trans_info_ptr->insertion_mode == INSERT_TRANS_NODE_BY_ORIGINAL_FORMAT_FRONT &&
                          (trans_info_ptr->dst_in_original_format != trans_info_ptr->dst_in_primary_format);
  if (is_src_dump_able || is_dst_dump_able) {
    (void)ge::AttrUtils::SetBool(op_desc_ptr, kAttrDumpAble, true);
  }
}  // namespace fe

Status TransNodeTransdataGenerator::AddAndSetTensor(const ge::GeShape &shape, const ge::Format &primary_format,
                                                    const int32_t &sub_format, const ge::DataType &dtype,
                                                    ge::OpDescPtr &op_desc_ptr) {
  int32_t input_c0_bit = GetC0BitValFromC0(trans_info_ptr_->src_out_c0_format);
  (void)ge::AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_INPUT_SRC_FORMAT_INT, static_cast<int64_t>(input_c0_bit));
  ge::Format input_format = static_cast<ge::Format>(ge::GetFormatFromSubAndC0(trans_info_ptr_->src_out_primary_format,
      trans_info_ptr_->src_out_sub_format, input_c0_bit));
  ge::GeTensorDesc input =
      ge::GeTensorDesc(trans_info_ptr_->src_out_shape, input_format, trans_info_ptr_->src_out_data_type);
  if (op_desc_ptr->AddInputDesc(TRANSDATA_INPUT_NAME, input) != SUCCESS) {
    FE_LOGD("CreateTransdataOp: op [Transdata]: add input desc unsuccessful.");
    return FAILED;
  }

  int32_t output_c0_bit = GetC0BitValFromC0(trans_info_ptr_->dst_in_c0_format);
  (void)ge::AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_OUTPUT_SRC_FORMAT_INT, static_cast<int64_t>(output_c0_bit));
  ge::Format output_format = static_cast<ge::Format>(ge::GetFormatFromSubAndC0(
      primary_format, sub_format, output_c0_bit));
  if (op_desc_ptr->AddOutputDesc(TRANSDATA_OUTPUT_NAME, ge::GeTensorDesc(shape, output_format, dtype)) != SUCCESS) {
    FE_LOGD("CreateTransdataOp: op [Transdata]: unable to add output desc.");
    return FAILED;
  }

  (void)SetTensorDescInfo(op_desc_ptr);
  return SUCCESS;
}

Status TransNodeTransdataGenerator::AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::GeShape &shape,
                                                 const ge::Format &out_primary_format, const int32_t &out_sub_format,
                                                 const ge::DataType &dtype) {
  TransInfoPtr trans_info_ptr = trans_info_ptr_;
  ge::OpDescPtr op_desc_ptr = CreateBasicOpDescForTransNode(TRANSDATA);
  FE_CHECK_NOTNULL(op_desc_ptr);

  FE_LOGD("Create [%s] node between [%s] and [%s] successfully!", TRANSDATA.c_str(),
          trans_info_ptr->src_op_desc->GetName().c_str(), trans_info_ptr->dst_op_desc->GetName().c_str());

  SetAttr(trans_info_ptr, out_primary_format, out_sub_format, op_desc_ptr);

  if (AddAndSetTensor(shape, out_primary_format, out_sub_format, dtype, op_desc_ptr) != SUCCESS) {
    return FAILED;
  }
  /*  If we can not find specific transdata, we just skip this operation
   * and keep running. */
  /* There are no trans-op for FRACTAL_Z_3D and FORMAT_FRACTAL_Z_3D_TRANSPOSE,
   * so if we encouter them, we just skip the check accuracy support and
   * still insert that kind of TransData into the graph and GE will then
   * fuse them will const node. */
  bool always_insert_flag = trans_info_ptr->src_out_primary_format == ge::FORMAT_FRACTAL_NZ ||
                            trans_info_ptr->dst_in_primary_format == ge::FORMAT_FRACTAL_NZ ||
                            (dtype != ge::DT_INT32 && dtype != ge::DT_DOUBLE);

  /*  If we can not find specific transdata, we just skip this operation
   * and keep running. */
  if (!always_insert_flag && !TransNodeCheckAccuracySupported(op_desc_ptr, true)) {
    FE_LOGW("[GraphOpt][Trans][TransData] Format conversion from %u to %u for op %s to %s is not supported by %s!",
            trans_info_ptr_->src_out_primary_format, trans_info_ptr_->dst_in_primary_format,
            trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str(),
            op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }

  if (!TransNodeCheckSupportedByFormatTune(fused_graph, op_desc_ptr)) {
    FE_LOGW("[GraphOpt][Trans][TransData] Format conversion from %u to %u for op %s to %s is not supported by %s!",
            trans_info_ptr_->src_out_primary_format, trans_info_ptr_->dst_in_primary_format,
            trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str(),
            op_desc_ptr->GetName().c_str());
    return FAILED;
  }

  // insert new op need add attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES
  // for data dump
  std::vector<std::string> original_names;
  if (!ge::AttrUtils::SetListStr(op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names)) {
    REPORT_FE_ERROR(
        "[GraphOptJdgInst][ShapeTrans][AddOpAndNd] Node[%s]: fail to set attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.",
        op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  ge::GeShape new_out_shape;
  ge::GeShape new_in_shape;
  vector<std::pair<int64_t, int64_t>> new_in_range;
  vector<std::pair<int64_t, int64_t>> new_out_range;
  GetShapeOfTransdata(op_desc_ptr, new_in_shape, new_out_shape, new_in_range, new_out_range, out_primary_format, dtype);
  SetTensorRealDimCountAndNewShape(op_desc_ptr, {new_in_shape}, new_out_shape);
  auto input_desc_ptr = op_desc_ptr->MutableInputDesc(0);
  FE_CHECK_NOTNULL(input_desc_ptr);
  auto output_desc_ptr = op_desc_ptr->MutableOutputDesc(0);
  FE_CHECK_NOTNULL(output_desc_ptr);
  int64_t input_reshape_type_mask = 0;
  if (!transformer::ExpandDimension::GenerateReshapeType(input_desc_ptr->GetOriginFormat(),
                                                         input_desc_ptr->GetFormat(),
                                                         input_desc_ptr->GetOriginShape().GetDimNum(),
                                                         trans_info_ptr->src_reshape_type,
                                                         input_reshape_type_mask)) {
    REPORT_FE_ERROR("%s Node[%s, %s]: failed to generate reshape type mask for input.", kStageAddOpAndNd.c_str(),
                    op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return FAILED;
  }
  (void)ge::AttrUtils::SetInt(input_desc_ptr, ge::ATTR_NAME_RESHAPE_TYPE_MASK, input_reshape_type_mask);
  int64_t output_reshape_type_mask = 0;
  if (!transformer::ExpandDimension::GenerateReshapeType(output_desc_ptr->GetOriginFormat(),
                                                         output_desc_ptr->GetFormat(),
                                                         output_desc_ptr->GetOriginShape().GetDimNum(),
                                                         trans_info_ptr->src_reshape_type,
                                                         output_reshape_type_mask)) {
    REPORT_FE_ERROR("%s Node[%s, %s]: failed to generate reshape type mask for output.", kStageAddOpAndNd.c_str(),
                    op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return FAILED;
  }
  (void)ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_RESHAPE_TYPE_MASK, output_reshape_type_mask);
  SetNewShapeRange(op_desc_ptr, new_in_range, new_out_range);
  SetNodeParallelGroupId(op_desc_ptr, trans_info_ptr_->src_op_desc, trans_info_ptr_->dst_op_desc);
  if (AddEdgesAndFreshTransInfo(fused_graph, op_desc_ptr) != SUCCESS) {
    FE_LOGD("Add edge unsuccessful!");
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
