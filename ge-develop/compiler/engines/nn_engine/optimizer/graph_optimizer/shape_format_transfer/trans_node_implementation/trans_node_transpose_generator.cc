/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transpose_generator.h"
#include "common/fe_inner_error_codes.h"
#include "common/util/op_info_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"
#include "graph/operator_factory.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"

namespace fe {
namespace {
const string kInsertedByFe = "_inserted_by_fe";

static const trans_pose_map kTransposeMap = {
    /* dst format is FORMAT_C1HWNCoC0 */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), ge::FORMAT_HWCN},
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), ge::FORMAT_HWCN},
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), ge::FORMAT_HWCN},
    {CalcStrategyId(ge::FORMAT_C1HWNCoC0, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), ge::FORMAT_HWCN},
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_C1HWNCoC0, kStrategyIdExtDeft), ge::FORMAT_HWCN},
    /* dst format is FORMAT_NC1HWC0 */
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), ge::FORMAT_NCHW},
    {CalcStrategyId(ge::FORMAT_CHWN, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), ge::FORMAT_NCHW},
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), ge::FORMAT_MAX},
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), ge::FORMAT_MAX},
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), ge::FORMAT_MAX},
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), ge::FORMAT_MAX},
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, kStrategyIdExtDeft), ge::FORMAT_MAX},
    /* dst format is FORMAT_FRACTAL_Z */
    {CalcStrategyId(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, kStrategyIdExtThree), ge::FORMAT_HWCN},
    {CalcStrategyId(ge::FORMAT_NHWC, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), ge::FORMAT_NCHW},
    {CalcStrategyId(ge::FORMAT_CHWN, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), ge::FORMAT_NCHW},
    {CalcStrategyId(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), ge::FORMAT_MAX},
    {CalcStrategyId(ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), ge::FORMAT_MAX},
    {CalcStrategyId(ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), ge::FORMAT_MAX},
    {CalcStrategyId(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), ge::FORMAT_MAX},
    {CalcStrategyId(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_Z, kStrategyIdExtDeft), ge::FORMAT_MAX},
};

Status SetTransposeOrder(ge::Format input_format, ge::Format output_format,
                         const std::map<ge::Format, uint32_t>& format_index_map,
                         const std::vector<std::vector<std::vector<int64_t>>>& perm_value_vector,
                         ge::OpDescPtr op_desc_ptr) {
  auto input_iter = format_index_map.find(input_format);
  if (input_iter == format_index_map.end()) {
    FE_LOGW("Can not find input format %u in format map!", input_format);
    return SUCCESS;
  }
  auto output_iter = format_index_map.find(output_format);
  if (output_iter == format_index_map.end()) {
    FE_LOGW("Cannot find output format %u in format map!", output_format);
    return SUCCESS;
  }
  if (perm_value_vector.empty()) {
    FE_LOGW("Perm value vector is empty!");
    return SUCCESS;
  }

  if (input_iter->second >= perm_value_vector.size()) {
    FE_LOGW("Index %u exceeds the first dimension of the Perm vector.", input_iter->second);
    return SUCCESS;
  }

  if (perm_value_vector[input_iter->second].empty()) {
    FE_LOGW("Perm value vector %u is empty!", input_iter->second);
    return SUCCESS;
  }

  if (output_iter->second >= perm_value_vector[input_iter->second].size()) {
    FE_LOGW("Index %u exceeds the second dimension of the Perm vector %u.", output_iter->second, input_iter->second);
    return SUCCESS;
  }

  std::vector<int64_t> attr_order = perm_value_vector[input_iter->second][output_iter->second];
  // 2D format set PERMUTE_ATTR_ORDER
  if (attr_order.size() == NCHW_DIMENSION_NUM) {
    if (!(ge::AttrUtils::SetListInt(op_desc_ptr, ge::PERMUTE_ATTR_ORDER, attr_order))) {
      FE_LOGW("Failed to set transpose order for op (name [%s] type [%s]). And the input format is [%u].",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), input_format);
    }
  }
  if (!(ge::AttrUtils::SetListInt(op_desc_ptr, PERM, attr_order))) {
    FE_LOGW("Failed to set transpose perm for op (name [%s] type [%s]). And the input format is [%u].",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), input_format);
  }

  std::vector<int64_t> permShape(1, attr_order.size());
  ge::GeShape constInPerm = ge::GeShape(permShape);
  if (op_desc_ptr->AddInputDesc(PERM, ge::GeTensorDesc(constInPerm, ge::FORMAT_ND, ge::DT_INT64)) != SUCCESS) {
    FE_LOGD("CreateTransPoseOp: op [TransPose]: add input desc unsuccessful.");
    return FAILED;
  }

  return SUCCESS;
}

void SetTransposeOrder(ge::Format input_format, ge::Format output_format, ge::OpDescPtr op_desc_ptr) {
  if (TransNodeBaseGenerator::Is3DFormat(input_format) && TransNodeBaseGenerator::Is3DFormat(output_format)) {
    SetTransposeOrder(input_format, output_format, FORMAT_3D_INDEX_MAP, PERM_VALUE_3D_VECTOR, op_desc_ptr);
  } else {
    SetTransposeOrder(input_format, output_format, FORMAT_INDEX_MAP, PERM_VALUE_VECTOR, op_desc_ptr);
  }
}
} // namespace

TransNodeTransposeGenerator::TransNodeTransposeGenerator(FEOpsKernelInfoStorePtr fe_ops_store_ptr,
                                                         TransInfoPtr trans_info_ptr)
    : TransNodeBaseGenerator(fe_ops_store_ptr, trans_info_ptr) {}

TransNodeTransposeGenerator::~TransNodeTransposeGenerator() {}

Status TransNodeTransposeGenerator::AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) {
  trans_info_ptr_ = trans_info_ptr;
  TransformDimsWithFormat(true);
  /* output format and dtype of source node should be the same as permute op's
   * input format and dtype! */
  ge::Format out_format_new_node = trans_info_ptr->dst_in_primary_format;
  int32_t out_sub_format = trans_info_ptr->dst_in_sub_format;

  auto trans_pose = kTransposeMap.find(trans_info_ptr->strategy_id);
  if (trans_pose != kTransposeMap.end()) {
    if (trans_pose->second == ge::FORMAT_MAX) {
      /* In Other Cases, we will not tranform by transpose op */
      out_format_new_node = trans_info_ptr->src_out_primary_format;
      out_sub_format = trans_info_ptr->src_out_sub_format;
    } else {
      out_format_new_node = trans_pose->second;
      out_sub_format = 0;
    }
  }

  return AddOpAndNode(fused_graph, out_format_new_node, out_sub_format, trans_info_ptr->src_out_data_type);
}

void TransNodeTransposeGenerator::GetNewShape(ge::OpDescPtr &op_desc_ptr, ge::Format format, ge::DataType dtype,
                                              ge::GeShape &newshape) {
  ShapeAndFormat output_shape_and_format_info = {trans_info_ptr_->src_out_shape,
                                                 newshape,
                                                 trans_info_ptr_->src_out_primary_format,
                                                 format,
                                                 dtype};
  /* Update output shape of transpose based on it's imple type */
  (void)GetShapeAccordingToFormat(output_shape_and_format_info);

  SetTensorRealDimCountAndNewShape(op_desc_ptr, {trans_info_ptr_->src_out_shape}, newshape);
  SetNewShapeRange(op_desc_ptr, trans_info_ptr_->src_out_range, trans_info_ptr_->dst_in_range);
}

Status TransNodeTransposeGenerator::AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::Format &primary_format,
                                                 const int32_t &sub_format, const ge::DataType &dtype) {
  if (trans_info_ptr_->src_out_primary_format == primary_format) {
    FE_LOGD("Format %u of %s is equal to format %u of %s.", trans_info_ptr_->src_out_primary_format,
            trans_info_ptr_->src_op_desc->GetName().c_str(), primary_format,
            trans_info_ptr_->dst_op_desc->GetName().c_str());
    return SUCCESS;
  }
  ge::OpDescPtr op_desc_ptr = CreateBasicOpDescForTransNode(TRANSPOSE);
  FE_CHECK_NOTNULL(op_desc_ptr);

  FE_LOGD("Create [%s] node between [%s] and [%s] successfully!", TRANSPOSE.c_str(),
          trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str());

  ge::GeShape newshape = trans_info_ptr_->src_out_shape;
  ge::Format input_format = static_cast<ge::Format>(
      ge::GetFormatFromSub(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_out_sub_format));
  if (op_desc_ptr->AddInputDesc(TRANSPOSE_INPUT_NAME, ge::GeTensorDesc(trans_info_ptr_->src_out_shape, input_format,
                                                                       trans_info_ptr_->src_out_data_type)) != SUCCESS) {
    FE_LOGD("CreateTransPoseOp: op [TransPose]: add input desc unsuccessful.");
    return FAILED;
  }

  ge::Format output_format = static_cast<ge::Format>(ge::GetFormatFromSub(primary_format, sub_format));
  if (op_desc_ptr->AddOutputDesc(TRANSPOSE_OUTPUT_NAME, ge::GeTensorDesc(newshape, output_format, dtype)) != SUCCESS) {
    FE_LOGD("CreateTransPoseOp: op [TransPose]: add output desc unsuccessful.");
    return FAILED;
  }
  /* Set required attr perm first */
  SetTransposeOrder(trans_info_ptr_->src_out_primary_format, primary_format, op_desc_ptr);
  int64_t inserted_by_fe = 1;
  ge::AttrUtils::SetInt(op_desc_ptr, kInsertedByFe, inserted_by_fe);
  (void)SetTensorDescInfo(op_desc_ptr);
  if (!TransNodeCheckSupportedByFormatTune(fused_graph, op_desc_ptr)) {
    FE_LOGW("[GraphOpt][Trans][TransPose] Format %u to %u of op %s to %s is not supported by %s!",
            trans_info_ptr_->src_out_primary_format, trans_info_ptr_->dst_in_primary_format,
            trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str(),
            op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  // insert new op need add attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES for data dump
  std::vector<std::string> original_names;
  if (!ge::AttrUtils::SetListStr(op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names)) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][AddOpAndNd] Node[%s]: Failed to set attribute datadump_original_op_names.",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }

  GetNewShape(op_desc_ptr, primary_format, dtype, newshape);
  SetNodeParallelGroupId(op_desc_ptr, trans_info_ptr_->src_op_desc, trans_info_ptr_->dst_op_desc);
  if (AddEdgesAndFreshTransInfo(fused_graph, op_desc_ptr) != SUCCESS) {
    FE_LOGD("Adding edge unsuccessful.");
    return FAILED;
  }

  return SUCCESS;
}

Status TransNodeTransposeGenerator::AddNecessaryPeerNodes(ge::ComputeGraph &fused_graph, ge::NodePtr new_node) const {
  (void)fused_graph;
  ge::OpDescPtr op_desc_ptr = new_node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);

  std::vector<int64_t> perm;
  ge::AttrUtils::GetListInt(op_desc_ptr, PERM, perm);
  op_desc_ptr->DelAttr(PERM);

  auto permInputDesc = op_desc_ptr->GetInputDesc(kTransposeInputPerm);
  ge::GeTensorPtr outTensor = nullptr;
  FE_MAKE_SHARED(outTensor = std::make_shared<ge::GeTensor>(permInputDesc), return FAILED);
  outTensor->SetData(reinterpret_cast<uint8_t *>(perm.data()), perm.size() * sizeof(int64_t));

  if (ge::OpDescUtils::AddConstOpToAnchor(new_node->GetInDataAnchor(kTransposeInputPerm), outTensor) != SUCCESS) {
    FE_LOGE("Failed to add perm for transpose %s", new_node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status TransNodeTransposeGenerator::SetTensorDescInfo(ge::OpDescPtr &op_desc_ptr) const {
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto input_tensor_0 = op_desc_ptr->MutableInputDesc(0);
  FE_CHECK_NOTNULL(input_tensor_0);
  auto input_tensor_1 = op_desc_ptr->MutableInputDesc(1);
  FE_CHECK_NOTNULL(input_tensor_1);

  input_tensor_0->SetOriginFormat(trans_info_ptr_->src_out_original_format);
  input_tensor_0->SetOriginShape(trans_info_ptr_->src_out_original_shape);
  input_tensor_1->SetOriginFormat(static_cast<ge::Format>(ge::GetPrimaryFormat(input_tensor_1->GetFormat())));
  input_tensor_1->SetOriginShape(input_tensor_1->GetShape());

  bool is_from_const_op = false;
  (void)ge::AttrUtils::GetBool(trans_info_ptr_->src_op_desc, kIsComeFromConstOp, is_from_const_op);
  bool is_src_from_const_op = is_from_const_op || trans_info_ptr_->src_op_desc->GetType() == CONSTANT ||
          trans_info_ptr_->src_op_desc->GetType() == CONSTANTOP;
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kIsComeFromConstOp, is_src_from_const_op);
  for (auto output_tensor : op_desc_ptr->GetAllOutputsDescPtr()) {
    output_tensor->SetOriginFormat(trans_info_ptr_->src_out_original_format);
    output_tensor->SetOriginShape(trans_info_ptr_->src_out_original_shape);
    if (!is_src_from_const_op) {
      GraphPassUtil::SetOutputDescAttr(trans_info_ptr_->src_out_tensor_desc_ptr,
                                       static_cast<int64_t>(trans_info_ptr_->src_anchor->GetIdx()),
                                       trans_info_ptr_->src_op_desc,
                                       output_tensor);
    }
  }
  return SUCCESS;
}
}  // namespace fe
