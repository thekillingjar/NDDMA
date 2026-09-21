/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_reshape_generator.h"
#include <sstream>
#include "common/fe_inner_error_codes.h"
#include "common/util/op_info_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"

namespace fe {
namespace {
Status GetReshapeSizeAndFirstDim(ge::Format src_out_primary_format, const ge::GeShape &src_out_shape,
                                 size_t &reshape_size, int64_t &first_dim) {
  if (src_out_primary_format == ge::FORMAT_FRACTAL_Z_3D) {
    reshape_size = 7;  // shape size of C1DHWNCoC0
  } else {
    reshape_size = 6;  // shape size of C1HWNCoC0
  }

  if (src_out_shape.GetDimNum() != reshape_size) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Reshape] The shape size is not equals 6 or 7, can not reshape.");
    return FAILED;
  }

  if (src_out_primary_format == ge::FORMAT_FRACTAL_Z_3D) {
    first_dim = src_out_shape.GetDim(C1DHWNCoC0_DIM_C1) * src_out_shape.GetDim(C1DHWNCoC0_DIM_D) *
                src_out_shape.GetDim(C1DHWNCoC0_DIM_H) * src_out_shape.GetDim(C1DHWNCoC0_DIM_W);
  } else {
    first_dim = src_out_shape.GetDim(C1HWNCoC0_DIM_C1) * src_out_shape.GetDim(C1HWNCoC0_DIM_H) *
                src_out_shape.GetDim(C1HWNCoC0_DIM_W);
  }
  return SUCCESS;
}
} // namespace

const int kNewNodeInputSizeLimit = 2;

TransNodeReshapeGenerator::TransNodeReshapeGenerator(FEOpsKernelInfoStorePtr fe_ops_store_ptr,
                                                     TransInfoPtr trans_info_ptr)
    : TransNodeBaseGenerator(fe_ops_store_ptr, trans_info_ptr) {}

TransNodeReshapeGenerator::~TransNodeReshapeGenerator() {}

Status TransNodeReshapeGenerator::AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) {
  trans_info_ptr_ = trans_info_ptr;
  auto src_out_shape_dim = trans_info_ptr->src_out_shape.GetDimNum();
  auto dst_in_shape_dim = trans_info_ptr->dst_in_shape.GetDimNum();
  auto src_out_primary_format = trans_info_ptr->src_out_primary_format;
  auto iter = std::find(FE_ORIGIN_FORMAT_VECTOR.begin(), FE_ORIGIN_FORMAT_VECTOR.end(), src_out_primary_format);
  /* Now we only support 1,2,3D to 4D and 4D to 1,2,3D. Reshape op requires
   * programmer to elucidate how to shape will change.
   * e.g. To Reshape 2D(nh) -> 4D(NCHW), we will add dimension c,
   * w to 2D op and shape of c,w is 1.
   * So 2D(nh) becomes 4D(n,1(c),h,1(w). */
  if ((src_out_shape_dim > dst_in_shape_dim && dst_in_shape_dim <= LOW_DIMENSION_NUM_THD && dst_in_shape_dim > 0 &&
       iter != FE_ORIGIN_FORMAT_VECTOR.end()) ||
      (src_out_shape_dim < dst_in_shape_dim && src_out_shape_dim <= LOW_DIMENSION_NUM_THD)) {
    /* NCHW -> 1,2,3D */
    /* 1,2,3D -> NCHW */
    bool increasing_flag = (dst_in_shape_dim > src_out_shape_dim);
    ge::GeShape new_shape;
    if (increasing_flag) {
      ExpandDimension(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_out_primary_format,
                      trans_info_ptr_->dst_reshape_type, trans_info_ptr_->src_out_shape, new_shape);
    } else {
      new_shape = trans_info_ptr_->dst_in_shape;
    }
    string new_shapestr = GetShapeDims(new_shape);
    FE_LOGD("The new shape after reshaping is %s.", GetShapeDims(new_shape).c_str());
    FE_LOGD("Source node is %s, dst node is %s.", trans_info_ptr_->src_op_desc->GetName().c_str(),
            trans_info_ptr_->dst_op_desc->GetName().c_str());
    return AddOpAndNode(fused_graph, new_shape, trans_info_ptr_->src_out_primary_format,
                        trans_info_ptr_->src_out_sub_format, trans_info_ptr->src_out_data_type);
  } else {
    auto src_op_pattern = trans_info_ptr->src_op_pattern;
    auto dst_op_pattern = trans_info_ptr->dst_op_pattern;

    // get previous out format and next in format
    src_out_primary_format = trans_info_ptr->init_src_out_primary_format;
    auto dst_in_primary_format = trans_info_ptr->init_dst_in_primary_format;
    bool src_fz_flag = src_out_primary_format == ge::FORMAT_FRACTAL_Z ||
                       src_out_primary_format == ge::FORMAT_FRACTAL_Z_3D;

    bool dst_fz_flag = dst_in_primary_format == ge::FORMAT_FRACTAL_Z ||
                       dst_in_primary_format == ge::FORMAT_FRACTAL_Z_3D;

    bool src_broadcast_reshape_flag =
        ((src_op_pattern == OP_PATTERN_BROADCAST || src_op_pattern == OP_PATTERN_BROADCAST_ENHANCED)
         && IsNeedReshape(trans_info_ptr_->src_op_desc) && src_fz_flag);

    bool dst_broadcast_reshape_flag =
        ((dst_op_pattern == OP_PATTERN_BROADCAST || dst_op_pattern == OP_PATTERN_BROADCAST_ENHANCED)
         && IsNeedReshape(trans_info_ptr_->dst_op_desc) && dst_fz_flag);

    bool src_reduce_reshape_flag = (src_op_pattern == OP_PATTERN_REDUCE && src_fz_flag);

    bool dst_reduce_reshape_flag = (dst_op_pattern == OP_PATTERN_REDUCE && dst_fz_flag);


    bool reshape_flag = src_broadcast_reshape_flag || dst_broadcast_reshape_flag ||
                        src_reduce_reshape_flag || dst_reduce_reshape_flag;

    if (reshape_flag) {
      FE_LOGD(
          "ReShapeReduce: source_node is %s, dst_node is %s; src_op_pattern is %d, dst_op_pattern is %d;"
          "srcOutFormat is %d, dst_in_primary_format is %d.",
          trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str(),
          src_op_pattern, dst_op_pattern, src_out_primary_format, dst_in_primary_format);
      return ReShapeReduce(fused_graph, trans_info_ptr);
    }
    FE_LOGW("Shapes and formats of src and dst are not correct. Source Node is %s, dst node is %s.",
            trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str());
    return SUCCESS;
  }
}

Status TransNodeReshapeGenerator::AddNecessaryPeerNodes(ge::ComputeGraph &fused_graph, ge::NodePtr new_node) const {
  stringstream op_name_temp;
  // The atomic id of trans nodes must be unique.(start from 0)
  op_name_temp << "trans_Const_" << GetTransAtomicId();
  if (new_node->GetOpDesc()->GetInputsSize() < kNewNodeInputSizeLimit) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Reshape] Reshape %s does not have two inputs.", new_node->GetName().c_str());
    return FAILED;
  }
  auto second_input_of_reshape = new_node->GetOpDesc()->GetInputDesc(1);
  auto output_of_reshape = new_node->GetOpDesc()->MutableOutputDesc(0);
  ge::GeTensorPtr const_out_tenosr = nullptr;
  FE_MAKE_SHARED(const_out_tenosr = std::make_shared<ge::GeTensor>(second_input_of_reshape), return FAILED);
  FE_CHECK_NOTNULL(const_out_tenosr);

  vector<int32_t> shape_data;
  for (auto ele : output_of_reshape->MutableShape().GetDims()) {
    shape_data.emplace_back(static_cast<int32_t>(ele));
  }
  Status ret = const_out_tenosr->SetData(reinterpret_cast<uint8_t *>(shape_data.data()),
                                         shape_data.size() * sizeof(int32_t));
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Reshape] Set bias data failed.");
    return ret;
  }
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_out_tenosr);
  FE_CHECK_NOTNULL(const_op_desc);
  FE_LOGD("Create const input [%s] for reshape [%s].", const_op_desc->GetName().c_str(), new_node->GetName().c_str());

  auto const_node = fused_graph.AddNode(const_op_desc);
  if (const_node == nullptr) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][Reshape] Failed to add const node.");
    return FAILED;
  }

  if (ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), new_node->GetInDataAnchor(1)) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Trans][Reshape] Failed to add edge between const %s and reshape %s.",
                    const_node->GetName().c_str(), new_node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TransNodeReshapeGenerator::SetTensorDescInfo(ge::OpDescPtr &op_desc_ptr) const {
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto input_tensor_0 = op_desc_ptr->MutableInputDesc(0);
  FE_CHECK_NOTNULL(input_tensor_0);
  auto input_tensor_1 = op_desc_ptr->MutableInputDesc(1);
  FE_CHECK_NOTNULL(input_tensor_1);

  input_tensor_0->SetOriginFormat(trans_info_ptr_->src_out_original_format);
  input_tensor_0->SetOriginShape(trans_info_ptr_->src_out_original_shape);

  input_tensor_1->SetOriginFormat(static_cast<ge::Format>(ge::GetPrimaryFormat(input_tensor_1->GetFormat())));
  input_tensor_1->SetOriginShape(input_tensor_1->GetShape());

  for (auto output_tensor : op_desc_ptr->GetAllOutputsDescPtr()) {
    output_tensor->SetOriginFormat(trans_info_ptr_->src_out_original_format);
    output_tensor->SetOriginShape(trans_info_ptr_->src_out_original_shape);
  }
  return SUCCESS;
}

Status TransNodeReshapeGenerator::AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::GeShape &shape,
                                               const ge::Format &primary_format, const int32_t &sub_format,
                                               const ge::DataType &dtype) {
  ge::OpDescPtr op_desc_ptr = CreateBasicOpDescForTransNode(RESHAPE);
  FE_CHECK_NOTNULL(op_desc_ptr);

  FE_LOGD("Create [%s] node between [%s] and [%s] successfully!", RESHAPE.c_str(),
          trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str());

  auto input_format = static_cast<ge::Format>(
      ge::GetFormatFromSub(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->src_out_sub_format));
  if (op_desc_ptr->AddInputDesc(RESHAPE_INPUT_NAME, ge::GeTensorDesc(trans_info_ptr_->src_out_shape, input_format,
                                                                     trans_info_ptr_->src_out_data_type)) != SUCCESS) {
    FE_LOGD("CreateReshapeOp: op [RESHAPE]: add input desc unsuccessful.");
    return FAILED;
  }
  int64_t size_of_reshape = static_cast<int64_t>(shape.GetDimNum());
  std::vector<int64_t> dims = {size_of_reshape};
  ge::GeShape const_shape = ge::GeShape(dims);
  auto output_format = static_cast<ge::Format>(ge::GetFormatFromSub(primary_format, sub_format));
  if (op_desc_ptr->AddInputDesc(RESHAPE_SHAPE_NAME, ge::GeTensorDesc(const_shape, output_format, ge::DT_INT32)) !=
      SUCCESS) {
    FE_LOGD("CreateReshapeOp: op [RESHAPE]: add input desc unsuccessful.");
    return FAILED;
  }

  if (op_desc_ptr->AddOutputDesc(RESHAPE_OUTPUT_NAME, ge::GeTensorDesc(shape, output_format, dtype)) != SUCCESS) {
    FE_LOGD("CreateReshapeOp: op [RESHAPE]: add output desc unsuccessful.");
    return FAILED;
  }

  /* The output shape of reshape depends on the weight value of its
   * second input which name is "shape". */
  std::vector<string> dep_inputs = {"shape"};
  op_desc_ptr->SetOpInferDepends(dep_inputs);

  op_desc_ptr->SetIsInputConst({false, true});
  /* Reshape op is not belong to any ops store */
  (void)ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
  (void)ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);

  (void)SetTensorDescInfo(op_desc_ptr);

  // insert new op need add attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES
  // for data dump
  std::vector<std::string> original_names;
  if (!ge::AttrUtils::SetListStr(op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names)) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][AddOpAndNd] Node[%s]: failed to set attr _datadump_original_op_names.",
                    op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  SetTensorRealDimCountAndNewShape(op_desc_ptr, {trans_info_ptr_->src_out_shape, const_shape}, shape);
  SetNewShapeRange(op_desc_ptr, trans_info_ptr_->src_out_range, trans_info_ptr_->dst_in_range);
  SetNodeParallelGroupId(op_desc_ptr, trans_info_ptr_->src_op_desc, trans_info_ptr_->dst_op_desc);
  if (AddEdgesAndFreshTransInfo(fused_graph, op_desc_ptr) != SUCCESS) {
    FE_LOGD("Add edge unsuccessful!");
    return FAILED;
  }
  return SUCCESS;
}

Status TransNodeReshapeGenerator::ReShapeReduce(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) {
  // get op.pattern value
  auto src_op_pattern = trans_info_ptr->src_op_pattern;
  auto dst_op_pattern = trans_info_ptr->dst_op_pattern;

  // get previous out format and next in format
  auto src_out_primary_format = trans_info_ptr->src_out_primary_format;
  auto dst_in_primary_format = trans_info_ptr->dst_in_primary_format;

  bool isFz = (src_out_primary_format == ge::FORMAT_FRACTAL_Z && dst_in_primary_format == ge::FORMAT_FRACTAL_Z) ||
              (src_out_primary_format == ge::FORMAT_FRACTAL_Z_3D && dst_in_primary_format == ge::FORMAT_FRACTAL_Z_3D);
  if (src_op_pattern == OP_PATTERN_REDUCE && dst_op_pattern == OP_PATTERN_REDUCE && isFz) {
    FE_LOGD("Previous ops and the next op are both reduce, with formats being either FRACTAL_Z or FRACTAL_Z_3D; reshaping is not required.");
    return SUCCESS;
  }

  ge::GeShape new_shape;
  if ((src_op_pattern == OP_PATTERN_REDUCE || src_op_pattern == OP_PATTERN_BROADCAST ||
       src_op_pattern == OP_PATTERN_BROADCAST_ENHANCED) && (CheckOriginFormatIdentifiable(dst_in_primary_format))) {
    FE_LOGD("Reduce or broadcast op from FRACTAL_Z or FRACTAL_Z_3D to ND requires calculating the new shape from 6D or 7D to 4D.");
    auto src_out_shape = trans_info_ptr->src_out_shape;
    size_t reshape_size;
    int64_t first_dim;
    if (GetReshapeSizeAndFirstDim(src_out_primary_format, src_out_shape, reshape_size, first_dim) != SUCCESS) {
      return FAILED;
    }

    std::vector<int64_t> new_dim_vec;
    new_dim_vec.push_back(first_dim);
    size_t FRACTZ_LAST_THRRE_DIM = 3;
    for (size_t i = FRACTZ_LAST_THRRE_DIM; i > 0; i--) {
      new_dim_vec.push_back(src_out_shape.GetDim(reshape_size - i));
    }
    new_shape = ge::GeShape(new_dim_vec);
  } else {
    FE_LOGD("Reduce or broadcast op need to reshape, new shape is dst_in_shape.");
    new_shape = trans_info_ptr->dst_in_shape;
  }

  string new_shapestr = GetShapeDims(new_shape);
  FE_LOGI("Reduce or broadcast op after reshape the new shape is %s.", new_shapestr.c_str());
  FE_LOGI("Reduce or broadcast op source node is %s, dst node is %s.", trans_info_ptr_->src_op_desc->GetName().c_str(),
          trans_info_ptr_->dst_op_desc->GetName().c_str());
  return AddOpAndNode(fused_graph, new_shape, trans_info_ptr_->src_out_primary_format,
                      trans_info_ptr_->src_out_sub_format, trans_info_ptr->src_out_data_type);
}
}  // namespace fe
