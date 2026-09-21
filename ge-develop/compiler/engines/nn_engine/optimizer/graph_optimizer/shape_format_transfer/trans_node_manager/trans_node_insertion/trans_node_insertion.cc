/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"
#include <common/unknown_shape_util.h>
#include "graph/types.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/fe_op_info_common.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "common/platform_utils.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_cast_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_reformat_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_reshape_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transdata_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transdatarnn_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transpose_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_squeeze_v2_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_unsqueeze_v2_generator.h"

namespace fe {
namespace {
const std::unordered_map<ge::Format, std::unordered_set<ge::Format>> kConflictFormat = {
    {ge::FORMAT_FRACTAL_Z, {ge::FORMAT_FRACTAL_ZN_LSTM, ge::FORMAT_FRACTAL_NZ}},
    {ge::FORMAT_FRACTAL_ZN_LSTM, {ge::FORMAT_FRACTAL_Z}}
};

void SubstituteNDWithOriginalFormat(TransInfoPtr &trans_info_ptr) {
  if (trans_info_ptr->src_out_primary_format == ge::FORMAT_ND) {
    if (trans_info_ptr->src_out_primary_format != trans_info_ptr->src_out_original_format &&
        !trans_info_ptr->src_out_shape.IsUnknownDimNum()) {
      trans_info_ptr->src_out_shape = trans_info_ptr->dst_in_tensor_desc_ptr->GetOriginShape();
    }
    auto dst_in_origin_format = trans_info_ptr->dst_in_tensor_desc_ptr->GetOriginFormat();
    trans_info_ptr->src_out_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(dst_in_origin_format));
    trans_info_ptr->init_src_out_primary_format = trans_info_ptr->src_out_primary_format;
    trans_info_ptr->src_out_sub_format = ge::GetSubFormat(dst_in_origin_format);
  }
  if (trans_info_ptr->dst_in_primary_format == ge::FORMAT_ND) {
    if (trans_info_ptr->dst_in_primary_format != trans_info_ptr->dst_in_original_format &&
        !trans_info_ptr->src_out_shape.IsUnknownDimNum()) {
      trans_info_ptr->dst_in_shape = trans_info_ptr->src_out_tensor_desc_ptr->GetOriginShape();
    }
    auto src_out_origin_format = trans_info_ptr->src_out_tensor_desc_ptr->GetOriginFormat();
    trans_info_ptr->dst_in_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(src_out_origin_format));
    trans_info_ptr->init_dst_in_primary_format = trans_info_ptr->dst_in_primary_format;
    trans_info_ptr->dst_in_sub_format = ge::GetSubFormat(src_out_origin_format);
  }
  if (trans_info_ptr->src_out_original_format == ge::FORMAT_ND) {
    trans_info_ptr->src_out_original_format = trans_info_ptr->dst_in_tensor_desc_ptr->GetOriginFormat();
  }
}
} // namespace

Status TransNodeInsertion::AddTransNodeType(TransNodeBaseGenerator *trans_node_type) {
  FE_CHECK(
      trans_node_type == nullptr,
      REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][AddTransNdType] transNodeType is null, resulting in AddTransNodeType failure"),
      return PARAM_INVALID);
  whole_trans_nodes_vector_.push_back(trans_node_type);
  return SUCCESS;
}

Status TransNodeInsertion::initialize() {
  FE_MAKE_SHARED(global_trans_info_ptr_ = std::make_shared<TransInfo>(), return FAILED);
  FE_CHECK_NOTNULL(global_trans_info_ptr_);
  TransInfoPtr trans_info_front;
  TransInfoPtr trans_info_end;
  FE_MAKE_SHARED(trans_info_front = std::make_shared<TransInfo>(), return FAILED);
  FE_CHECK_NOTNULL(trans_info_front);
  FE_MAKE_SHARED(trans_info_end = std::make_shared<TransInfo>(), return FAILED);
  FE_CHECK_NOTNULL(trans_info_end);
  trans_info_front_and_end_.push_back(trans_info_front);
  trans_info_front_and_end_.push_back(trans_info_end);
  AddTransNodeType(new (std::nothrow) TransNodeReshapeGenerator(fe_ops_store_info_ptr_, global_trans_info_ptr_));
  AddTransNodeType(new (std::nothrow) TransNodeTransposeGenerator(fe_ops_store_info_ptr_, global_trans_info_ptr_));
  AddTransNodeType(new (std::nothrow) TransNodeCastGenerator(fe_ops_store_info_ptr_, global_trans_info_ptr_));
  AddTransNodeType(new (std::nothrow) TransNodeTransdataGenerator(fe_ops_store_info_ptr_, global_trans_info_ptr_));
  AddTransNodeType(new (std::nothrow) TransNodeTransdataRNNGenerator(fe_ops_store_info_ptr_, global_trans_info_ptr_));
  AddTransNodeType(new (std::nothrow) TransNodeReformatGenerator(fe_ops_store_info_ptr_, global_trans_info_ptr_));
  AddTransNodeType(new (std::nothrow) TransNodeSqueezeV2Generator(fe_ops_store_info_ptr_, global_trans_info_ptr_));
  AddTransNodeType(new (std::nothrow) TransNodeUnsqueezeV2Generator(fe_ops_store_info_ptr_, global_trans_info_ptr_));
  trans_nodes_insertion_strategy_ = STRATEGY_CONSECUTIVE_PRINCIPLE;
  return SUCCESS;
}

TransNodeInsertion::TransNodeInsertion(FEOpsKernelInfoStorePtr fe_ops_store_ptr)
    : fe_ops_store_info_ptr_(fe_ops_store_ptr), global_trans_info_ptr_(nullptr) {}

TransNodeInsertion::~TransNodeInsertion() {
  for (auto trans_node : whole_trans_nodes_vector_) {
    delete trans_node;
  }
}

void TransNodeInsertion::SetTransInfoForInsertionModeFront() {
  *trans_info_front_and_end_[0] = *global_trans_info_ptr_;
  auto dst_in_origin_format = global_trans_info_ptr_->dst_in_tensor_desc_ptr->GetOriginFormat();
  trans_info_front_and_end_[0]->src_out_primary_format =
      static_cast<ge::Format>(ge::GetPrimaryFormat(dst_in_origin_format));
  trans_info_front_and_end_[0]->init_src_out_primary_format = trans_info_front_and_end_[0]->src_out_primary_format;
  trans_info_front_and_end_[0]->src_out_sub_format = ge::GetSubFormat(dst_in_origin_format);
  trans_info_front_and_end_[0]->src_out_c0_format = ge::GetC0Value(dst_in_origin_format);
  trans_info_front_and_end_[0]->src_out_shape = global_trans_info_ptr_->dst_in_tensor_desc_ptr->GetOriginShape();
  trans_info_front_and_end_[0]->src_out_range = GetOriginShapeRange(*global_trans_info_ptr_->dst_in_tensor_desc_ptr);

  trans_info_front_and_end_[0]->src_out_original_format =
      global_trans_info_ptr_->dst_in_tensor_desc_ptr->GetOriginFormat();
  trans_info_front_and_end_[0]->src_out_original_shape =
      global_trans_info_ptr_->dst_in_tensor_desc_ptr->GetOriginShape();
  trans_info_front_and_end_[0]->insertion_mode = INSERT_TRANS_NODE_BY_ORIGINAL_FORMAT_FRONT;
}

void TransNodeInsertion::SetTransInfoForInsertionModeEnd() {
  *trans_info_front_and_end_[1] = *global_trans_info_ptr_;
  auto src_out_origin_format = global_trans_info_ptr_->src_out_tensor_desc_ptr->GetOriginFormat();
  trans_info_front_and_end_[1]->dst_in_primary_format =
      static_cast<ge::Format>(ge::GetPrimaryFormat(src_out_origin_format));
  trans_info_front_and_end_[1]->init_dst_in_primary_format = trans_info_front_and_end_[1]->dst_in_primary_format;
  trans_info_front_and_end_[1]->dst_in_sub_format = ge::GetSubFormat(src_out_origin_format);
  trans_info_front_and_end_[1]->dst_in_c0_format = ge::GetC0Value(src_out_origin_format);
  trans_info_front_and_end_[1]->dst_in_shape = global_trans_info_ptr_->src_out_tensor_desc_ptr->GetOriginShape();
  trans_info_front_and_end_[1]->dst_in_range = GetOriginShapeRange(*global_trans_info_ptr_->src_out_tensor_desc_ptr);
  trans_info_front_and_end_[1]->dst_in_original_format =
      global_trans_info_ptr_->src_out_tensor_desc_ptr->GetOriginFormat();
  trans_info_front_and_end_[1]->dst_in_original_shape =
      global_trans_info_ptr_->src_out_tensor_desc_ptr->GetOriginShape();
  trans_info_front_and_end_[1]->insertion_mode = INSERT_TRANS_NODE_BY_ORIGINAL_FORMAT_END;
}

Status TransNodeInsertion::FillTransInfo(const ge::InDataAnchorPtr &dst_anchor, const ge::OutDataAnchorPtr &src_anchor,
                                       const ge::NodePtr &src_node, const ge::NodePtr &dst_node,
                                       ConcecutivePrinciple &use_concecutive_principle) {
  global_trans_info_ptr_->dst_anchor = dst_anchor;
  global_trans_info_ptr_->dst_op_desc = dst_node->GetOpDesc();
  global_trans_info_ptr_->dst_imply_type = static_cast<OpImplType>(0);
  global_trans_info_ptr_->src_anchor = src_anchor;
  global_trans_info_ptr_->src_op_desc = src_node->GetOpDesc();
  global_trans_info_ptr_->src_imply_type = static_cast<OpImplType>(0);
  global_trans_info_ptr_->src_node_ptr = src_node;
  global_trans_info_ptr_->dst_node_ptr = dst_node;
  uint32_t src_anchor_index = static_cast<uint32_t>(global_trans_info_ptr_->src_anchor->GetIdx());
  uint32_t dst_anchor_index = static_cast<uint32_t>(global_trans_info_ptr_->dst_anchor->GetIdx());

  global_trans_info_ptr_->src_out_tensor_desc_ptr =
      global_trans_info_ptr_->src_op_desc->MutableOutputDesc(src_anchor_index);
  if (global_trans_info_ptr_->src_out_tensor_desc_ptr == nullptr) {
    FE_LOGE("Src node name: %s, type: %s, output %u is null", global_trans_info_ptr_->src_op_desc->GetNamePtr(),
            global_trans_info_ptr_->src_op_desc->GetTypePtr(), src_anchor_index);
    return FAILED;
  }

  auto src_out_format = global_trans_info_ptr_->src_out_tensor_desc_ptr->GetFormat();
  global_trans_info_ptr_->src_out_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(src_out_format));
  global_trans_info_ptr_->init_src_out_primary_format = global_trans_info_ptr_->src_out_primary_format;
  global_trans_info_ptr_->src_out_sub_format = static_cast<ge::Format>(ge::GetSubFormat(src_out_format));
  global_trans_info_ptr_->src_out_data_type = global_trans_info_ptr_->src_out_tensor_desc_ptr->GetDataType();
  if (HasC0Format(src_out_format) || FE_ORIGIN_FORMAT_SET.count(src_out_format) != 0) {
    global_trans_info_ptr_->src_out_c0_format = ge::GetC0Value(src_out_format);
  } else {
    global_trans_info_ptr_->src_out_c0_format = GetC0BitByDataType(global_trans_info_ptr_->src_out_data_type);
  }
  global_trans_info_ptr_->src_out_shape = global_trans_info_ptr_->src_out_tensor_desc_ptr->GetShape();
  global_trans_info_ptr_->src_out_range = GetShapeRange(*global_trans_info_ptr_->src_out_tensor_desc_ptr);

  /* Here we want to make sure the input of trans node is correct.
   * Shape of const variable may be not correct, so we use their successor's
   * shape. When input is place_holder we
   * need to check it's father's op type is const or variable.
   * And if its father's original shape is empty, we just use it's father's
   * current shape.
   * */
  global_trans_info_ptr_->dst_in_tensor_desc_ptr =
      global_trans_info_ptr_->dst_op_desc->MutableInputDesc(dst_anchor_index);
  if (global_trans_info_ptr_->dst_in_tensor_desc_ptr == nullptr) {
    FE_LOGE("Dst node name: %s, type: %s, input %u is null", global_trans_info_ptr_->dst_op_desc->GetNamePtr(),
            global_trans_info_ptr_->dst_op_desc->GetTypePtr(), dst_anchor_index);
    return FAILED;
  }
  global_trans_info_ptr_->is_source_weight = CheckOpConstOrVariableInOriGraph(global_trans_info_ptr_->src_op_desc);
  global_trans_info_ptr_->is_dst_weight = CheckOpConstOrVariableInOriGraph(global_trans_info_ptr_->dst_op_desc);

  auto dst_in_format = global_trans_info_ptr_->dst_in_tensor_desc_ptr->GetFormat();
  global_trans_info_ptr_->dst_in_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(dst_in_format));
  global_trans_info_ptr_->init_dst_in_primary_format = global_trans_info_ptr_->dst_in_primary_format;
  global_trans_info_ptr_->dst_in_sub_format = static_cast<ge::Format>(ge::GetSubFormat(dst_in_format));
  global_trans_info_ptr_->dst_in_c0_format = ge::GetC0Value(dst_in_format);
  global_trans_info_ptr_->dst_in_data_type = global_trans_info_ptr_->dst_in_tensor_desc_ptr->GetDataType();
  if (HasC0Format(dst_in_format) || FE_ORIGIN_FORMAT_SET.count(dst_in_format) != 0) {
    global_trans_info_ptr_->dst_in_c0_format = ge::GetC0Value(dst_in_format);
  } else {
    global_trans_info_ptr_->dst_in_c0_format = GetC0ValByDataType(global_trans_info_ptr_->dst_in_data_type);
  }
  global_trans_info_ptr_->dst_in_shape = global_trans_info_ptr_->dst_in_tensor_desc_ptr->GetShape();
  global_trans_info_ptr_->dst_in_range = GetShapeRange(*global_trans_info_ptr_->dst_in_tensor_desc_ptr);

  FE_LOGD("Dst node name: %s, Type: %s, Dtype1: %s, dst_anchor_index: %u.", dst_node->GetName().c_str(),
          dst_node->GetType().c_str(), DTypeToStr(global_trans_info_ptr_->dst_in_data_type).c_str(), dst_anchor_index);
  FE_LOGD("Src node name: %s, type: %s, Dtype1: %s, src_anchor_index: %u.", src_node->GetName().c_str(),
          src_node->GetType().c_str(), DTypeToStr(global_trans_info_ptr_->src_out_data_type).c_str(), src_anchor_index);

  global_trans_info_ptr_->src_op_desc_type = global_trans_info_ptr_->src_op_desc->GetType();
  global_trans_info_ptr_->dst_op_desc_type = global_trans_info_ptr_->dst_op_desc->GetType();

  global_trans_info_ptr_->src_out_original_shape = global_trans_info_ptr_->src_out_tensor_desc_ptr->GetOriginShape();
  global_trans_info_ptr_->src_out_original_format = global_trans_info_ptr_->src_out_tensor_desc_ptr->GetOriginFormat();

  global_trans_info_ptr_->dst_in_original_shape = global_trans_info_ptr_->dst_in_tensor_desc_ptr->GetOriginShape();
  global_trans_info_ptr_->dst_in_original_format = global_trans_info_ptr_->dst_in_tensor_desc_ptr->GetOriginFormat();
  global_trans_info_ptr_->insertion_mode = INSERT_TRANS_NODE_BY_CONSECUTIVE_PRINCIPLE;

  int64_t hidden_size = RNN_HIDDEN_SIZE_DEFAULT_VALUE;
  int64_t input_size = RNN_INPUT_SIZE_DEFAULT_VALUE;
  int64_t state_size = RNN_STATE_SIZE_DEFAULT_VALUE;
  if (global_trans_info_ptr_->dst_in_primary_format == ge::FORMAT_FRACTAL_ZN_RNN ||
      global_trans_info_ptr_->dst_in_primary_format == ge::FORMAT_ND_RNN_BIAS) {
    (void)ge::AttrUtils::GetInt(dst_node->GetOpDesc(), "hidden_size", hidden_size);
    (void)ge::AttrUtils::GetInt(dst_node->GetOpDesc(), "input_size", input_size);
    (void)ge::AttrUtils::GetInt(dst_node->GetOpDesc(), "state_size", state_size);
  } else {
    (void)ge::AttrUtils::GetInt(src_node->GetOpDesc(), "hidden_size", hidden_size);
    (void)ge::AttrUtils::GetInt(src_node->GetOpDesc(), "input_size", input_size);
    (void)ge::AttrUtils::GetInt(src_node->GetOpDesc(), "state_size", state_size);
  }
  global_trans_info_ptr_->hidden_size = hidden_size;
  global_trans_info_ptr_->input_size = input_size;
  global_trans_info_ptr_->state_size = state_size;
  GetReshapeType();
  SubstituteNDWithOriginalFormat(global_trans_info_ptr_);
  use_concecutive_principle = IsAbleToUseConcecutivePrinciple();
  if (use_concecutive_principle != ConcecutivePrinciple::kConcecutive) {
    SetTransInfoForInsertionModeFront();
    SetTransInfoForInsertionModeEnd();
  }

  FE_LOGD(
      "Format of dst node is %s, format of source node is %s, src out sub format is %d, dst in sub format is %d, "
      "dst in original format is %s, src out original format is %s, src out c0 format is %d, dst in c0 format is %d, "
      "use_concecutive_principle is %d.",
      FormatToStr(global_trans_info_ptr_->dst_in_primary_format).c_str(),
      FormatToStr(global_trans_info_ptr_->src_out_primary_format).c_str(), global_trans_info_ptr_->src_out_sub_format,
      global_trans_info_ptr_->dst_in_sub_format, FormatToStr(global_trans_info_ptr_->dst_in_original_format).c_str(),
      FormatToStr(global_trans_info_ptr_->src_out_original_format).c_str(), global_trans_info_ptr_->src_out_c0_format,
      global_trans_info_ptr_->dst_in_c0_format, static_cast<int32_t>(use_concecutive_principle));
  return SUCCESS;
}

ConcecutivePrinciple TransNodeInsertion::IsAbleToUseConcecutivePrinciple() {
  /* If two consecutive nodes' original format is not the same,
   * we will insert trans-nodes by their own format and original dtype. */
  bool nz_format = global_trans_info_ptr_->src_out_primary_format == ge::FORMAT_FRACTAL_NZ ||
                   global_trans_info_ptr_->dst_in_primary_format == ge::FORMAT_FRACTAL_NZ;
  if (global_trans_info_ptr_->src_out_original_format != global_trans_info_ptr_->dst_in_original_format &&
      !nz_format) {
    return ConcecutivePrinciple::kOriFormatUnConcecutive;
  }

  /* If the original shape of two closed tensor is not equal, we
   * use original format to insert trans-nodes */
  bool origin_dims_equal = global_trans_info_ptr_->src_out_original_shape.GetDims() ==
                           global_trans_info_ptr_->dst_in_original_shape.GetDims();
  if (!origin_dims_equal) {
    return ConcecutivePrinciple::kOriShapeUnConcecutive;
  }

  bool format_equal = global_trans_info_ptr_->src_out_primary_format == global_trans_info_ptr_->dst_in_primary_format;
  bool dims_equal = global_trans_info_ptr_->src_out_shape.GetDims() == global_trans_info_ptr_->dst_in_shape.GetDims();
  if ((nz_format || format_equal) && !dims_equal) {
    return ConcecutivePrinciple::kShapeUnConcecutive;
  }

  if (std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(),
                global_trans_info_ptr_->src_out_primary_format) != FE_GROUP_RELA_FORMAT_VECTOR.end() &&
      global_trans_info_ptr_->src_out_primary_format == global_trans_info_ptr_->dst_in_primary_format && !dims_equal) {
    return ConcecutivePrinciple::kRelaFormatShapeUnConcecutive;
  }

  if (global_trans_info_ptr_->src_reshape_type != global_trans_info_ptr_->dst_reshape_type && !nz_format) {
    return ConcecutivePrinciple::kReShapeTypeUnConcecutive;
  }

  bool is_conflict = kConflictFormat.count(global_trans_info_ptr_->src_out_primary_format) != 0 &&
                     kConflictFormat.at(global_trans_info_ptr_->src_out_primary_format).count(
                         global_trans_info_ptr_->dst_in_primary_format) != 0;
  if (is_conflict) {
    return ConcecutivePrinciple::kConflictFormat;
  }

  bool is_src_dump_able = IsDumpableOp(global_trans_info_ptr_->src_op_desc) &&
          global_trans_info_ptr_->src_out_primary_format != global_trans_info_ptr_->src_out_original_format;
  bool is_dst_dump_able = IsDumpableOp(global_trans_info_ptr_->dst_op_desc) &&
          global_trans_info_ptr_->dst_in_primary_format != global_trans_info_ptr_->dst_in_original_format;
  if (is_src_dump_able || is_dst_dump_able) {
    return ConcecutivePrinciple::kDumpableUnConcecutive;
  }

  return ConcecutivePrinciple::kConcecutive;
}

Status TransNodeInsertion::InsertTransNodes(ge::ComputeGraph &fused_graph, ge::NodePtr dst_node) {
  ge::OpDescPtr dst_op_desc = dst_node->GetOpDesc();
  for (auto &dst_anchor : dst_node->GetAllInDataAnchors()) {
    if (dst_anchor == nullptr) {
      continue;
    }
    ge::OutDataAnchorPtr src_anchor = dst_anchor->GetPeerOutAnchor();
    if (src_anchor == nullptr) {
      continue;
    }
    ge::NodePtr src_node = src_anchor->GetOwnerNode();
    if (src_node == nullptr) {
      continue;
    }
    ge::OpDescPtr src_op_desc = src_node->GetOpDesc();
    if (src_op_desc == nullptr) {
      continue;
    }

    FE_LOGD("Edge: src:%s, src_index:%d, dst:%s, dst_index:%d.", src_op_desc->GetName().c_str(), src_anchor->GetIdx(),
            dst_op_desc->GetName().c_str(), dst_anchor->GetIdx());

    ConcecutivePrinciple use_concecutive_principle = ConcecutivePrinciple::kConcecutive;
    Status ret = FillTransInfo(dst_anchor, src_anchor, src_node, dst_node, use_concecutive_principle);
    if (ret != SUCCESS) {
      return ret;
    }

    if (use_concecutive_principle == ConcecutivePrinciple::kConcecutive) {
      ret = InsertTransOpByConcecutiveStrategy(fused_graph);
      if (ret != SUCCESS) {
        return ret;
      }
    } else {
      ret = InsertTransOpByOriginalFormat(fused_graph);
      if (ret != SUCCESS) {
        return ret;
      }
    }
  }
  return SUCCESS;
}

void TransNodeInsertion::GetDstReshapeType(const OpKernelInfoPtr &op_kernel) {
  std::string prop_reshape_type;
  (void)ge::AttrUtils::GetStr(global_trans_info_ptr_->dst_in_tensor_desc_ptr, ge::ATTR_NAME_RESHAPE_INFER_TYPE,
                              prop_reshape_type);
  if (op_kernel == nullptr) {
    global_trans_info_ptr_->dst_reshape_type = prop_reshape_type;
    /* Original A -> A, the reshape_type is the same for both ends */
    trans_info_front_and_end_[1]->src_reshape_type = prop_reshape_type;
    trans_info_front_and_end_[1]->dst_reshape_type = prop_reshape_type;
    return;
  }
  IndexNameMap input_map;
  if (GetInputIndexNameMap(*global_trans_info_ptr_->dst_op_desc, *op_kernel, input_map) == SUCCESS) {
    uint32_t in_anchor_index = static_cast<uint32_t>(global_trans_info_ptr_->dst_anchor->GetIdx());
    InputOrOutputInfoPtr intput_info_ptr = nullptr;
    if (op_kernel->GetInputInfoByName(input_map[in_anchor_index], intput_info_ptr) == SUCCESS) {
      std::string reshape_type = intput_info_ptr->GetReshapeType();
      if (reshape_type.empty() && !prop_reshape_type.empty()) {
        reshape_type = prop_reshape_type;
      }
      global_trans_info_ptr_->dst_reshape_type = reshape_type;
      /* Original A -> A, the reshape_type is the same for both ends */
      trans_info_front_and_end_[1]->src_reshape_type = reshape_type;
      trans_info_front_and_end_[1]->dst_reshape_type = reshape_type;
    }
  }
  global_trans_info_ptr_->dst_op_pattern = op_kernel->GetOpPattern();
  trans_info_front_and_end_[1]->src_op_pattern = op_kernel->GetOpPattern();
  trans_info_front_and_end_[1]->dst_op_pattern = op_kernel->GetOpPattern();
}

void TransNodeInsertion::GetSrcReshapeType(const OpKernelInfoPtr &op_kernel) {
  std::string prop_reshape_type;
  (void)ge::AttrUtils::GetStr(global_trans_info_ptr_->src_out_tensor_desc_ptr, ge::ATTR_NAME_RESHAPE_INFER_TYPE,
                              prop_reshape_type);
  if (op_kernel == nullptr) {
    global_trans_info_ptr_->src_reshape_type = prop_reshape_type;
    /* A->Original A, the reshape_type is the same for both ends */
    trans_info_front_and_end_[0]->src_reshape_type = prop_reshape_type;
    trans_info_front_and_end_[0]->dst_reshape_type = prop_reshape_type;
    return;
  }
  IndexNameMap output_map;
  if (GetOutputIndexNameMap(*global_trans_info_ptr_->src_op_desc, *op_kernel, output_map) == SUCCESS) {
    uint32_t out_anchor_index = static_cast<uint32_t>(global_trans_info_ptr_->src_anchor->GetIdx());
    InputOrOutputInfoPtr output_info_ptr = nullptr;
    if (op_kernel->GetOutputInfoByName(output_map[out_anchor_index], output_info_ptr) == SUCCESS) {
      std::string reshape_type = output_info_ptr->GetReshapeType();
      if (reshape_type.empty() && !prop_reshape_type.empty()) {
        reshape_type = prop_reshape_type;
      }
      global_trans_info_ptr_->src_reshape_type = reshape_type;
      /* A->Original A, the reshape_type is the same for both ends */
      trans_info_front_and_end_[0]->src_reshape_type = reshape_type;
      trans_info_front_and_end_[0]->dst_reshape_type = reshape_type;
    }
  }
  global_trans_info_ptr_->src_op_pattern = op_kernel->GetOpPattern();
  trans_info_front_and_end_[0]->src_op_pattern = op_kernel->GetOpPattern();
  trans_info_front_and_end_[0]->dst_op_pattern = op_kernel->GetOpPattern();
}

void TransNodeInsertion::InitReshapeType() {
  global_trans_info_ptr_->src_reshape_type = "";
  trans_info_front_and_end_[0]->src_reshape_type = "";
  trans_info_front_and_end_[1]->src_reshape_type = "";

  global_trans_info_ptr_->dst_reshape_type = "";
  trans_info_front_and_end_[0]->dst_reshape_type = "";
  trans_info_front_and_end_[1]->dst_reshape_type = "";

  global_trans_info_ptr_->src_op_pattern = OP_PATTERN_OP_KERNEL;
  trans_info_front_and_end_[0]->src_op_pattern = OP_PATTERN_OP_KERNEL;
  trans_info_front_and_end_[0]->dst_op_pattern = OP_PATTERN_OP_KERNEL;

  global_trans_info_ptr_->dst_op_pattern = OP_PATTERN_OP_KERNEL;
  trans_info_front_and_end_[1]->src_op_pattern = OP_PATTERN_OP_KERNEL;
  trans_info_front_and_end_[1]->dst_op_pattern = OP_PATTERN_OP_KERNEL;
}

Status TransNodeInsertion::GetReshapeType() {
  InitReshapeType();

  OpKernelInfoPtr src_op_kernel = OpsKernelManager::Instance(fe_ops_store_info_ptr_->GetFEOpsKernelInfoStoreName())
                                      .GetHighPrioOpKernelInfo(global_trans_info_ptr_->src_op_desc_type);
  GetSrcReshapeType(src_op_kernel);

  OpKernelInfoPtr dst_op_kernel = OpsKernelManager::Instance(fe_ops_store_info_ptr_->GetFEOpsKernelInfoStoreName())
                                      .GetHighPrioOpKernelInfo(global_trans_info_ptr_->dst_op_desc_type);
  GetDstReshapeType(dst_op_kernel);
  return SUCCESS;
}

bool TransNodeInsertion::IsInsertCastFirst(const TransInfoPtr trans_info_ptr) {
  // if C0=8, make sure transdata at the side of C0=8
  if (trans_info_ptr->dst_in_c0_format == SHAPE_NUMBER_8 && trans_info_ptr->src_out_c0_format != SHAPE_NUMBER_8) {
    return true;
  }
  int32_t dst_in_c0 =
      (HasC0Format(trans_info_ptr->dst_in_tensor_desc_ptr->GetFormat())) ? trans_info_ptr->dst_in_c0_format :
      GetC0ValByDataType(trans_info_ptr->dst_in_data_type);
  int32_t src_out_c0 =
      (HasC0Format(trans_info_ptr->src_out_tensor_desc_ptr->GetFormat())) ? trans_info_ptr->src_out_c0_format :
      GetC0ValByDataType(trans_info_ptr->src_out_data_type);
  // if src and dst C0 not same, keep transdata at the heavy format side
  if (dst_in_c0 != src_out_c0) {
      if (FE_HEAVY_FORMAT_SET.count(trans_info_ptr->dst_in_primary_format) != 0 &&
          FE_HEAVY_FORMAT_SET.count(trans_info_ptr->src_out_primary_format) == 0) {
        return true;
      }
      if (FE_HEAVY_FORMAT_SET.count(trans_info_ptr->src_out_primary_format) != 0 &&
          FE_HEAVY_FORMAT_SET.count(trans_info_ptr->dst_in_primary_format) == 0) {
        return false;
      }
  }
  int src_data_type_bits = GetDataTypeBits(trans_info_ptr->src_out_data_type);
  int dst_data_type_bits = GetDataTypeBits(trans_info_ptr->dst_in_data_type);
  // if data type reduce bits from src to dst, insert cast first
  if (src_data_type_bits > dst_data_type_bits && trans_info_ptr->src_out_c0_format != SHAPE_NUMBER_8) {
    return true;
  }
  // if bits of src and dst data type are the same, keep transdata at the heavy format side
  // for example (NCHW/bf16) -> (5HD/fp16), insert cast first, so the transdata op can be fused with the following op
  if (src_data_type_bits == dst_data_type_bits && trans_info_ptr->dst_in_c0_format > 0) {
    return true;
  }
  return false;
}

/* For normal node if the source data type is fp16 we insert cast at the
 * end, other wise we insert cast first */
Status TransNodeInsertion::InsertCastGeneralCase(const TransInfoPtr trans_info_ptr,
                                                 const uint32_t front_strategy_vector_index,
                                                 const uint32_t end_strategy_vector_index,
                                                 std::vector<std::vector<uint32_t>> &strategy_vector_combination) {
  if (strategy_vector_combination.empty()) {
    FE_LOGW("Strategy combination is empty.");
    return FAILED;
  }

  if (end_strategy_vector_index >= strategy_vector_combination.size() ||
      front_strategy_vector_index >= strategy_vector_combination.size()) {
    FE_LOGW("End index %u or front index %u is invalid. The size is %zu.", end_strategy_vector_index,
            front_strategy_vector_index, strategy_vector_combination.size());
    return FAILED;
  }

  if (IsInsertCastFirst(trans_info_ptr)) {
    FE_LOGD("Insert Cast at the beginning of the process.");
    strategy_vector_combination[end_strategy_vector_index].insert(
        strategy_vector_combination[end_strategy_vector_index].begin(), CAST_INDEX);
  } else {
    FE_LOGD("Insert Cast at the end.");
    strategy_vector_combination[front_strategy_vector_index].push_back(CAST_INDEX);
  }
  return SUCCESS;
}

Status TransNodeInsertion::CombineAllStrategy(TransInfoPtr trans_info_ptr, uint64_t global_strategy_id,
                                              std::vector<std::vector<uint32_t>> &strategy_vector_combination) {
  (void) global_strategy_id;
  if (strategy_vector_combination.empty()) {
    FE_LOGW("Strategy combination is empty.");
    return FAILED;
  }

  if (trans_info_ptr->src_out_data_type != trans_info_ptr->dst_in_data_type) {
    uint32_t front_strategy_vector_index = 0;
    uint32_t end_strategy_vector_index = strategy_vector_combination.size() - 1;
    return InsertCastGeneralCase(trans_info_ptr, front_strategy_vector_index, end_strategy_vector_index,
                                 strategy_vector_combination);
  }

  return SUCCESS;
}

Status TransNodeInsertion::GenerateStrategy(std::vector<std::vector<uint32_t>> &strategy_vector_combination) {
  uint64_t strategy_ext_val = CalcStrategyIdExtraVal(global_trans_info_ptr_);
  uint64_t strategy_id = CalcStrategyId(global_trans_info_ptr_->src_out_primary_format,
                                        global_trans_info_ptr_->dst_in_primary_format, strategy_ext_val);

  global_trans_info_ptr_->strategy_id = strategy_id;

  auto strategy = trans_nodes_insertion_strategy_.find(strategy_id);
  if (strategy == trans_nodes_insertion_strategy_.end()) {
    FE_LOGW("Trans situation is not supported, src format %u, dst format %u.",
            global_trans_info_ptr_->src_out_primary_format, global_trans_info_ptr_->dst_in_primary_format);
    return FAILED;
  }
  if (strategy_vector_combination.empty()) {
    FE_LOGW("Strategy Vector is empty!");
    return FAILED;
  }
  strategy_vector_combination[0] = strategy->second;
  return CombineAllStrategy(global_trans_info_ptr_, strategy_id, strategy_vector_combination);
}

Status TransNodeInsertion::InsertTransOpByConcecutiveStrategy(ge::ComputeGraph &fused_graph) {
  if (global_trans_info_ptr_->src_out_primary_format == global_trans_info_ptr_->dst_in_primary_format &&
      global_trans_info_ptr_->src_out_data_type == global_trans_info_ptr_->dst_in_data_type) {
    FE_LOGD("The format and data type of the source node %s and the destination node %s are the same.",
            global_trans_info_ptr_->src_op_desc_type.c_str(), global_trans_info_ptr_->dst_op_desc_type.c_str());
    return SUCCESS;
  }
  /* Lower dimensional op to higher dimension op:
   * Reshape -> Permute -> Cast -> TransData -> TransDataRNN;
   * higher dimensional op to lower(or equal) dimension op:
   * TransDataRNN -> TransData -> Cast -> Permute -> Reshape. */
  std::vector<std::vector<uint32_t>> strategy_vector_combination;
  strategy_vector_combination.emplace_back(std::vector<uint32_t>());
  GenerateStrategy(strategy_vector_combination);
  Status ret_value;
  for (auto transnode_idx : strategy_vector_combination[0]) {
    if (transnode_idx >= FORBIDDEN_INDEX) {
      REPORT_FE_ERROR("[GraphOpt][Trans][InsertTransByConcec] We do not support transactions from %s to %s between %s and %s.",
                      FormatToStr(global_trans_info_ptr_->src_out_primary_format).c_str(),
                      FormatToStr(global_trans_info_ptr_->dst_in_primary_format).c_str(),
                      global_trans_info_ptr_->src_op_desc->GetName().c_str(),
                      global_trans_info_ptr_->dst_op_desc->GetName().c_str());
      return FAILED;
    }
    ret_value = whole_trans_nodes_vector_[transnode_idx]->AddTransNode(fused_graph, global_trans_info_ptr_);
    if (ret_value != SUCCESS) {
      return ret_value;
    }
  }
  return SUCCESS;
}

Status TransNodeInsertion::UpdateNextTransInfo(uint32_t end_strategy_index) {
  if (end_strategy_index < 1) {
    return FAILED;
  }
  auto front_strategy_index = end_strategy_index - 1;
  trans_info_front_and_end_[front_strategy_index]->src_op_desc =
      trans_info_front_and_end_[end_strategy_index]->src_op_desc;
  trans_info_front_and_end_[front_strategy_index]->src_node_ptr =
      trans_info_front_and_end_[end_strategy_index]->src_node_ptr;
  trans_info_front_and_end_[front_strategy_index]->src_anchor =
      trans_info_front_and_end_[end_strategy_index]->src_anchor;
  trans_info_front_and_end_[front_strategy_index]->src_op_desc_type =
      trans_info_front_and_end_[end_strategy_index]->src_op_desc_type;

  trans_info_front_and_end_[front_strategy_index]->src_out_data_type =
      trans_info_front_and_end_[end_strategy_index]->src_out_data_type;

  trans_info_front_and_end_[front_strategy_index]->is_source_weight =
      trans_info_front_and_end_[end_strategy_index]->is_source_weight;

  trans_info_front_and_end_[front_strategy_index]->src_out_tensor_desc_ptr =
      trans_info_front_and_end_[end_strategy_index]->src_out_tensor_desc_ptr;
  return SUCCESS;
}

Status TransNodeInsertion::InsertTransOpByOriginalFormat(ge::ComputeGraph &fused_graph) {
  /* 1st strategy is the front strategy, which inserts trans-nodes
   * from input's original format to the current format.
   * 2nd strategy is the end strategy, which inserts trans-nodes
   * from output's current format to its original format. */
  vector<vector<uint32_t>> strategy_vector_combination;
  strategy_vector_combination.emplace_back(vector<uint32_t>());
  strategy_vector_combination.emplace_back(vector<uint32_t>());

  GenerateStrategyByOrignalFormat(strategy_vector_combination);
  if (strategy_vector_combination.empty()) {
    FE_LOGW("The strategy vector is empty!");
    return FAILED;
  }
  Status ret_value;
  auto combination_size = strategy_vector_combination.size();
  /* We need to insert trans-nodes by END strategy first then
   * by FRONT strategy. So we insert trans-nodes reversely from index size-1,
   * size-2 ... 0 through vector strategy_vector_combination */
  for (uint32_t i = 0; i < combination_size; i++) {
    auto strategy_index = (combination_size - 1) - i;
    auto strategy_vector = strategy_vector_combination[strategy_index];
    FE_LOGD("Step %zu from format %s to format %s.", strategy_index,
            FormatToStr(trans_info_front_and_end_[strategy_index]->src_out_primary_format).c_str(),
            FormatToStr(trans_info_front_and_end_[strategy_index]->dst_in_primary_format).c_str());
    for (auto transnode_idx : strategy_vector) {
      if (transnode_idx >= FORBIDDEN_INDEX) {
        REPORT_FE_ERROR("[GraphOpt][Trans][InsertTransByOri] We do not support transactions from %s to %s between %s and %s.",
                        FormatToStr(global_trans_info_ptr_->src_out_primary_format).c_str(),
                        FormatToStr(global_trans_info_ptr_->dst_in_primary_format).c_str(),
                        global_trans_info_ptr_->src_op_desc->GetName().c_str(),
                        global_trans_info_ptr_->dst_op_desc->GetName().c_str());
        return FAILED;
      }
      ret_value = whole_trans_nodes_vector_[transnode_idx]->AddTransNode(fused_graph,
                                                                         trans_info_front_and_end_[strategy_index]);
      if (ret_value != SUCCESS) {
        return ret_value;
      }
    }
    /* After insert trans-nodes from src current to src original,
     * we need to update the src-node for insertion from dst original to dst
     * current. */
    if (i == 0) {
      UpdateNextTransInfo(strategy_index);
    }
  }

  return SUCCESS;
}

Status TransNodeInsertion::GenerateStrategyByOrignalFormat(
    std::vector<std::vector<uint32_t>> &strategy_vector_combination) {
  for (size_t i = 0; i < strategy_vector_combination.size(); i++) {
    if (i >= trans_info_front_and_end_.size()) {
      FE_LOGW("Index %zu exceeds the size of trans-info %zu.", i, trans_info_front_and_end_.size());
      return FAILED;
    }
    uint64_t strategy_ext_val = CalcStrategyIdExtraVal(trans_info_front_and_end_[i]);
    uint64_t strategy_id = CalcStrategyId(trans_info_front_and_end_[i]->src_out_primary_format,
                                          trans_info_front_and_end_[i]->dst_in_primary_format, strategy_ext_val);

    trans_info_front_and_end_[i]->strategy_id = strategy_id;

    auto strategy = trans_nodes_insertion_strategy_.find(strategy_id);
    if (strategy == trans_nodes_insertion_strategy_.end()) {
      FE_LOGW("Trans situation is not supported, src format %u, dst format %u.",
              global_trans_info_ptr_->src_out_primary_format, global_trans_info_ptr_->dst_in_primary_format);
      return FAILED;
    }
    strategy_vector_combination[i] = strategy->second;
  }
  /* This ID is created by the source nodes' and dst nodes. */
  uint64_t global_strategy_ext_val = CalcStrategyIdExtraVal(global_trans_info_ptr_);
  uint64_t global_strategy_id = CalcStrategyId(global_trans_info_ptr_->src_out_primary_format,
                                               global_trans_info_ptr_->dst_in_primary_format, global_strategy_ext_val);

  return CombineAllStrategy(global_trans_info_ptr_, global_strategy_id, strategy_vector_combination);
}
}  // namespace fe
