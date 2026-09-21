/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/node_optimizer/split_c_to_n_optimizer.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "common/fe_graph_common.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
const int kNDimIndex = 1;
const int kRealDimNchwTo5Hd = 0;
const int kInputShapeLimit = 4;
const int kValidNcDimSize = 2;

bool SplitCToNOptimizer::CheckSplitDim(const ge::OpDescPtr &op_desc) const {
  int64_t split_dim = -1;
  (void)ge::AttrUtils::GetInt(op_desc, SPLIT_DIM, split_dim);
  ge::GeTensorDescPtr input_tensor = op_desc->MutableInputDesc(0);
  if(input_tensor == nullptr){
    return false;
  }
  ge::GeShape input_orinal_shape_shape = input_tensor->GetOriginShape();

  if (split_dim < 0) {
    FE_LOGD("Split_dim[%ld] is a negative number; changing it to positive.", split_dim);
    split_dim = static_cast<int64_t>(input_orinal_shape_shape.GetDimNum()) + split_dim;
  }
  FE_LOGD("GetRealSplitDim start node: %s, positive split dim: %ld, input original shape size: %zu.",
          op_desc->GetName().c_str(), split_dim, input_orinal_shape_shape.GetDimNum());
  return split_dim == 1;
}

bool SplitCToNOptimizer::MeetAlignmentConditionFromNCHWTo5HD(const ge::OpDescPtr &op_desc) const {
  for (const auto &cur_desc_ptr : op_desc->GetAllOutputsDescPtr()) {
    if (cur_desc_ptr == nullptr) {
      return false;
    }
    auto origin_c = cur_desc_ptr->GetOriginShape().GetDim(1);
    ge::DataType data_type = cur_desc_ptr->GetDataType();
    bool cond = (data_type == ge::DT_FLOAT16 && origin_c % 16 == 0) ||
                (data_type == ge::DT_INT8 && origin_c % 32 == 0) ||
                (data_type == ge::DT_INT4 && origin_c % 64 == 0);
    if (!cond) {
      FE_LOGD("[Op:%s] Does not meet the condition_nchw_to_nc1hwc0 requirement.", op_desc->GetName().c_str());
      return false;
    }
  }
  FE_LOGD("[Op:%s] Satisfies the alignment condition for NCHW to NC1HWC0 conversion.", op_desc->GetName().c_str());
  return true;
}

bool SplitCToNOptimizer::MeetDimNumConditionFromNDToNZ(const ge::OpDescPtr &op_desc) const {
  ge::GeTensorDescPtr input_tensor = op_desc->MutableInputDesc(0);
  if(input_tensor == nullptr){
    return false;
  }
  ge::GeShape input_orinal_shape_shape = input_tensor->GetOriginShape();
  if (input_orinal_shape_shape.GetDimNum() < kInputShapeLimit) {
    return false;
  }
  return true;
}

bool SplitCToNOptimizer::CheckAxis(const ge::OpDescPtr &op_desc) const {
  for (const auto &cur_desc_ptr : op_desc->GetAllOutputsDescPtr()) {
    if (cur_desc_ptr == nullptr) {
      return false;
    }
    auto ori_shape = cur_desc_ptr->GetOriginShape().GetDims();
    if (ori_shape.size() < kValidNcDimSize) {
      FE_LOGD("[%s]: original output shape is invalid", op_desc->GetName().c_str());
      return false;
    }
    int64_t origin_n = ori_shape[0];
    if (origin_n != 1) {
      FE_LOGD("First axis is %ld, does not meet optimization condition.", origin_n);
      return false;
    }
  }
  return true;
}

bool SplitCToNOptimizer::CheckCommonCondition(const ge::ComputeGraph &graph, const ge::NodePtr &node,
                                              const ge::OpDescPtr &op_desc) const {
  bool is_not_split = op_desc->GetType() != fe::SPLITD && op_desc->GetType() != fe::SPLITVD;
  string node_name = op_desc->GetName();
  if (is_not_split) {
    return false;
  }

  if (UnknownShapeUtils::IsUnknownShapeOp(*node->GetOpDesc())) {
    FE_LOGD("Unknown shape operation for split op[%s] cannot be optimized.", node_name.c_str());
    return false;
  }

  if (!checker_helper.InputCheck(node)) {
    FE_LOGD("Split input check unsuccessful, %s cannot be optimized.", node_name.c_str());
    return false;
  }

  if (!checker_helper.OutputCheck(node)) {
    FE_LOGD("Split output check unsuccessful, %s cannot be optimized.", node_name.c_str());
    return false;
  }

  if (InvalidMemType(op_desc)) {
    FE_LOGD("Split mem type check unsuccess, %s can not optimize.", node_name.c_str());
    return false;
  }

  if (FeGraphCommon::IsNodeOfUnknownGraph(*node)) {
    FE_LOGD("Graph %s is a dynamic graph; %s cannot perform optimization.", graph.GetName().c_str(), node_name.c_str());
    return false;
  }

  return true;
}
bool SplitCToNOptimizer::NeedSkip(const ge::ComputeGraph &graph, const ge::NodePtr &node,
                                  const ge::OpDescPtr &op_desc) const {
  // check whether split_dim can be changed from C(1) to N(0)
  auto input_tensor = op_desc->MutableInputDesc(0);
  if (input_tensor == nullptr) {
    return true;
  }
  auto input_format = ge::GetPrimaryFormat(input_tensor->GetFormat());
  ge::Format input_orinal_format = input_tensor->GetOriginFormat();

  bool condition_nd_nz = (input_orinal_format == ge::FORMAT_ND && input_format == ge::FORMAT_FRACTAL_NZ);
  bool condition_nchw_5hd = (input_orinal_format == ge::FORMAT_NCHW && input_format == ge::FORMAT_NC1HWC0);

  if (CheckSplitDim(op_desc) && CheckAxis(op_desc) && ((input_orinal_format == input_format) ||
     (condition_nd_nz && MeetDimNumConditionFromNDToNZ(op_desc)) ||
     (condition_nchw_5hd && MeetAlignmentConditionFromNCHWTo5HD(op_desc))) &&
     CheckCommonCondition(graph, node, op_desc)) {
    return false;
  } else {
    return true;
  }
}

Status SplitCToNOptimizer::SetFusionVirtualOp(const ge::ComputeGraph &graph) const {
  FE_LOGD("Starting SplitCTONOptimizer.");
  ge::ComputeGraph::Vistor<ge::NodePtr> nodes = graph.GetDirectNode();
  for (auto &node : nodes) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc);
    string node_name = op_desc->GetName();
    bool is_no_task = false;
    (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, is_no_task);
    if (is_no_task) {
      FE_LOGD("Node:[%s] has been set no_task attribute, need to skip.", node_name.c_str());
      continue;
    }
    if (NeedSkip(graph, node, op_desc)) {
      FE_LOGD("Node %s needs to be skipped.", node_name.c_str());
      continue;
    }
    FE_LOGD("Node[%s] start to set attribute.", node->GetName().c_str());
    (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOTASK, true);
    (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
    (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
  }
  FE_LOGD("Finished executing SplitCToNOptimizer.");
  return fe::SUCCESS;
}
}
