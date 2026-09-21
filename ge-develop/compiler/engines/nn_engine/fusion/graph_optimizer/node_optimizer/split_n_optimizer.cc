/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/node_optimizer/split_n_optimizer.h"
#include <memory>
#include <vector>
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_graph_common.h"
#include "graph/debug/ge_attr_define.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "runtime/mem.h"

using namespace ge;
namespace fe {
const int kInputShapeLimit1 = 2;
const int kRealDimNchwToHwcn = 3;

void SplitNOptimizer::GetRealSplitDimFromOriginalFormatToFormat(const ge::OpDescPtr &op_desc,
                                                                int64_t &split_dim) const {
  (void)ge::AttrUtils::GetInt(op_desc, SPLIT_DIM, split_dim);
  ge::GeTensorDesc input_tensor = op_desc->GetInputDesc(0);
  auto input_format = ge::GetPrimaryFormat(input_tensor.GetFormat());
  ge::Format input_orinal_format = input_tensor.GetOriginFormat();
  ge::GeShape input_orinal_shape_shape = input_tensor.GetOriginShape();

  if (split_dim < 0) {
    FE_LOGD("Split_dim[%ld] is a negative number; changing it to a positive number.", split_dim);
    split_dim = static_cast<int64_t>(input_orinal_shape_shape.GetDimNum()) + split_dim;
  }
  FE_LOGD("GetRealSplitDim start node: %s, positive split dimension: %ld, input original shape size: %zu.",
          op_desc->GetName().c_str(), split_dim, input_orinal_shape_shape.GetDimNum());

  bool condition_nd_fractalz = input_orinal_format == FORMAT_ND && input_format == FORMAT_FRACTAL_NZ;
  if (condition_nd_fractalz) {
    if (input_orinal_shape_shape.GetDimNum() == kInputShapeLimit1) {
      if (split_dim == 0) {
        split_dim = 1;
      } else if (split_dim == 1) {
        split_dim = 0;
      }
    }
    FE_LOGD("Meet condition_nd_fractalz with value %d, changing split to %ld.", condition_nd_fractalz, split_dim);
  }
  bool condition_nchw_hwcn = input_orinal_format == FORMAT_NCHW  && input_format == FORMAT_HWCN;
  if (condition_nchw_hwcn) {
    FE_LOGD("meet condition_nchw_to_hwcn: %d, changing split to %d.", condition_nchw_hwcn, kRealDimNchwToHwcn);
    split_dim = kRealDimNchwToHwcn;
  }
  FE_LOGD("GetRealSplitDim end node:%s, splitdim: %ld.", op_desc->GetName().c_str(), split_dim);
}

bool SplitNOptimizer::InputCheck(ge::NodePtr split_node) const {
  for (auto in_node : split_node->GetInDataNodes()) {
    std::string op_type;
    if (ge::NodeUtils::GetConstOpType(in_node, op_type)) {
      FE_LOGD("In node %s of type %s, %s cannot be optimized.", in_node->GetName().c_str(), op_type.c_str(),
              split_node->GetName().c_str());
      return false;
    }
    if ((in_node->GetType() == TRANSDATA || in_node->GetType() == RESHAPE) && (in_node->GetAllInDataAnchors().size() != 0)) {
      FE_CHECK_NOTNULL(in_node->GetInDataAnchor(0));
      auto peerOutAnchor = in_node->GetInDataAnchor(0)->GetPeerOutAnchor();
      if (peerOutAnchor == nullptr) {
        FE_LOGD("The first peer of in_node[%s] in the anchor is null.", in_node->GetName().c_str());
        return false;
      }
      auto pre_in_node = peerOutAnchor->GetOwnerNode();
      if (pre_in_node != nullptr && IsRootGraphData(pre_in_node->GetType())) {
        FE_LOGD("Data->TransData/Reshape->split. Data: pre_in_node[%s], %s cannot be optimized.",
                pre_in_node->GetName().c_str(), split_node->GetName().c_str());
        return false;
      }
    }
    bool is_no_task = false;
    (void)ge::AttrUtils::GetBool(in_node->GetOpDesc(), ge::ATTR_NAME_NOTASK, is_no_task);
    if (is_no_task) {
      FE_LOGD("In node %s, the presence of the no_task attribute means that %s cannot be optimized.", in_node->GetName().c_str(),
              split_node->GetName().c_str());
        return false;
    }
    vector<int64_t> output_index;
    (void)ge::AttrUtils::GetListInt(in_node->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, output_index);
    if (!output_index.empty()) {
      FE_LOGD("Node [%s] has an atomic_output attribute and cannot be optimized.", in_node->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool SplitNOptimizer::InvalidNodeType(const string& node_type) {
  bool invalid_type = node_type == NETOUTPUT || node_type == "HcomBroadcast" || node_type == "HcomAllGather" ||
                      node_type == "HcomAllReduce" || node_type == "HcomReduceScatter" || node_type == "HcomReduce" ||
                      node_type == SPLITD || node_type == SPLITVD;
  if (invalid_type) {
    FE_LOGD("next node type: %s, cannot be optimized.", node_type.c_str());
    return true;
  }
  return false;
}

bool SplitNOptimizer::InvalidNodeAttr(const ge::OpDescPtr& node_desc) {
  int imply_type = 0;
  (void)ge::AttrUtils::GetInt(node_desc, ge::ATTR_NAME_IMPLY_TYPE, imply_type);
  string fusion_virtual_op = "";
  (void)ge::AttrUtils::GetStr(node_desc, ge::ATTR_NAME_FUSION_VIRTUAL_OP, fusion_virtual_op);
  bool is_continous_input = false;
  (void)ge::AttrUtils::GetBool(node_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
  vector<int64_t> output_index;
  (void)ge::AttrUtils::GetListInt(node_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, output_index);
  bool no_task = false;
  (void)ge::AttrUtils::GetBool(node_desc, ge::ATTR_NAME_NOTASK, no_task);
  if (no_task) {
    FE_LOGD("Next node %s has _no_task attribute, can't optimize.", node_desc->GetName().c_str());
    return true;
  }
  if (imply_type != 1) {
    FE_LOGD("Next node is not an aicore node, imply_type: %d.", imply_type);
    return true;
  }
  if (fusion_virtual_op != "") {
    FE_LOGD("Next node [%s] has fusion_virtual_op attribute, can't optimize.", node_desc->GetName().c_str());
    return true;
  }
  if (is_continous_input) {
    FE_LOGD("Next node [%s] has the continous_input attribute, cannot optimize.", node_desc->GetName().c_str());
    return true;
  }
  if (!output_index.empty()) {
    FE_LOGD("Next node [%s] has an atomic attribute; cannot optimize.", node_desc->GetName().c_str());
    return true;
  }
  return false;
}

bool SplitNOptimizer::OutputCheck(ge::NodePtr split_node) const {
  for (auto output_anchor : split_node->GetAllOutDataAnchors()) {
    for (size_t i = 0; i < output_anchor->GetPeerInDataAnchors().size(); i++) {
      ge::NodePtr next_node = output_anchor->GetPeerInDataAnchors().at(i)->GetOwnerNode();
      ge::OpDescPtr next_node_desc = next_node->GetOpDesc();
      FE_CHECK_NOTNULL(next_node_desc);
      FE_LOGD("Split node next_node name is %s, type: %s.", next_node_desc->GetName().c_str(),
              next_node_desc->GetType().c_str());
      if (InvalidNodeType(next_node_desc->GetType())) {
        FE_LOGD("Output node [%s] has an invalid node type; %s cannot be optimized.", next_node_desc->GetName().c_str(),
                split_node->GetName().c_str());
        return false;
      }
      if (InvalidNodeAttr(next_node_desc)) {
        FE_LOGD("Output node [%s] has an invalid node attribute; %s cannot be optimized.", next_node_desc->GetName().c_str(),
                split_node->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

bool SplitNOptimizer::NeedSkip(const ge::NodePtr &node, const ge::OpDescPtr &op_desc) const {
  bool is_not_split = op_desc->GetType() != fe::SPLITD && op_desc->GetType() != fe::SPLITVD;
  string node_name = op_desc->GetName();
  if (is_not_split) {
    return true;
  }
  int64_t split_dim = -1;
  (void)GetRealSplitDimFromOriginalFormatToFormat(op_desc, split_dim);
  if (split_dim != 0) {
    FE_LOGD("split_dim is not 0; %s cannot be optimized.", node_name.c_str());
    return true;
  }

  if (UnknownShapeUtils::IsUnknownShapeOp(*node->GetOpDesc())) {
    FE_LOGD("Unknown shape for split operation [%s], cannot be optimized.", node_name.c_str());
    return true;
  }

  if (!InputCheck(node)) {
    FE_LOGD("Split input check unsuccessful, %s cannot be optimized.", node_name.c_str());
    return true;
  }

  if (!OutputCheck(node)) {
    FE_LOGD("Split output check unsuccessful, %s cannot be optimized.", node_name.c_str());
    return true;
  }

  if (InvalidMemType(op_desc)) {
    FE_LOGD("Split memory type check unsuccessful, %s cannot be optimized.", node_name.c_str());
    return true;
  }
  if (FeGraphCommon::IsNodeOfUnknownGraph(*node)) {
    FE_LOGD("Owner graph is a dynamic graph; %s cannot be optimized.", node_name.c_str());
    return true;
  }
  return false;
}

Status SplitNOptimizer::SetFusionVirtualOp(const ge::ComputeGraph &graph) const {
  FE_LOGD("Starting SplitNOptimizer.");
  ge::ComputeGraph::Vistor<ge::NodePtr> nodes = graph.GetDirectNode();
  for (auto &node : nodes) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc);
    string node_name = op_desc->GetName();
    if (NeedSkip(node, op_desc)) {
      FE_LOGD("Node [%s] needs to be skipped.", node_name.c_str());
      continue;
    }
    FE_LOGD("Node[%s] start to set attribute.", node->GetName().c_str());
    (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOTASK, true);
    (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
    (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
  }
  FE_LOGD("End execution of SplitNOptimizer.");
  return fe::SUCCESS;
}
}  // namespace fe
