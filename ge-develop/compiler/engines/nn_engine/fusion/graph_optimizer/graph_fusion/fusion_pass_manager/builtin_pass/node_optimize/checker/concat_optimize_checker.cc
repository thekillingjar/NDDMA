/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/checker/concat_optimize_checker.h"
#include "graph/types.h"
#include <algorithm>
#include "common/platform_utils.h"
#include "common/fe_context_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
bool ConcatOptimizeChecker::Check(const ge::NodePtr &node_ptr) const {
  return !IsInputFromSameNode(node_ptr) && IsDimC(node_ptr, CONCAT_DIM, true) && IsDimCAligned(node_ptr) &&
         is_pre_node_valid(node_ptr) && is_next_node_valid(node_ptr, 1, false) && IsNCHWOrNHWC(node_ptr) &&
         IsInputNotData(node_ptr) && !UnknownShapeUtils::IsUnknownShapeOp(*node_ptr->GetOpDesc());
}

bool ConcatOptimizeChecker::CheckWithQuant(const ge::NodePtr &node_ptr) const {
  return !IsInputFromSameNode(node_ptr) && IsDimC(node_ptr, CONCAT_DIM, true) && IsDimCAlignedWithQuant(node_ptr) &&
         is_pre_node_valid(node_ptr) && is_next_node_valid(node_ptr, 1, true) && IsNCHWOrNHWC(node_ptr);
}

bool ConcatOptimizeChecker::IsNCHWOrNHWC(const ge::NodePtr &node_ptr) const {
  for (auto &input_desc : node_ptr->GetOpDesc()->GetAllInputsDescPtr()) {
    if (input_desc->GetOriginFormat() != ge::FORMAT_NCHW && input_desc->GetOriginFormat() != ge::FORMAT_NHWC) {
      return false;
    }
    if (input_desc->MutableShape().GetDimNum() != CONCAT_SHAPE_DIM_DEFAULT) {
      FE_LOGD("The input dimension of the concat operator must be 4.");
      return false;
    }
  }
  return true;
}

bool ConcatOptimizeChecker::IsInputFromSameNode(const ge::NodePtr &node_ptr) const {
  string node_name = node_ptr->GetName();
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  size_t input_size = op_desc_ptr->GetInputsSize();
  if (input_size < 2) {
    return false;
  }

  // 1. get the pre_op_desc_ptr0
  ge::NodePtr pre_node_ptr0;
  Status status = NodeOptimizeUtils::GetPreNode(node_ptr, 0, pre_node_ptr0);
  if (status != SUCCESS) {
    FE_LOGD("Node[%s]: unable to obtain the previous node of input0.", node_name.c_str());
    return false;
  }
  ge::OpDescPtr pre_op_desc_ptr0 = pre_node_ptr0->GetOpDesc();

  // 2. check the other inputs
  for (size_t i = 1; i != input_size; ++i) {
    ge::NodePtr pre_node;
    status = NodeOptimizeUtils::GetPreNode(node_ptr, i, pre_node);
    if (status != SUCCESS) {
      FE_LOGD("Node[%s]: get the previous node of the input [%zu] not successfully.", node_name.c_str(), i);
      return false;
    }
    if (pre_node->GetOpDesc() != pre_op_desc_ptr0) {
      return false;
    }
  }
  FE_LOGD("Node[%s]: all inputs are from the same node, check unsuccessful.", node_name.c_str());
  return true;
}

bool ConcatOptimizeChecker::IsDimCAligned(const ge::NodePtr &node_ptr) const {
  string node_name = node_ptr->GetName();
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  size_t input_size = op_desc_ptr->GetInputsSize();
  for (size_t i = 0; i != input_size; ++i) {
    // 1. do nothing for the last one
    if (i == input_size - 1) {
      continue;
    }

    // 2. get the dim_c
    ge::GeTensorDesc tensor_desc = op_desc_ptr->GetInputDesc(i);
    int dim_c = 0;
    Status status = GetDimC(tensor_desc, dim_c);
    if (status != SUCCESS) {
      FE_LOGD("Node[%s]: get the dim C of the input [%zu] not successfully, check unsuccessful.",
          node_name.c_str(), i);
      return false;
    }

    // 3. check the dim_c
    auto quant_data_type = tensor_desc.GetOriginDataType();
    if (!IsDimCOfInputAligned(tensor_desc, dim_c, quant_data_type)) {
      FE_LOGD("Node[%s]: The dimension C of the input [%zu] is not aligned; check unsuccessful.", node_name.c_str(), i);
      return false;
    }
  }
  return true;
}

bool ConcatOptimizeChecker::IsDimCAlignedWithQuant(const ge::NodePtr &node_ptr) const {
  string node_name = node_ptr->GetName();
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  size_t input_size = op_desc_ptr->GetInputsSize();
  for (size_t i = 0; i != input_size; ++i) {
    // 1. do nothing for the last one
    if (i == input_size - 1) {
      continue;
    }

    // 2. get the dim_c
    ge::GeTensorDesc tensor_desc = op_desc_ptr->GetInputDesc(i);
    int dim_c = 0;
    Status status = GetDimC(tensor_desc, dim_c);
    if (status != SUCCESS) {
      FE_LOGD("Node[%s]: get the dim C of the input [%zu] not successfully, check unsuccessful.",
          node_name.c_str(), i);
      return false;
    }

    // 3. check the dim_c
    ge::NodePtr concat_next_node = nullptr;
    ge::DataType quant_data_type = tensor_desc.GetOriginDataType();
    if (node_ptr->GetOutDataAnchor(0) != nullptr && !node_ptr->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty()) {
      ge::OutDataAnchor::Vistor<ge::InDataAnchorPtr> peer_in_data_anchors =
          node_ptr->GetOutDataAnchor(0)->GetPeerInDataAnchors();
      ge::InDataAnchorPtr in_data_anchor_ptr = peer_in_data_anchors.at(0);
      if (in_data_anchor_ptr != nullptr &&
          in_data_anchor_ptr->GetOwnerNode() != nullptr &&
          in_data_anchor_ptr->GetOwnerNode()->GetType() == QUANT) {
        auto quant_node = in_data_anchor_ptr->GetOwnerNode();
        FE_CHECK(quant_node == nullptr || quant_node->GetOpDesc()== nullptr,
          FE_LOGD("Get quant op desc unsuccessful."), return FAILED);
        quant_data_type = quant_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
      }
    }

    if (!IsDimCOfInputAligned(tensor_desc, dim_c, quant_data_type)) {
      FE_LOGD("Node[%s]: The dimension C of the input [%zu] is not aligned; check unsuccessful.", node_name.c_str(), i);
      return false;
    }
  }
  return true;
}

bool ConcatOptimizeChecker::is_pre_node_valid(const ge::NodePtr &node_ptr) const {
  string node_name = node_ptr->GetName();
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  size_t input_size = op_desc_ptr->GetInputsSize();
  for (size_t i = 0; i != input_size; ++i) {
    ge::NodePtr pre_node_ptr;
    Status status = NodeOptimizeUtils::GetPreNode(node_ptr, i, pre_node_ptr);
    if (status != SUCCESS) {
      FE_LOGD("Node[%s]: get the previous node of the input [%zu] not successfully, check unsuccessful.",
              node_ptr->GetName().c_str(), i);
      return false;
    }
    ge::OpDescPtr pre_op_desc_ptr = pre_node_ptr->GetOpDesc();
    bool is_continous_input = false;
    bool is_continous_output = false;
    bool is_ref = false;
    bool no_task = false;
    bool output_reuse_input = false;
    bool no_padding_continuous_input = false;
    bool no_padding_continuous_output = false;
    (void)ge::AttrUtils::GetBool(pre_op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_INPUT, is_continous_input);
    (void)ge::AttrUtils::GetBool(pre_op_desc_ptr, ge::ATTR_NAME_CONTINUOUS_OUTPUT, is_continous_output);
    (void)ge::AttrUtils::GetBool(pre_op_desc_ptr, ge::ATTR_NAME_REFERENCE, is_ref);
    (void)ge::AttrUtils::GetBool(pre_op_desc_ptr, ge::ATTR_NAME_NOTASK, no_task);
    (void)ge::AttrUtils::GetBool(pre_op_desc_ptr, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, output_reuse_input);
    (void)ge::AttrUtils::GetBool(pre_op_desc_ptr, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                                 no_padding_continuous_input);
    (void)ge::AttrUtils::GetBool(pre_op_desc_ptr, ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT,
                                 no_padding_continuous_output);
    if (is_continous_input || is_continous_output || is_ref || no_task || output_reuse_input ||
        no_padding_continuous_input || no_padding_continuous_output) {
      FE_LOGD("Node [%s]: The previous node [%s] is not supported; check unsuccessful.", node_name.c_str(),
              pre_node_ptr->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool ConcatOptimizeChecker::is_next_node_valid(ge::NodePtr concat_node, uint32_t depth, bool has_relu) const {
  for (auto &output_anchor : concat_node->GetAllOutDataAnchors()) {
    auto peer_in_anchors = output_anchor->GetPeerInDataAnchors();
    for (size_t i = 0; i < peer_in_anchors.size(); ++i) {
      ge::NodePtr next_node = peer_in_anchors.at(i)->GetOwnerNode();
      if (next_node == nullptr) {
        return false;
      }
      ge::OpDescPtr next_node_desc = next_node->GetOpDesc();
      if (next_node_desc == nullptr) {
        return false;
      }
      uint32_t in_data_anchor_index = peer_in_anchors.at(i)->GetIdx();

      string next_node_name = next_node_desc->GetName();
      ge::GeTensorDescPtr geTensorDescPtr = next_node_desc->MutableInputDesc(in_data_anchor_index);
      if (geTensorDescPtr == nullptr) {
        return false;
      }
      int64_t format = ge::FORMAT_RESERVED;
      (void)ge::AttrUtils::GetInt(*geTensorDescPtr, ge::ATTR_NAME_STORAGE_FORMAT, format);
      ge::Format storage_format  = static_cast<ge::Format>(format);
      bool no_need_optimize = next_node_desc->GetType() == NETOUTPUT &&
                              (ge::GetPrimaryFormat(geTensorDescPtr->GetFormat()) == ge::FORMAT_NC1HWC0 ||
                               storage_format == ge::FORMAT_NC1HWC0);
      if (no_need_optimize) {
        FE_LOGD("Next node %s is netoutput; %s cannot be optimized.", next_node_name.c_str(),
                concat_node->GetName().c_str());
        return false;
      }
      if (depth > 0 && (next_node_desc->GetType() == QUANT ||
                        (has_relu && next_node_desc->GetType() == LEAKYRELU) ||
                        (has_relu && next_node_desc->GetType() == RELU))) {
        return is_next_node_valid(next_node, depth - 1, has_relu);
      }
    }
  }
  return true;
}
}  // namespace fe
