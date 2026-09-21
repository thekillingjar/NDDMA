/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "param_calculate/tensorsize_calculator.h"

#include "common/fe_error_code.h"
#include "common/fe_type_utils.h"
#include "common/fe_log.h"
#include "common/constants_define.h"
#include "common/math_util.h"
#include "common/op_tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
namespace{
  const int64_t kNetEdgeDdrType = 2;
  const std::string OP_TYPE_END = "End";
  const std::string OP_TYPE_NETOUTPUT = "NetOutput";
}

Status TensorSizeCalculator::CalculateOpTensorSize(ge::NodePtr node) {
  FE_LOGD("Begin calculating tensor size for op [%s, %s].", node->GetNamePtr(),
          node->GetTypePtr());
  int32_t output_real_calc_flag = 0;
  (void)CalcInputOpTensorSize(node, output_real_calc_flag);
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc op_desc = *(op_desc_ptr.get());
  bool ret = ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, output_real_calc_flag);
  FE_LOGD("FE get tensor_actual_size: [%d], return value: [%d].", output_real_calc_flag, ret);
  auto output_data_anchors = node->GetAllOutDataAnchors();
  (void)CalcOutputOpTensorSize(*(op_desc_ptr.get()), output_real_calc_flag, output_data_anchors);
  FE_LOGD("Finished calculating the tensor size for operation [%s, %s].", node->GetNamePtr(), node->GetTypePtr());
  (void)op_desc.DelAttr(fe::ATTR_NAME_UNKNOWN_SHAPE);
 
  return SUCCESS;
}

Status TensorSizeCalculator::CalcSingleTensorSize(const ge::OpDesc &op_desc,
    const ge::GeTensorDescPtr &tensor_desc_ptr, const string &direction, size_t i, bool output_real_calc_flag,
    int64_t &tensor_size) {
  ge::DataType data_type = tensor_desc_ptr->GetDataType();
  auto &shape = tensor_desc_ptr->MutableShape();
  // if tensor_no_tiling_mem_type, need to multiply by the maximum value of the shape_range
  if (shape.IsUnknownShape()) {
    bool is_tensor_desc_mem = false;
    (void)ge::AttrUtils::GetBool(tensor_desc_ptr, ge::ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_tensor_desc_mem);
    if (is_tensor_desc_mem) {
      std::vector<std::pair<int64_t, int64_t>> shape_range;
      (void)tensor_desc_ptr->GetShapeRange(shape_range);
      tensor_size = GetSizeByDataType(data_type);
      for (const auto &range : shape_range) {
        FE_MUL_OVERFLOW(tensor_size, range.second, tensor_size);
      }
      if (tensor_size < 0) {
        FE_LOGW("Tensor [%s] with index [%zu] of operation [%s, %s] has a negative value, its size is %ld.",
                direction.c_str(), i, op_desc.GetName().c_str(), op_desc.GetType().c_str(), tensor_size);
        return FAILED;
      }
      (void)OpTensorUtils::CalibrateTensorSize(tensor_size);
      FE_LOGD("Tensor %s [%zu] of op[%s, %s] is not tiling, which size is %ld.",
              direction.c_str(), i, op_desc.GetName().c_str(), op_desc.GetType().c_str(), tensor_size);
      return SUCCESS;
    }
    FE_LOGD("Tensor %s [%zu] of op [%s, %s] has a dynamic shape. There is no need to calculate its size.",
            direction.c_str(), i, op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return SUCCESS;
  }

  std::vector<int64_t> dims = shape.GetDims();
  if (OpTensorUtils::CalcTensorSize(dims, data_type, output_real_calc_flag, tensor_size) != SUCCESS) {
    FE_LOGW("Failed to calculate the size of tensor %s [%zu] for operator [%s, %s].", direction.c_str(),
            i, op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TensorSizeCalculator::CalcInputOpTensorSize(const ge::NodePtr &node, const int32_t &output_real_calc_flag) {
  size_t input_size = node->GetOpDesc()->GetAllInputsSize();
  for (size_t i = 0; i < input_size; i++) {
    ge::GeTensorDescPtr tensor_desc_ptr = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    int64_t tensor_size = -1;
    int64_t input_memory_scope = 0;
    (void)ge::AttrUtils::GetInt(tensor_desc_ptr, ge::ATTR_NAME_TENSOR_MEMORY_SCOPE, input_memory_scope);
    (void)ge::AttrUtils::GetInt(tensor_desc_ptr, ge::ATTR_NAME_SPECIAL_INPUT_SIZE, tensor_size);
    FE_LOGD("Node[%s, %s] input[%zu] has an input_memory_scope of %ld, with a special tensor size of %ld.",
            node->GetNamePtr(), node->GetTypePtr(), i, input_memory_scope, tensor_size);
    if (input_memory_scope == kNetEdgeDdrType && tensor_size > -1) {
      auto in_anchor = node->GetInDataAnchor(i);
      FE_CHECK_NOTNULL(in_anchor);
      auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      FE_CHECK_NOTNULL(peer_out_anchor);
      int32_t peer_idx = peer_out_anchor->GetIdx();
      auto peer_node = peer_out_anchor->GetOwnerNode();
      FE_CHECK_NOTNULL(peer_node);
      auto peer_tensor_desc = peer_node->GetOpDesc()->MutableOutputDesc(peer_idx);
      FE_CHECK_NOTNULL(peer_tensor_desc);
      (void)OpTensorUtils::CalibrateTensorSize(tensor_size);
      FE_LOGD("Node[%s, %s], the aligned tensor size for input[%zu] is %ld, peer node[%s, %s], peer index[%d].",
              node->GetNamePtr(), node->GetTypePtr(), i, tensor_size, peer_node->GetNamePtr(),
              peer_node->GetTypePtr(), peer_idx);
      ge::TensorUtils::SetSize(*peer_tensor_desc, tensor_size);
      ge::TensorUtils::SetSize(*tensor_desc_ptr, tensor_size);
      continue;
    }
    Status result = CalcSingleTensorSize(*(node->GetOpDesc()), tensor_desc_ptr, "input", i,
                                         output_real_calc_flag, tensor_size);
    if (result != SUCCESS) {
      continue;
    }
    FE_LOGD("Node[%s, %s], the tensor size of input[%zu] is %ld.", node->GetNamePtr(), node->GetTypePtr(), i,
            tensor_size);
    ge::TensorUtils::SetSize(*tensor_desc_ptr, tensor_size);
  }
  return SUCCESS;
}

bool TensorSizeCalculator::IsEndNoteOuputNode(const ge::OutDataAnchorPtr &anchor) {
  auto peer_in_anchors = anchor->GetPeerInDataAnchors();
  for (auto peer_in_anchor : peer_in_anchors) {
    ge::NodePtr peer_in_node = peer_in_anchor->GetOwnerNode();
    FE_LOGD("The next node is[%s, %s]", peer_in_node->GetNamePtr(), peer_in_node->GetTypePtr());
    if((peer_in_node->GetType() == OP_TYPE_END) || (peer_in_node->GetType() == OP_TYPE_NETOUTPUT)) {
      return true;
    }
  }
  return false;
}

Status TensorSizeCalculator::CalcOutputOpTensorSize(const ge::OpDesc &op_desc, const int32_t &output_real_calc_flag,
                                                    const ge::Node::Vistor<ge::OutDataAnchorPtr> &output_data_anchors) {
  size_t output_size = op_desc.GetOutputsSize();
  for (size_t i = 0; i < output_size; i++) {
    ge::GeTensorDescPtr tensor_desc_ptr = op_desc.MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    int64_t tensor_size = -1;
    Status result = SUCCESS;
    bool is_end_netoutput = IsEndNoteOuputNode(output_data_anchors.at(i));
    FE_LOGD("The current node is [%s, %s], and the next node is either NetOutput or End [%d].",
            op_desc.GetNamePtr(), op_desc.GetTypePtr(), is_end_netoutput);
    if (is_end_netoutput && ge::AttrUtils::GetInt(tensor_desc_ptr, ge::ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size)) {
      if (!output_real_calc_flag) {
        result = OpTensorUtils::CalibrateTensorSize(tensor_size);
      }
    } else {
      result = CalcSingleTensorSize(op_desc, tensor_desc_ptr, "output", i, output_real_calc_flag, tensor_size);
    }

    if (result != SUCCESS) {
      continue;
    }
    FE_LOGD("The tensor size of output[%zu] is %ld.", i, tensor_size);
    ge::TensorUtils::SetSize(*tensor_desc_ptr, tensor_size);
  }
  return SUCCESS;
}
}  // namespace fe
