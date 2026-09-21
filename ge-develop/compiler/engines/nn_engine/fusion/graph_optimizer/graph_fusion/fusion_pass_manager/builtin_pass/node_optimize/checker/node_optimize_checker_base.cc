/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/checker/node_optimize_checker_base.h"
namespace fe {
bool NodeOptimizeCheckerBase::IsDimC(const ge::NodePtr &node_ptr, const string &dim_attr, bool is_input) const {
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  string node_name = node_ptr->GetName();
  // 1. get the dim attribute
  int dim_attr_value = 0;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, dim_attr, dim_attr_value)) {
    FE_LOGD("Node[%s]: Unable to retrieve the concat_dim attribute. Check unsuccessful.", node_name.c_str());
    return false;
  }
  ge::OpDesc::Vistor<ge::GeTensorDesc> all_tensor_desc =
      is_input ? op_desc_ptr->GetAllInputsDesc() : op_desc_ptr->GetAllOutputsDesc();

  int i = 0;
  for (auto &tensor_desc : all_tensor_desc) {
    // 2. get the postion of the c axis
    int pos = 0;
    Status status = GetPosOfDimC(tensor_desc, pos);
    if (status != SUCCESS) {
      FE_LOGD("Node[%s]: get the dim_c position of the input [%d] unsuccessfully.", node_name.c_str(), i);
      return false;
    }

    // 3. if the dim_attr_value < 0, add the dim_num
    int dim_num = tensor_desc.GetOriginShape().GetDimNum();
    if (dim_attr_value < 0) {
      dim_attr_value += dim_num;
    }
    if (pos != dim_attr_value) {
      FE_LOGD("Node[%s]: the dim_c position of the input [%d] does not match concat_dim, check unsuccessful.",
          node_name.c_str(), i);
      return false;
    }
    i++;
  }
  return true;
}
Status NodeOptimizeCheckerBase::GetDimC(const ge::GeTensorDesc &tensor_desc, int &dim_c) const {
  int pos = 0;
  Status status = GetPosOfDimC(tensor_desc, pos);
  if (status != SUCCESS) {
    FE_LOGD("Unable to get the position of dim C.");
    return FAILED;
  }
  ge::GeShape shape = tensor_desc.GetOriginShape();
  int dim_num = shape.GetDimNum();
  if (pos + 1 > dim_num) {
    FE_LOGD("The position + 1 exceeds dim_num. The position is [%d] and dim_num is [%d].", pos, dim_num);
    return FAILED;
  }
  dim_c = shape.GetDim(pos);
  return SUCCESS;
}

Status NodeOptimizeCheckerBase::GetPosOfDimC(const ge::GeTensorDesc &tensor_desc, int &pos) const {
  ge::Format origin_format = tensor_desc.GetOriginFormat();
  if (origin_format == ge::FORMAT_NCHW) {
    pos = NCHW_DIM_C;
  } else if (origin_format == ge::FORMAT_NHWC) {
    pos = NHWC_DIM_C;
  } else if (origin_format == ge::FORMAT_HWCN) {
    pos = HWCN_DIM_C;
  } else if (origin_format == ge::FORMAT_CHWN) {
    pos = CHWN_DIM_C;
  } else {
    FE_LOGD("Unsupported origin_format: [%d].", origin_format);
    return FAILED;
  }
  return SUCCESS;
}

bool NodeOptimizeCheckerBase::IsDimCOfInputAligned(const ge::GeTensorDesc &tensor_desc, const int &dim_c,
                                                   const ge::DataType &quant_data_type) const {
  ge::DataType origin_data_type = tensor_desc.GetOriginDataType();
  if (quant_data_type == ge::DT_INT8 || quant_data_type == ge::DT_INT4) {
    origin_data_type = quant_data_type;
  }
  if (origin_data_type == ge::DT_FLOAT16 || origin_data_type == ge::DT_FLOAT) {
    return dim_c % 16 == 0;
  } else if (origin_data_type == ge::DT_INT8) {
    return dim_c % 32 == 0;
  } else if (origin_data_type == ge::DT_INT4) {
    return dim_c % 64 == 0;
  }
  FE_LOGD("Unsupported origin_data_type: [%d].", origin_data_type);
  return false;
}

bool NodeOptimizeCheckerBase::IsInputNotData(const ge::NodePtr &node_ptr) const {
  auto input_nodes = node_ptr->GetInDataNodes();
  if (input_nodes.empty()) {
    return false;
  }
  ge::NodePtr input_node = input_nodes.at(0);
  if (input_node == nullptr) {
    return false;
  }
  if (IsRootGraphData(input_node->GetOpDesc()->GetType())) {
    return false;
  }
  return true;
}
}  // namespace fe
