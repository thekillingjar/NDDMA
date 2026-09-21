/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/spacesize_calculator/spacesize_calculator.h"

#include "common/fe_inner_attr_define.h"
#include "common/fe_error_code.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "graph/utils/tensor_utils.h"
#include "param_calculate/tensorsize_calculator.h"

namespace fe {
SpaceSizeCalculator::SpaceSizeCalculator() {}

SpaceSizeCalculator::~SpaceSizeCalculator() {}

Status SpaceSizeCalculator::CalculateRunningParams(const ge::ComputeGraph &graph) const {
  FE_LOGD("Begin calculating running parameters for each operation in graph [%s].", graph.GetName().c_str());

  for (auto &node_ptr : graph.GetDirectNode()) {
    FE_CHECK_NOTNULL(node_ptr);
    string op_type = node_ptr->GetType();
    if (op_type == OP_TYPE_PLACE_HOLDER || op_type == OP_TYPE_END) {
      continue;
    }
    ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);

    (void)TensorSizeCalculator::CalculateOpTensorSize(node_ptr);
  }
  FE_LOGD("Finish calculating the running parameters of each operation in graph [%s].", graph.GetName().c_str());
  return SUCCESS;
}

Status SpaceSizeCalculator::CalculateAICoreRunningParams(const ge::ComputeGraph &graph) const {
  FE_LOGD("Begin calculating running parameters for each operation in graph [%s].", graph.GetName().c_str());

  for (auto node_ptr : graph.GetDirectNode()) {
    FE_CHECK_NOTNULL(node_ptr);
    string op_type = node_ptr->GetType();
    if (op_type == OP_TYPE_PLACE_HOLDER || op_type == OP_TYPE_END) {
      continue;
    }
    ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);

    // only deal aicore node
    if (!ge::AttrUtils::HasAttr(op_desc_ptr, FE_IMPLY_TYPE)) {
      continue;
    }

    Status status = TensorSizeCalculator::CalculateOpTensorSize(node_ptr);
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][CalcTensorSize][CalcAicoreRunPara] Node[%s, %s]: failed to calculate tensor size.",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return status;
    }
  }
  FE_LOGD("Graph[%s]: completed calculation of running parameters for each operation.", graph.GetName().c_str());
  return SUCCESS;
}

}  // namespace fe
