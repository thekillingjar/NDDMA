/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_tiling/op_tiling_utils.h"
#include <string>
#include "graph/utils/attr_utils.h"

namespace optiling {
void ReplaceEmptyShapeOfTensorDesc(const ge::OpDescPtr &op_desc, std::vector<int32_t> &indexes) {
  const size_t input_size = op_desc->GetAllInputsSize();
  for (size_t i = 0; i < input_size; ++i) {
    const ge::GeTensorDescPtr tensor_desc_ptr = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    if (tensor_desc_ptr->MutableShape().IsScalar()) {
      indexes.push_back(static_cast<int32_t>(i));
      tensor_desc_ptr->MutableShape().SetDimNum(1);
      (void)tensor_desc_ptr->MutableShape().SetDim(0, 1);
    }
  }

  const size_t output_size = op_desc->GetOutputsSize();
  for (size_t i = 0; i < output_size; ++i) {
    const ge::GeTensorDescPtr tensor_desc_ptr = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    if (tensor_desc_ptr->MutableShape().IsScalar()) {
      indexes.push_back(static_cast<int32_t>(-1 - i));
      tensor_desc_ptr->MutableShape().SetDimNum(1);
      (void)tensor_desc_ptr->MutableShape().SetDim(0, 1);
    }
  }
}

void RecoveryEmptyShapeOfTensorDesc(const ge::OpDescPtr &op_desc, const std::vector<int32_t> &indexes) {
  for (const int32_t &index : indexes) {
    ge::GeTensorDescPtr tensor_desc_ptr;
    if (index >= 0) {
      tensor_desc_ptr = op_desc->MutableInputDesc(static_cast<uint32_t>(index));
    } else {
      tensor_desc_ptr = op_desc->MutableOutputDesc(static_cast<uint32_t>(std::abs(index) - 1));
    }
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    tensor_desc_ptr->MutableShape().SetDimNum(0);
  }
}
}  // namespace optiling
