/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reshape_kernel.h"

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"
#include "common/checker.h"

namespace ge {
namespace {
const int32_t kReshapeDataIndex = 0;
const int32_t kOutputDescFirstIndex = 0;
const size_t kReshapeOutputSize = 1;
const size_t kReshapeInputSize = 2;
}  // namespace

Status ReshapeKernel::Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                              std::vector<ge::GeTensorPtr> &v_output) {
  GELOGD("Reshape folding kernel in.");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Input opdesc is nullptr.");
    return PARAM_INVALID;
  }
  if ((input.size() != kReshapeInputSize) || (op_desc_ptr->GetOutputsSize() != kReshapeOutputSize)) {
    GELOGW("Unexpected Reshape node, node input size: %zu, node output size: %zu, node name: %s", input.size(),
           op_desc_ptr->GetOutputsSize(), op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }
  auto input_tensor_desc = op_desc_ptr->GetInputDescPtr(kReshapeDataIndex);
  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(kOutputDescFirstIndex);
  GE_ASSERT_NOTNULL(input_tensor_desc);
  auto in_shape_size = input_tensor_desc->GetShape().GetShapeSize();
  auto out_shape_size = output_tensor_desc.GetShape().GetShapeSize();
  // check input_shape_size should equal with output_shape_size except scalar
  if ((in_shape_size != out_shape_size) && ((in_shape_size + out_shape_size) != 1)) {
    GELOGW("Reshape %s input shape size %ld not match with output shape size %ld", op_desc_ptr->GetName().c_str(),
           in_shape_size, out_shape_size);
    return NOT_CHANGED;
  }
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("Failed to fold node %s, out of memory", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  // print output tensor information, and will be deleted
  GELOGI("Reshape op %s output tensor data size is %zu.", op_desc_ptr->GetName().c_str(), output_ptr->GetData().size());
  size_t data_dim_size = output_ptr->GetTensorDesc().GetShape().GetDims().size();
  GELOGI("Reshape op %s output tensor dim size is %zu.", op_desc_ptr->GetName().c_str(), data_dim_size);

  if (output_ptr->SetData(input.at(kReshapeDataIndex)->GetData()) != GRAPH_SUCCESS) {
    GELOGW("Compute: SetData failed");
  }
  v_output.emplace_back(output_ptr);
  GELOGI("Reshape folding kernel success.");
  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(RESHAPE, ReshapeKernel);
}  // namespace ge
