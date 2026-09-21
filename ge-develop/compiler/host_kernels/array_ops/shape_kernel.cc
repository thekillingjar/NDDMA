/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shape_kernel.h"

#include <memory>

#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"
#include "framework/common/types.h"

namespace ge {
namespace {
const size_t kShapeInputSize = 1U;
const size_t kShapeOutputSize = 1U;
REGISTER_COMPUTE_NODE_KERNEL(SHAPE, ShapeKernel);
}  // namespace
Status ShapeKernel::Compute(const NodePtr &node, std::vector<GeTensorPtr> &v_output) const {
  GELOGD("ShapeKernel in");
  if (node == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return FAILED;
  }
  const OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const bool size_check = ((op_desc->GetInputsSize() != kShapeInputSize) ||
      (op_desc->GetOutputsSize() != kShapeOutputSize));
  if (size_check) {
    GELOGW("Size check fail, inputs size:%zu, outputs size:%zu", op_desc->GetInputsSize(), op_desc->GetOutputsSize());
    return NOT_CHANGED;
  }
  const auto &input_desc = op_desc->MutableInputDesc(0U);
  GE_CHECK_NOTNULL(input_desc);
  if (KernelUtils::IsUnknownShape(input_desc->GetShape()) ||
      KernelUtils::IsUnknownShape(input_desc->GetOriginShape())) {
    GELOGW("Input shape is unknown, ignore shape kernel.");
    return NOT_CHANGED;
  }
  const std::vector<int64_t> dims = input_desc->GetOriginShape().GetDims();
  const Status ret = KernelUtils::ConstructTensorDescWithData(op_desc->GetOutputDesc(0U), dims, v_output);
  if (ret != SUCCESS) {
    GELOGE(ret, "Shape kernel construct tensor desc failed!");
    return ret;
  }
  GELOGD("Shape kernel success");
  return SUCCESS;
}
}  // namespace ge
