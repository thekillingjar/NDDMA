/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shape_n_kernel.h"

#include <memory>

#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"
#include "framework/common/types.h"

namespace ge {
namespace {
REGISTER_COMPUTE_NODE_KERNEL(SHAPEN, ShapeNKernel);
}  // namespace

Status ShapeNKernel::Compute(const NodePtr &node, std::vector<GeTensorPtr> &v_output) const {
  GELOGD("ShapeN kernel in");
  if (node == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return FAILED;
  }
  const OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetAllInputsSize() != op_desc->GetAllOutputsDescSize()) {
    GELOGW("ShapeN kernel, input and output are not the same size. Input size:%zu, output size:%zu",
           op_desc->GetAllInputsSize(), op_desc->GetAllOutputsDescSize());
    return NOT_CHANGED;
  }

  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); i++) {
    const auto &input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(input_desc);
    if (KernelUtils::IsUnknownShape(input_desc->GetShape())) {
      GELOGW("Input %zu shape is unknown, ignore shape_n kernel.", i);
      return NOT_CHANGED;
    }
    const std::vector<int64_t> dims = input_desc->GetShape().GetDims();
    const Status ret =
        KernelUtils::ConstructTensorDescWithData(op_desc->GetOutputDesc(static_cast<uint32_t>(i)), dims, v_output);
    if (ret != SUCCESS) {
      GELOGE(PARAM_INVALID, "ShapeN kernel construct tensor desc failed, i:%zu", i);
      return ret;
    }
  }

  GELOGD("ShapeN kernel success");
  return SUCCESS;
}
}  // namespace ge
