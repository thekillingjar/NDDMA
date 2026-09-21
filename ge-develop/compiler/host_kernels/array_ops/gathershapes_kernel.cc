/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gathershapes_kernel.h"

#include <memory>

#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"
#include "framework/common/types.h"

namespace ge {
namespace {
const size_t kInvalidInputSize = 0UL;
const size_t kOutputSize = 1UL;
const size_t kAxesSize = 2UL;

Status CheckAxisSize(const std::vector<std::vector<int64_t>> &axes, const OpDescPtr &op_desc) {
  for (const auto &axis : axes) {
    if (axis.size() != kAxesSize) {
      GELOGE(FAILED, "attr [axes] is invalid, axis is %zu in axes ", axis.size());
      return FAILED;
    }

    const auto &input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(axis[0U]));
    if ((axis[0U] >= static_cast<int64_t>(op_desc->GetInputsSize())) ||
        (axis[1U] >= static_cast<int64_t>(input_desc->MutableShape().GetDimNum()))) {
      GELOGE(FAILED, "attr [axes] is invalid, axis[0] is %ld, axis[1] is %ld", axis[0U], axis[1U]);
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace
Status GatherShapesKernel::Compute(const NodePtr &node, std::vector<GeTensorPtr> &v_output) const {
  GELOGD("GatherShapesKernel in");
  GE_CHECK_NOTNULL(node);
  const OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::vector<std::vector<int64_t>> axes;
  if (!ge::AttrUtils::GetListListInt(op_desc, "axes", axes)) {
    GELOGE(FAILED, "Get axes failed.");
    return FAILED;
  }
  const bool is_size_valid =
      ((op_desc->GetInputsSize() == kInvalidInputSize) || (op_desc->GetOutputsSize() != kOutputSize) || axes.empty());
  if (is_size_valid) {
    GELOGE(FAILED, "Size check fail, inputs size:%zu, outputs size:%zu, attr size:%zu", op_desc->GetInputsSize(),
           op_desc->GetOutputsSize(), axes.size());
    return FAILED;
  }

  for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
    const auto &input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(input_desc);
    if (input_desc->GetShape().GetDims() == UNKNOWN_RANK) {
      GELOGW("[GatherShapes kernel] Input shape is unknown, ignore shape kernel.");
      return NOT_CHANGED;
    }
  }

  Status ret = CheckAxisSize(axes, op_desc);
  if (ret != SUCCESS) {
    return ret;
  }

  std::vector<int64_t> output_shape;
  for (size_t i = 0U; i < axes.size(); ++i) {
    const auto &input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(axes[i][0U]));
    const int64_t ele = static_cast<int64_t>(input_desc->MutableShape().GetDim(static_cast<size_t>(axes[i][1U])));
    output_shape.emplace_back(ele);
  }
  if (std::find(output_shape.begin(), output_shape.end(), UNKNOWN_DIM) != output_shape.end()) {
    GELOGW("[GatherShapes kernel] gathershapes has unknown rank, ignore shape kernel.");
    return NOT_CHANGED;
  }
  ret = KernelUtils::ConstructTensorDescWithData(op_desc->GetOutputDesc(0U), output_shape, v_output);
  if (ret != SUCCESS) {
    GELOGE(ret, "GatherShapes kernel construct tensor desc failed!");
    return ret;
  }
  GELOGD("GatherShapes kernel success");
  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(GATHERSHAPES, GatherShapesKernel);
}  // namespace ge
