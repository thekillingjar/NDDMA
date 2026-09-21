/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "unsqueeze_kernel.h"
#include <memory>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
constexpr uint32_t kInputDescIndex = 0;
constexpr uint32_t kOutputDescIndex = 0;
constexpr size_t kSqueezeInputSize = 1;
constexpr size_t kSqueezeOutputSize = 1;
}  // namespace

Status UnsqueezeKernel::Compute(const NodePtr &node_ptr) const {
  GE_CHECK_NOTNULL(node_ptr);
  if (!KernelUtils::CheckFormatSupported(node_ptr)) {
    GELOGW("CheckFormatSupported failed");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status UnsqueezeKernel::Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                                std::vector<ge::GeTensorPtr> &v_output) {
  GE_CHECK_NOTNULL(op_desc_ptr);
  GELOGD("SqueezeKernel in: node[%s]", op_desc_ptr->GetName().c_str());
  bool is_check_failed = ((op_desc_ptr->GetInputsSize() != kSqueezeInputSize) ||
                          (op_desc_ptr->GetOutputsSize() != kSqueezeOutputSize) || (input.size() != kSqueezeInputSize));
  if (is_check_failed) {
    GELOGW("Size check fail, node[%s] inputs size:%zu, outputs size:%zu, input size:%zu",
           op_desc_ptr->GetName().c_str(), op_desc_ptr->GetInputsSize(), op_desc_ptr->GetOutputsSize(), input.size());
    return NOT_CHANGED;
  }

  auto tensor_desc = op_desc_ptr->GetOutputDesc(kOutputDescIndex);
  GeTensorPtr output_ptr = MakeShared<ge::GeTensor>(tensor_desc);
  GE_CHECK_NOTNULL(output_ptr);

  auto input_tensor = input.at(kInputDescIndex);
  GE_CHECK_NOTNULL(input_tensor);

  if (output_ptr->SetData(input_tensor->GetData()) != GRAPH_SUCCESS) {
    GELOGW("Compute: SetData failed");
  }
  v_output.emplace_back(output_ptr);
  GELOGD("UnsqueezeKernel success: node[%s]", op_desc_ptr->GetName().c_str());
  return SUCCESS;
}
REGISTER_COMPUTE_NODE_KERNEL(UNSQUEEZE, UnsqueezeKernel);
REGISTER_COMPUTE_NODE_KERNEL(UNSQUEEZEV2, UnsqueezeKernel);
}  // namespace ge
