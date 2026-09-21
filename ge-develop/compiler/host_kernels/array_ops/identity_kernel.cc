/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "identity_kernel.h"
#include "host_kernels/kernel_factory.h"
#include "framework/common/types.h"

namespace {
constexpr uint32_t kInputDescIndex = 0;
constexpr uint32_t kOutputDescIndex = 0;
}  // namespace

namespace ge {
Status IdentityKernel::Compute(const ge::OpDescPtr op_desc, const std::vector<ge::ConstGeTensorPtr> &input,
                               std::vector<ge::GeTensorPtr> &v_output) {
  if (op_desc == nullptr) {
    GELOGE(PARAM_INVALID, "IdentityKernel op_desc is null.");
    return NOT_CHANGED;
  }
  if (input.empty()) {
    GELOGE(PARAM_INVALID, "Node [%s] inputs is empty.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }
  if (op_desc->GetOutputsSize() < 1) {
    GELOGE(PARAM_INVALID, "Node [%s] output is empty.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }
  GELOGD("IdentityKernel in: node[%s]", op_desc->GetName().c_str());

  auto out_tensor_desc = op_desc->GetOutputDesc(kOutputDescIndex);
  GeTensorPtr output_ptr = MakeShared<ge::GeTensor>(out_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Node [%s] make shared failed.", op_desc->GetName().c_str());
    return OUT_OF_MEMORY;
  }
  auto input_tensor_ptr = input.at(kInputDescIndex);
  if (input_tensor_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Node [%s] get input failed.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }
  if (output_ptr->SetData(input_tensor_ptr->GetData()) != GRAPH_SUCCESS) {
    GELOGW("Compute: SetData failed");
    return NOT_CHANGED;
  }
  v_output.emplace_back(output_ptr);
  GELOGD("IdentityKernel success: node[%s]", op_desc->GetName().c_str());

  return SUCCESS;
}
REGISTER_COMPUTE_NODE_KERNEL(IDENTITY, IdentityKernel);
REGISTER_COMPUTE_NODE_KERNEL(PLACEHOLDERWITHDEFAULT, IdentityKernel);
}  // namespace ge
