/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expanddims_kernel.h"

#include <memory>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const int32_t kExpandDimsIndexZero = 0;
const size_t kExpandDimsOutputDescNum = 1;
const size_t kExpandDimsInputNum = 2;
}  // namespace
Status ExpanddimsKernel::Compute(const NodePtr &node_ptr) const {
  GELOGD("Expanddims dimension kernel in.");
  if (node_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "parameter is nullptr");
    return PARAM_INVALID;
  }
  Status ret = KernelUtils::CheckDimensionNodeInfo(node_ptr);
  if (ret != SUCCESS) {
    GELOGW("GetDimensionNodeInfo failed");
    return ret;
  }

  if (!KernelUtils::CheckFormatSupported(node_ptr)) {
    GELOGW("CheckFormatSupported failed");
    return NOT_CHANGED;
  }
  GELOGD("Expanddims dimension kernel success.");
  return SUCCESS;
}
Status ExpanddimsKernel::Compute(const ge::OpDescPtr op_desc_ptr,
                                 const std::vector<ge::ConstGeTensorPtr> &input,
                                 std::vector<ge::GeTensorPtr> &v_output) {
  GELOGD("Expanddims folding kernel in.");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Input opdesc is nullptr.");
    return PARAM_INVALID;
  }
  if ((input.size() != kExpandDimsInputNum) || (op_desc_ptr->GetOutputsSize() != kExpandDimsOutputDescNum)) {
    GELOGW("Unexpected ExpandDims node, node input size: %zu, node output size: %zu, node name: %s", input.size(),
           op_desc_ptr->GetOutputsSize(), op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(kExpandDimsIndexZero);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("Failed to fold node %s, out of memory", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  // print output tensor information, and will be deleted
  GELOGI("Expanddims op %s output tensor data size is %zu.", op_desc_ptr->GetName().c_str(),
         output_ptr->GetData().size());
  size_t data_dim_size = output_ptr->GetTensorDesc().GetShape().GetDims().size();
  GELOGI("Expanddims op %s output tensor dim size is %zu.", op_desc_ptr->GetName().c_str(), data_dim_size);

  if (output_ptr->SetData(input.at(kExpandDimsIndexZero)->GetData()) != GRAPH_SUCCESS) {
    GELOGW("Compute: SetData failed");
  }
  v_output.emplace_back(output_ptr);
  GELOGD("Expanddims folding kernel success.");
  return SUCCESS;
}
REGISTER_COMPUTE_NODE_KERNEL(EXPANDDIMS, ExpanddimsKernel);
}  // namespace ge
