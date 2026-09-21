/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_kernels/custom_ops/reformat_kernel.h"
#include "formats/utils/formats_trans_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const size_t kReFormatInputSize = 1;
const size_t kReformatFirstInput = 0;
const size_t kReformatFirstOutput = 0;
}  // namespace

Status ReFormatKernel::ValidateInput(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) const {
  GE_CHECK_NOTNULL(op_desc_ptr);
  if (op_desc_ptr->GetInputsSize() != kReFormatInputSize) {
    GELOGW("trans_op has more than 1 input_size.");
    return PARAM_INVALID;
  }
  if (input.empty()) {
    GELOGE(PARAM_INVALID, "Input tensor vector is empty");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status ReFormatKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                               std::vector<GeTensorPtr> &v_output) {
  GELOGD("ReFormatKernel begin.");
  Status status = ValidateInput(op_desc_ptr, input);
  if (status != SUCCESS) {
    return status;
  }

  ConstGeTensorPtr const_weight_ptr = input[kReformatFirstInput];
  if (const_weight_ptr == nullptr) {
    GELOGW("Parameter's invalid, Input_0 is nullptr.");
    return NOT_CHANGED;
  }

  const auto &op_desc = op_desc_ptr->MutableOutputDesc(kReformatFirstOutput);
  const auto &op_desc_in = op_desc_ptr->MutableInputDesc(kReformatFirstInput);
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(op_desc_in);
  const auto &src_shape = op_desc_in->GetShape().GetDims();
  const auto &src_dtype = op_desc_in->GetDataType();
  const auto &dst_shape = op_desc->GetShape().GetDims();
  const auto &dst_dtype = op_desc->GetDataType();
  if (src_dtype != dst_dtype || src_shape != dst_shape) {
    GELOGW("Check params failed. src data type %s and shape %s should be equal to dst data type %s and shape %s",
           TypeUtils::DataTypeToSerialString(src_dtype).c_str(), formats::ShapeToString(src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(dst_dtype).c_str(), formats::ShapeToString(dst_shape).c_str());
    return NOT_CHANGED;
  }
  if (!KernelUtils::CheckSizeForTransOp(const_weight_ptr, op_desc_ptr)) {
    GELOGW("CheckSize failed, input size(shape %s) is not equal to weight size(shape %s)",
           formats::ShapeToString(src_shape).c_str(),
           formats::ShapeToString(const_weight_ptr->GetTensorDesc().GetShape()).c_str());
    return NOT_CHANGED;
  }
  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(kReformatFirstOutput));
  if (output_ptr == nullptr) {
    GELOGW("Create shared ptr for GeTensor failed");
    return NOT_CHANGED;
  }
  GE_IF_BOOL_EXEC(output_ptr->SetData(input.at(0)->GetData()) != GRAPH_SUCCESS,
                  GELOGW("set data failed");
                  return NOT_CHANGED);
  v_output.emplace_back(output_ptr);
  GELOGD("ReFormatKernel success.");
  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(REFORMAT, ReFormatKernel);
}  // namespace ge
