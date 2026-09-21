/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flattenv2_kernel.h"

#include <memory>

#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"
#include "framework/common/types.h"

namespace ge {
namespace {
const size_t kFirstDataIndex = 0UL;
const size_t kValidSize = 1UL;

void GetAndConvertAxis(const OpDescPtr &op_desc_ptr, int64_t &axis, int64_t &end_axis) {
  if (!AttrUtils::GetInt(op_desc_ptr, "axis", axis)) {
    axis = 1;
  }
  if (!AttrUtils::GetInt(op_desc_ptr, "end_axis", end_axis)) {
    end_axis = -1;
  }
  GeTensorDesc x_desc = op_desc_ptr->GetInputDesc("x");
  int64_t dim_count = static_cast<int64_t>(x_desc.GetShape().GetDimNum());
  if (axis < 0) {
    axis += dim_count;
  }
  if (end_axis < 0) {
    end_axis += dim_count;
  }
}

bool IsFlattenV2ParamsValid(const OpDescPtr &op_desc_ptr) {
  size_t input_size = op_desc_ptr->GetInputsSize();
  size_t output_size = op_desc_ptr->GetOutputsSize();
  if (input_size != kValidSize || output_size != kValidSize) {
    GELOGE(PARAM_INVALID, "input_size or output_size is invalid");
    return false;
  }

  GeTensorDesc x_desc = op_desc_ptr->GetInputDesc("x");
  if (KernelUtils::IsUnknownShape(x_desc.GetShape())) {
    GELOGE(FAILED, "shape is unknown.");
    return false;
  }

  int64_t axis = 0;
  int64_t end_axis = 0;
  GetAndConvertAxis(op_desc_ptr, axis, end_axis);
  const int64_t dim_num = static_cast<int64_t>(x_desc.GetShape().GetDimNum());
  if (axis < 0 || axis >= dim_num) {
    GELOGE(PARAM_INVALID, "axis out of range! axis is %ld", axis);
    return false;
  }
  if (end_axis < 0 || end_axis >= dim_num) {
    GELOGE(PARAM_INVALID, "end_axis out of range! end_axis is %ld", end_axis);
    return false;
  }
  if (axis > end_axis) {
    GELOGE(PARAM_INVALID, "axis after end_axis! axis is %ld, end_axis is %ld", axis, end_axis);
    return false;
  }
  return true;
}

void ComputeShape(const GeTensorDesc &x_desc, const OpDescPtr &op_desc_ptr, GeTensorDesc &y_desc) {
  const auto x_shape_dim = x_desc.GetShape().GetDims();
  int64_t axis = 0;
  int64_t end_axis = 0;
  GetAndConvertAxis(op_desc_ptr, axis, end_axis);
  std::vector<int64_t> y_shape_dim;
  for (int64_t i = 0; i < axis; i++) {
    y_shape_dim.emplace_back(x_shape_dim[i]);
  }
  int64_t dim_val = 1;
  for (int64_t i = axis; i < (end_axis + 1); i++) {
    dim_val = dim_val * x_shape_dim[i];
  }
  y_shape_dim.emplace_back(dim_val);

  for (int64_t i = (end_axis + 1); i <static_cast<int64_t>(x_shape_dim.size()); i++) {
    y_shape_dim.emplace_back(x_shape_dim[i]);
  }

  GeShape y_shape(y_shape_dim);
  y_desc.SetShape(y_shape);
}
}  // namespace

Status FlattenV2Kernel::Compute(const NodePtr &node_ptr) const {
  GELOGD("FlattenV2 dimension kernel in");
  GE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc_ptr);
  if (!IsFlattenV2ParamsValid(op_desc_ptr)) {
    GELOGW("Params are invalid");
    return NOT_CHANGED;
  }
  GELOGD("FlattenV2 dimension kernel success.");
  return SUCCESS;
}

Status FlattenV2Kernel::Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                                std::vector<ge::GeTensorPtr> &v_output) {
  GELOGD("FlattenV2 folding kernel in.");
  GE_CHECK_NOTNULL(op_desc_ptr);
  if (!IsFlattenV2ParamsValid(op_desc_ptr)) {
    GELOGW("Params are invalid");
    return NOT_CHANGED;
  }

  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(kFirstDataIndex);
  auto input_desc = op_desc_ptr->GetInputDesc(kFirstDataIndex);
  ComputeShape(input_desc, op_desc_ptr, output_tensor_desc);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("Failed to fold node %s, out of memory", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }
  GELOGI("FlattenV2 op %s output tensor data size is %zu", op_desc_ptr->GetName().c_str(),
         output_ptr->GetData().size());

  size_t data_dim_size = output_ptr->GetTensorDesc().GetShape().GetDims().size();
  GELOGI("FlattenV2 op %s output tensor dim size is %zu", op_desc_ptr->GetName().c_str(), data_dim_size);
  if (output_ptr->SetData(input.at(kFirstDataIndex)->GetData()) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Compute: SetData failed");
    return FAILED;
  }
  v_output.emplace_back(output_ptr);
  GELOGD("FlattenV2 folding kernel success.");
  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(FLATTENV2, FlattenV2Kernel);
}  // namespace ge
