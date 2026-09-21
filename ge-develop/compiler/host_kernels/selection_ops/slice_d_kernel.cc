/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slice_d_kernel.h"

#include <memory>

#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const int64_t kDimMinusOne = -1;
const int64_t kDimZero = 0;
const int64_t KStrideLengthOne = 1;
const size_t kSliceDInputSize = 1;
const size_t kSliceDOutputSize = 1;
const char *const kSliceDAttrBegin = "offsets";
const char *const kSliceDAttrSize = "size";
}  // namespace
Status SliceDKernel::SliceDCheck(const OpDescPtr &op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                                 std::vector<int64_t> &begin_list, std::vector<int64_t> &size_list) const {
  // Check input size and output size
  if ((input.size() != kSliceDInputSize) || (op_desc_ptr->GetInputsSize() != kSliceDInputSize) ||
      (op_desc_ptr->GetOutputsSize() != kSliceDOutputSize)) {
    GELOGW("Unexpected SliceD node, node input size: %zu, node output size: %zu, node name: %s.", input.size(),
           op_desc_ptr->GetOutputsSize(), op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  ConstGeTensorPtr x_tensor = input.at(0);  // index 0 is guaranteed to be valid by input size check.
  if (x_tensor == nullptr) {
    GELOGW("SliceDKernel input tensor is nullptr.");
    return PARAM_INVALID;
  }
  // Check data
  if (x_tensor->GetData().size() == 0) {
    GELOGW("SliceDKernel data size of input is 0, node name: %s.", op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }

  // Get attr;
  if (!AttrUtils::GetListInt(op_desc_ptr, kSliceDAttrBegin, begin_list)) {
    GELOGW("SliceDKernel get attr begin failed, node name: %s.", op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetListInt(op_desc_ptr, kSliceDAttrSize, size_list)) {
    GELOGW("SliceDKernel get attr size failed, node name: %s.", op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  // Check attr;
  std::vector<int64_t> x_dims = x_tensor->GetTensorDesc().GetShape().GetDims();
  size_t x_dim_size = x_dims.size();
  if (x_dim_size != begin_list.size() || x_dim_size != size_list.size()) {
    GELOGW("SliceDKernel rank of all shapes must be the same, input: %zu, begin: %zu, size: %zu, node name: %s.",
           x_dim_size, begin_list.size(), size_list.size(), op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  for (size_t i = 0; i < x_dim_size; i++) {
    int64_t x_dim_i = x_dims[i];
    int64_t begin_i = begin_list[i];
    int64_t size_i = size_list[i];
    if ((begin_i < kDimZero) || (begin_i > x_dim_i) || (size_i < kDimMinusOne) || (size_i > x_dim_i)) {
      GELOGW(
          "SliceDKernel dim[%zu] of attr is out of range, node name: %s, begin_i is %ld, valid range[0, %ld], size_i "
          "is %ld, valid range[-1, %ld].",
          i, op_desc_ptr->GetName().c_str(), begin_i, x_dim_i, size_i, x_dim_i);
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status SliceDKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                             std::vector<GeTensorPtr> &v_output) {
  GELOGD("SliceDKernel in");
  if (op_desc_ptr == nullptr) {
    GELOGW("SliceDKernel input opdesc is nullptr.");
    return NOT_CHANGED;
  }

  std::vector<int64_t> begin_list;
  std::vector<int64_t> size_list;
  if (SliceDCheck(op_desc_ptr, input, begin_list, size_list) != SUCCESS) {
    GELOGW("SliceDKernel input is invalid, failed to fold node.");
    return NOT_CHANGED;
  }

  ConstGeTensorPtr x_tensor = input.at(0);  // index 0 is guaranteed to be valid by input size check.
  std::vector<int64_t> x_dims = x_tensor->GetTensorDesc().GetShape().GetDims();
  std::vector<int64_t> stride_list;

  bool has_zero_dim = false;
  for (size_t i = 0; i < x_dims.size(); i++) {
    int64_t x_dim_i = x_dims[i];
    int64_t begin_i = begin_list[i];
    int64_t size_i = size_list[i];
    if (size_i == kDimMinusOne) {
      size_i = x_dim_i - begin_i;
      size_list[i] = size_i;
    } else if (begin_i + size_i > x_dim_i) {
      GELOGW(
          "SliceDKernel dim[%zu] of attr size is out of range, node name: %s, begin_i:%ld, size_i:%ld, x_dim_i:%ld, "
          "x_dims:%s.",
          i, op_desc_ptr->GetNamePtr(), begin_i, size_i, x_dim_i, ToString(x_dims).c_str());
      return NOT_CHANGED;
    }
    stride_list.push_back(KStrideLengthOne);

    // 0 appears in dims of input tensor or size tensor
    if (size_i == kDimZero || x_dim_i == kDimZero) {
      has_zero_dim = true;
    }
  }

  auto x_data_type = x_tensor->GetTensorDesc().GetDataType();
  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(0);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("Failed to fold node %s, out of memory", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  output_ptr->MutableTensorDesc().SetShape(GeShape(size_list));
  output_ptr->MutableTensorDesc().SetDataType(x_data_type);
  if (has_zero_dim) {
    v_output.emplace_back(output_ptr);
    GELOGI("SliceD folding kernel success, and output tensor has no data.");
    return SUCCESS;
  }

  void *data = reinterpret_cast<void *>(const_cast<uint8_t *>(x_tensor->GetData().data()));
  int64_t x_data_size = x_tensor->GetTensorDesc().GetShape().GetShapeSize();

  Status ret = CheckOutputDims(size_list, op_desc_ptr);
  if (ret != SUCCESS) {
      return ret;
  }

  ret = OpUtils::SetOutputSliceData(data, x_data_size, x_data_type, x_dims, begin_list, size_list,
                                    output_ptr.get(), stride_list);
  if (ret != SUCCESS) {
    GELOGW("Set output data of SliceD failed.");
    return NOT_CHANGED;
  }

  v_output.emplace_back(output_ptr);
  GELOGI("SliceD folding kernel success.");
  return SUCCESS;
}

Status SliceDKernel::CheckOutputDims(const std::vector<int64_t> &output_dims, const OpDescPtr attr) const {
  // check dim not all less than 0
  for (auto dim : output_dims) {
    if (dim > 0) {
      return SUCCESS;
    }
  }
  GELOGW("all output dim <=0, can't be processed. op_name : %s", attr->GetName().c_str());
  return NOT_CHANGED;
}

REGISTER_COMPUTE_NODE_KERNEL(SLICED, SliceDKernel);
}  // namespace ge
