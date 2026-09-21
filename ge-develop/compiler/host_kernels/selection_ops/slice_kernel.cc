/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slice_kernel.h"

#include <set>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const size_t kSliceInputSize = 3;
const size_t kSliceInputIndexX = 0;
const size_t kSliceInputIndexBegin = 1;
const size_t kSliceInputIndexSize = 2;
const std::set<DataType> kIndexNumberType = {DT_INT32, DT_INT64};
const std::set<ge::DataType> kSupportedDataTypeToLength = {
    DT_BOOL,
    DT_INT64,
    DT_UINT64,
    DT_FLOAT,
    DT_INT32,
    DT_UINT32,
    DT_INT8,
    DT_UINT8,
    DT_INT16,
    DT_UINT16,
    DT_FLOAT16,
    DT_DOUBLE,
    DT_DUAL,
    DT_DUAL_SUB_INT8,
    DT_DUAL_SUB_UINT8,
    DT_COMPLEX64,
    DT_COMPLEX128,
    DT_QINT8,
    DT_QINT16,
    DT_QINT32,
    DT_QUINT8,
    DT_QUINT16,
};
struct SliceDataParam {
  std::vector<int64_t> input_dims;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> begin_vec;
  std::vector<int64_t> stride_vec;
};
// stride means begin and size input of slice
void GetValueOfStride(const std::vector<ge::ConstGeTensorPtr> &input, std::vector<int64_t> &orig_begin_vec,
                      std::vector<int64_t> &orig_size_vec) {
  const ConstGeTensorPtr &begin_tensor = input[kSliceInputIndexBegin];
  const ConstGeTensorPtr &size_tensor = input[kSliceInputIndexSize];

  const auto data_type = begin_tensor->GetTensorDesc().GetDataType();
  const size_t vec_size = begin_tensor->GetData().size() / static_cast<size_t>(GetSizeByDataType(data_type));
  if (data_type == DT_INT32) {
    const int32_t *begin = reinterpret_cast<const int32_t *>(begin_tensor->GetData().data());
    const int32_t *size = reinterpret_cast<const int32_t *>(size_tensor->GetData().data());
    for (size_t i = 0; i < vec_size; ++i) {
      orig_begin_vec.emplace_back(begin[i]);
      orig_size_vec.emplace_back(size[i]);
    }
  } else {
    const int64_t *begin = reinterpret_cast<const int64_t *>(begin_tensor->GetData().data());
    const int64_t *size = reinterpret_cast<const int64_t *>(size_tensor->GetData().data());
    for (size_t i = 0; i < vec_size; ++i) {
      orig_begin_vec.emplace_back(begin[i]);
      orig_size_vec.emplace_back(size[i]);
    }
  }
}
Status GetSliceDataParams(const ConstGeTensorPtr &x_tensor, const std::vector<int64_t> &orig_begin_vec,
                          const std::vector<int64_t> &orig_size_vec,
                          SliceDataParam &slice_data_param) {
  const ge::GeShape &x_shape = x_tensor->GetTensorDesc().GetShape();
  const size_t dim_size = x_shape.GetDimNum();
  if (dim_size != orig_begin_vec.size() || dim_size != orig_size_vec.size()) {
    GELOGW("Rank of x input %zu not match with offset_size(%zu) or size_input size (%zu)", dim_size, orig_begin_vec.size(), orig_size_vec.size());
    return NOT_CHANGED;
  }

  for (size_t i = 0; i < dim_size; ++i) {
    int64_t begin_i = orig_begin_vec[i];
    int64_t size_i = orig_size_vec[i];
    int64_t dim_i = x_shape.GetDim(i);
    if (size_i < 0) {
      GE_IF_BOOL_EXEC(((dim_i - begin_i) > INT32_MAX) || ((dim_i - begin_i) < INT32_MIN),
                      GELOGE(PARAM_INVALID, " %ld and %ld sub can result in overflow!.", dim_i, begin_i);
                      return INTERNAL_ERROR);
      size_i = dim_i - begin_i;
    }
    slice_data_param.input_dims.push_back(dim_i);
    slice_data_param.output_dims.push_back(size_i);
    slice_data_param.stride_vec.push_back(1);
  }
  slice_data_param.begin_vec = orig_begin_vec;
  return SUCCESS;
}
}  // namespace

Status SliceKernel::Compute(const OpDescPtr attr, const std::vector<ConstGeTensorPtr> &input,
                            std::vector<GeTensorPtr> &v_output) {
  GELOGD("SliceKernel in.");
  if (attr == nullptr) {
    GELOGW("Input opdescptr is nullptr.");
    return NOT_CHANGED;
  }
  // check input size
  if (input.size() != kSliceInputSize) {
    GELOGW("The number of input for slice must be %zu.", kSliceInputSize);
    return NOT_CHANGED;
  }
  Status ret = CheckInputDatatypeSupported(input);
  if (ret != SUCCESS) {
    return ret;
  }

  std::vector<int64_t> begin_vec;
  std::vector<int64_t> orig_size_vec;
  GetValueOfStride(input, begin_vec, orig_size_vec);

  ConstGeTensorPtr x_tensor = input[kSliceInputIndexX];
  SliceDataParam slice_data_param;
  ret = GetSliceDataParams(x_tensor, begin_vec, orig_size_vec, slice_data_param);
  if (ret != SUCCESS) {
    return ret;
  }

  ret = CheckOutputDims(slice_data_param.output_dims, attr);
  if (ret != SUCCESS) {
    return ret;
  }

  // construct tensorDesc
  ge::GeShape output_shape(slice_data_param.output_dims);
  auto attr_output_tensor_desc = attr->GetOutputDesc(0);
  GeTensorDesc output_tensor_desc(attr_output_tensor_desc);
  output_tensor_desc.SetShape(output_shape);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("make_shared ge::GeTensor failed, node name %s.", attr->GetName().c_str());
    return NOT_CHANGED;
  }

  void *data = const_cast<uint8_t *>(x_tensor->GetData().data());
  GE_CHECK_NOTNULL(data);
  // datatype/ type_size checked before
  auto data_type = x_tensor->GetTensorDesc().GetDataType();
  uint32_t type_size = 0;
  (void) TypeUtils::GetDataTypeLength(data_type, type_size);
  size_t data_size = x_tensor->GetData().size() / type_size;
  ret = OpUtils::SetOutputSliceData(data, static_cast<int64_t>(data_size), data_type, slice_data_param.input_dims,
                                    slice_data_param.begin_vec, slice_data_param.output_dims, output_ptr.get(),
                                    slice_data_param.stride_vec);
  if (ret != SUCCESS) {
    GELOGW("SetOutputSliceData failed.");
    return NOT_CHANGED;
  }
  v_output.push_back(output_ptr);
  GELOGD("SliceKernel success.");
  return SUCCESS;
}

Status SliceKernel::CheckInputDatatypeSupported(const std::vector<ConstGeTensorPtr> &input) const {
  ConstGeTensorPtr x_tensor = input[kSliceInputIndexX];
  ConstGeTensorPtr begin = input[kSliceInputIndexBegin];
  ConstGeTensorPtr size = input[kSliceInputIndexSize];
  if (x_tensor == nullptr || begin == nullptr || size == nullptr) {
    GELOGW("input tensor is nullptr.");
    return NOT_CHANGED;
  }

  // check supported data type in input_x
  auto data_type = x_tensor->GetTensorDesc().GetDataType();
  if (kSupportedDataTypeToLength.count(data_type) == 0) {
    GELOGW("input_x data_type is [%s], does not supported!", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }
  uint32_t type_size = 0;
  bool is_success = TypeUtils::GetDataTypeLength(data_type, type_size);
  if (!is_success) {
    return NOT_CHANGED;
  }

  // check supported data type in input_begin
  if (kIndexNumberType.find(begin->GetTensorDesc().GetDataType()) == kIndexNumberType.end()) {
    GELOGW("Data type of StridedSlice OP(begin) must be int32 or int64");
    return NOT_CHANGED;
  }
  // check supported data type in input_size
  if (kIndexNumberType.find(size->GetTensorDesc().GetDataType()) == kIndexNumberType.end()) {
    GELOGW("Data type of StridedSlice OP(size) must be int32 or int64");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status SliceKernel::CheckOutputDims(const std::vector<int64_t> &output_dims, const OpDescPtr attr) const {
  // check dim not all less than 0
  for (auto dim : output_dims) {
    if (dim > 0) {
      return SUCCESS;
    }
  }
  GELOGW("all output dim <=0, can't be processed. op_name : %s", attr->GetName().c_str());
  return NOT_CHANGED;
}

REGISTER_COMPUTE_NODE_KERNEL(SLICE, SliceKernel);
}  // namespace ge
