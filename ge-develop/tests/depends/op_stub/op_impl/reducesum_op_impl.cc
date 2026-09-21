/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "common/checker.h"
#include "graph/utils/type_utils.h"

using namespace gert;
namespace ops {
template <typename T1, typename T2>
bool IsDimValid(const T1 shape_size, const T2 dim_value) {
  int64_t minimum_num = static_cast<int64_t>(shape_size) * (-1);
  int64_t maximum_num = static_cast<int64_t>(shape_size) - 1;

  return static_cast<int64_t>(dim_value) >= minimum_num && static_cast<int64_t>(dim_value) <= maximum_num;
}

template <typename T>
ge::graphStatus ReduceDimsWithKeepDims(const gert::Shape* x_shape, const T* axes_dims, int32_t axes_size,
                                       gert::Shape* output_shape) {
  T dim_num = x_shape->GetDimNum();
  *output_shape = *x_shape;
  for (int32_t i = 0; i < axes_size; i++) {
    if (IsDimValid(dim_num, axes_dims[i])) {
      GELOGE(ge::PARAM_INVALID, "axes_dims[%d] is invalid", i);
      return ge::PARAM_INVALID;
    }
    T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
    output_shape->SetDim(dim, 1);
  }
  GELOGD("ReduceDimsWithKeepDims is SUCCESS");
  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDimsWithoutKeepDims(const gert::Shape* x_shape, const T* axes_dims, int32_t axes_size,
                                          gert::Shape* output_shape) {
  T dim_num = x_shape->GetDimNum();
  output_shape->SetDimNum(0);
  for (T j = 0; j < dim_num; j++) {
    bool reduce_flag = false;
    for (int32_t i = 0; i < axes_size; i++) {
      if (IsDimValid(dim_num, axes_dims[i])) {
        GELOGE(ge::PARAM_INVALID, "axes_dims[%d] is invalid", i);
        return ge::PARAM_INVALID;
      }
      T dim = axes_dims[i] < 0 ? axes_dims[i] + dim_num : axes_dims[i];
      if (dim == j) {
        reduce_flag = true;
        break;
      }
    }
    if (!reduce_flag) {
      output_shape->AppendDim(x_shape->GetDim(j));
    }
  }
  GELOGD("ReduceDimsWithoutKeepDims is SUCCESS");
  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDims(const gert::Shape* x_shape, const gert::Tensor* axes_tensor, int32_t axes_size,
                           const bool keep_dims, gert::Shape* output_shape) {
  const T* axes_dims = axes_tensor->GetData<T>();
  if (keep_dims) {
    return ReduceDimsWithKeepDims<T>(x_shape, axes_dims, axes_size, output_shape);
  }
  return ReduceDimsWithoutKeepDims<T>(x_shape, axes_dims, axes_size, output_shape);
}

ge::graphStatus InferShape4ReduceCommon(InferShapeContext* context) {
  auto in_shape = context->GetInputShape(0);
  GE_ASSERT_NOTNULL(in_shape);
  auto axes_tensor = context->GetInputTensor(1);
  GE_ASSERT_NOTNULL(axes_tensor);
  auto out_shape = context->GetOutputShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);

  const bool* keep_dims = attrs->GetAttrPointer<bool>(0);
  GE_ASSERT_NOTNULL(keep_dims);

  auto axes_size = static_cast<int32_t>(axes_tensor->GetShapeSize());
//  GE_ASSERT(axes_size > 0, "axes size = %d must big than 0!", axes_size);

  auto dtype = axes_tensor->GetDataType();
//  auto dtype = ge::DT_INT32;
  GE_ASSERT(dtype == ge::DT_INT32 || dtype == ge::DT_INT64,
           "axes datatype ", ge::TypeUtils::DataTypeToSerialString(dtype), " must in (int32, int64)");
  if (dtype == ge::DT_INT32) {
    return ReduceDims<int32_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
  }
  return ReduceDims<int64_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
}

ge::graphStatus TilingForReduceSum(TilingContext *context) {
  (void) context;
  return ge::GRAPH_SUCCESS;
}
struct StubReduceSumTilingData {
  uint64_t tiling_data[8];
};
ge::graphStatus TilingParseForReduceSum(KernelContext *context) {
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(ReduceSum)
    .InferShape(InferShape4ReduceCommon)
    .InputsDataDependency({1})
    .Tiling(TilingForReduceSum)
    .TilingParse<StubReduceSumTilingData>(TilingParseForReduceSum);;
}  // namespace ops
