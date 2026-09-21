/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tensor.h"
#include "common/util.h"

#include "exe_graph/runtime/infer_shape_context.h"

namespace gert {
template<typename T>
ge::graphStatus ReshapeInferShapeImpl(const T *reshape_dims, const Shape &x_shape, Shape &output_shape,
                                      int32_t reshape_size) {
  constexpr T UNKNOWN_DIM = -1;
  output_shape.SetDimNum(reshape_size);
  auto x_shape_size = x_shape.GetShapeSize();
  int64_t output_shapesize = 1;
  size_t unknown_dim_idx = std::numeric_limits<size_t>::max();
  for (int32_t i = 0; i < reshape_size; i++) {
    if (reshape_dims[i] != UNKNOWN_DIM) {
      output_shape.SetDim(i, reshape_dims[i]);
      output_shapesize *= reshape_dims[i];
    } else {

      output_shape.SetDim(i, 1);
      unknown_dim_idx = i;
    }
  }
  if (unknown_dim_idx == std::numeric_limits<size_t>::max() && output_shapesize == x_shape_size) {
    return ge::GRAPH_SUCCESS;
  } else if (unknown_dim_idx != std::numeric_limits<size_t>::max() && x_shape_size % output_shapesize == 0) {
    output_shape.SetDim(unknown_dim_idx, x_shape_size / output_shapesize);
    return ge::GRAPH_SUCCESS;
  }
  GELOGE(ge::FAILED, "unknown_dim_idx[%zu], output_shapesize[%lld], x_shape_size[%lld]",
    unknown_dim_idx, output_shapesize, x_shape_size);
  return ge::GRAPH_FAILED;
}
ge::graphStatus InferShapeForReshape(InferShapeContext *context) {
  auto x_shape = context->GetInputShape(0);
  auto shape_tensor = context->GetInputTensor(1);
  auto output_shape = context->GetOutputShape(0);
  if (x_shape == nullptr || shape_tensor == nullptr || output_shape == nullptr) {
    GELOGE(ge::FAILED, "x_shape or shape_tensor or output_shape is null.");
    return ge::GRAPH_FAILED;
  }

  auto reshape_size = static_cast<int32_t>(shape_tensor->GetShapeSize());
  if (reshape_size < 1) {
    GELOGE(ge::FAILED, "reshape_size[%d] < 1", reshape_size);
    return ge::GRAPH_FAILED;
  }
  if (shape_tensor->GetDataType() == ge::DT_INT32) {
    auto reshape_data = shape_tensor->GetData<int32_t>();
    if (reshape_data == nullptr) {
      GELOGE(ge::FAILED, "reshape_data is null, data type is DT_INT32");
      return ge::GRAPH_FAILED;
    }
    return ReshapeInferShapeImpl<int32_t>(reshape_data, *x_shape, *output_shape, reshape_size);
  } else {
    auto reshape_data = shape_tensor->GetData<int64_t>();
    if (reshape_data == nullptr) {
      GELOGE(ge::FAILED, "reshape_data is null, data type is DT_INT64");
      return ge::GRAPH_FAILED;
    }
    return ReshapeInferShapeImpl<int64_t>(reshape_data, *x_shape, *output_shape, reshape_size);
  }
}
IMPL_OP(Reshape).InferShape(InferShapeForReshape).InputsDataDependency({1});

ge::graphStatus InferShapeForShapeN(InferShapeContext *context) {
  auto extend_kernel_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto input_num = extend_kernel_context->GetComputeNodeInputNum();
  for (size_t i = 0U; i < input_num; ++i) {
    auto x_shape = context->GetDynamicInputShape(0U, i);
    auto y_shape = context->GetOutputShape(i);
    if ((x_shape == nullptr) || (y_shape == nullptr)) {
      return ge::GRAPH_FAILED;
    }
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, static_cast<int64_t>(x_shape->GetDimNum()));
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(Shape).InferShape(InferShapeForShapeN);

ge::graphStatus InferShapeForUnsqueeze(gert::InferShapeContext* context) {
  const auto x_shape = context->GetInputShape(0);
  auto y_shape = context->GetOutputShape(0);
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  GE_CHECK_NOTNULL(attrs);
  auto axes = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(0);

  if (x_shape == nullptr || axes == nullptr || y_shape == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "X_shape or axes or y_shape pointer must not be nullptr!");
    return ge::GRAPH_FAILED;
  }

  const auto dim_size = x_shape->GetDimNum();
  const auto out_dim_size = dim_size + axes->GetSize();
  y_shape->SetDimNum(out_dim_size);
  if (out_dim_size > y_shape->kMaxDimNum) {
    GELOGE(ge::GRAPH_FAILED, "DimNum of output shape is %zu, larger than kMaxDimNum which is %zu!", out_dim_size,
           (y_shape->kMaxDimNum));
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0UL; i < out_dim_size; ++i) {
    y_shape->SetDim(i, 0);
  }

  for (size_t i = 0UL; i < axes->GetSize(); ++i) {
    const int64_t axis = (axes->GetData())[i];
    const int64_t real_axis = (axis < 0) ? (axis + static_cast<int64_t>(out_dim_size)) : axis;
    if ((real_axis < 0) || (real_axis >= static_cast<int64_t>(out_dim_size))) {
      GELOGE(ge::GRAPH_FAILED, "Unsqueeze axis must be in range [-%zu, %zu)!", out_dim_size, out_dim_size);
      return ge::GRAPH_FAILED;
    }
    if (y_shape->GetDim(real_axis) == 1) {
      GELOGE(ge::GRAPH_FAILED, "Unsqueeze axis repeated!");
      return ge::GRAPH_FAILED;
    }
    y_shape->SetDim(real_axis, 1);
  }

  size_t idx = 0UL;
  for (size_t i = 0UL; i < out_dim_size; ++i) {
    if (y_shape->GetDim(i) != 1) {
      y_shape->SetDim(i, x_shape->GetDim(idx));
      ++idx;
    }
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Unsqueeze).InferShape(InferShapeForUnsqueeze);
}  // namespace gert
