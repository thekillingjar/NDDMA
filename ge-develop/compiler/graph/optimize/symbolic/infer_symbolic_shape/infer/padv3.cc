/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
constexpr size_t PAIR = 2UL;

graphStatus PadV3InferShape(const gert::InferSymbolShapeContext* context, const gert::SymbolShape* x_shape,
                            const gert::SymbolTensor* paddings_tensor, gert::SymbolShape* y_shape) {
  const auto paddings_value = paddings_tensor->GetSymbolicValue();
  GE_ASSERT_NOTNULL(paddings_value);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto paddings_contiguous = attrs->GetAttrPointer<bool>(1);
  GE_ASSERT_NOTNULL(paddings_contiguous);
  // input shape check
  const size_t input_dim_size = x_shape->GetDimNum();
  GE_ASSERT(input_dim_size != 0UL, "input shape cannot empty");
  const auto paddings_size = paddings_tensor->GetSymbolicValue()->size();
  GE_ASSERT(paddings_size > 0UL, "Invalid paddings, must be non-empty!");

  const auto &paddings_num = paddings_tensor->GetOriginSymbolShape().GetSymbolShapeSize();
  ASSERT_SYMBOL_EQ(paddings_num, Symbol(input_dim_size * PAIR));
  // infer by paddings_contiguous
  size_t index_cof = 1UL;
  size_t index_offset = input_dim_size;
  if (*paddings_contiguous) {
    index_cof = PAIR;
    index_offset = 1UL;
  }
  for (size_t i = 0UL; i < input_dim_size; ++i) {
    auto pad_front = paddings_value->at(index_cof * i);
    auto pad_end = paddings_value->at(index_cof * i + index_offset);
    y_shape->AppendDim(x_shape->GetDim(i) + pad_front + pad_end);
  }
  return GRAPH_SUCCESS;
}

graphStatus InferShape4PadV3(gert::InferSymbolShapeContext* context) {
  const auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  auto y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(y_shape);
  const auto paddings_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(paddings_tensor);
  const auto paddings_desc = context->GetInputDesc(1);
  GE_ASSERT_NOTNULL(paddings_desc);
  const auto paddings_dtype = paddings_desc->GetDataType();
  GE_ASSERT(paddings_dtype == DT_INT32 || paddings_dtype == DT_INT64, "paddings data type must is int32 or int64, it is %d", paddings_dtype);
  return PadV3InferShape(context, x_shape, paddings_tensor, y_shape);;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(PadV3).InferSymbolShape(InferShape4PadV3);
}  // namespace
}  // namespace ge
