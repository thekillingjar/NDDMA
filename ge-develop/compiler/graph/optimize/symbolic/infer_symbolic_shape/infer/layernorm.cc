/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
graphStatus InferShape4LayerNorm(gert::InferSymbolShapeContext *context) {
  auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  auto y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(y_shape);
  auto mean_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(mean_shape);
  auto var_shape = context->GetOutputSymbolShape(2);
  GE_ASSERT_NOTNULL(var_shape);

  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto *begin_norm_axis_ptr = attrs->GetAttrPointer<int64_t>(0);
  GE_ASSERT_NOTNULL(begin_norm_axis_ptr);
  if (!SymbolicInferUtil::IsDimValid(x_shape->GetDimNum(), *begin_norm_axis_ptr)) {
    GELOGE(PARAM_INVALID, "axis=%d  but input_x is %d", *begin_norm_axis_ptr, x_shape->GetDimNum());
    return PARAM_INVALID;
  }
  int64_t begin_norm_axis_val = *begin_norm_axis_ptr < 0
                                  ? *begin_norm_axis_ptr + static_cast<int64_t>(x_shape->GetDimNum())
                                  : *begin_norm_axis_ptr;
  *y_shape = *x_shape;
  *mean_shape = *x_shape;
  for (size_t i = 0; i < x_shape->GetDimNum(); ++i) {
    if (static_cast<int64_t>(i) >= begin_norm_axis_val) {
      mean_shape->MutableDims()[i] = Symbol(1);
    } else {
      mean_shape->MutableDims()[i] = x_shape->GetDim(i);
    }
  }
  *var_shape = *mean_shape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LayerNorm).InferSymbolShape(InferShape4LayerNorm);
} // namespace
} // namespace ge