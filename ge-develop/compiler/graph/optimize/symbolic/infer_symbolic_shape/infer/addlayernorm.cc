/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
graphStatus InferShape4AddLayerNorm(gert::InferSymbolShapeContext *context) {
  auto x1_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x1_shape);
  auto gamma_shape = context->GetInputSymbolShape(2);
  GE_UNSUPPORTED_IF_NULL(gamma_shape);
  auto beta_shape = context->GetInputSymbolShape(3);
  GE_UNSUPPORTED_IF_NULL(beta_shape);
  GE_ASSERT(gamma_shape->GetDimNum() == beta_shape->GetDimNum());
  for (size_t i = 0U; i < gamma_shape->GetDimNum(); i++) {
    ASSERT_SYMBOL_EQ(gamma_shape->GetDim(i), beta_shape->GetDim(i));
  }

  auto y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(y_shape);
  auto mean_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(mean_shape);
  auto rstd_shape = context->GetOutputSymbolShape(2);
  GE_ASSERT_NOTNULL(rstd_shape);
  auto x_shape = context->GetOutputSymbolShape(3);
  GE_ASSERT_NOTNULL(x_shape);

  *y_shape = *x1_shape;
  *x_shape = *x1_shape;
  auto shape(*x1_shape);
  if (shape.GetDimNum() > 0U) {
    shape.MutableDims()[shape.GetDimNum() - 1U] = ge::kSymbolOne;
  }
  *mean_shape = shape;
  *rstd_shape = shape;
  return ge::GRAPH_SUCCESS;
}


IMPL_OP_INFER_SYMBOL_SHAPE_INNER(AddLayerNorm).InferSymbolShape(InferShape4AddLayerNorm);
}
} // namespace ge