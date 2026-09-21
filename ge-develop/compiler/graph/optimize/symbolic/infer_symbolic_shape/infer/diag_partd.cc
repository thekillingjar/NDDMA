/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
namespace ge {
namespace {

constexpr size_t INPUT_TO_OUTPUT_DIMS_TIMES = 2UL;
constexpr size_t EVEN = 2UL;

graphStatus InferShape4DiagPartD(gert::InferSymbolShapeContext* context) {
  const auto input_x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_x_shape);
  const auto assist_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(assist_shape);
  GE_ASSERT_EQ(input_x_shape->GetDimNum(), assist_shape->GetDimNum());
  for (size_t i = 0UL; i < input_x_shape->GetDims().size(); ++i) {
    ASSERT_SYMBOL_EQ(input_x_shape->GetDim(i), assist_shape->GetDim(i));
  }

  GE_ASSERT(input_x_shape->GetDimNum() % EVEN == 0, "The input dimension must be an even number,"
                                                 " input dim is %zu", input_x_shape->GetDimNum());
  const auto output_shape_len = input_x_shape->GetDimNum() / INPUT_TO_OUTPUT_DIMS_TIMES;

  auto output_y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(output_y_shape);

  for (size_t i = 0UL; i < output_shape_len; i++) {
    output_y_shape->AppendDim(input_x_shape->GetDim(i));
  }
  return GRAPH_SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(DiagPartD).InferSymbolShape(InferShape4DiagPartD);
}  // namespace
}  // namespace ge
