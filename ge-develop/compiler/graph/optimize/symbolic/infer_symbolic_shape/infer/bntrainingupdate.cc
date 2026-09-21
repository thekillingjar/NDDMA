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
graphStatus InferShape4BNTrainingUpdate(gert::InferSymbolShapeContext *context) {
  auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  auto scale_shape = context->GetInputSymbolShape(3);
  GE_UNSUPPORTED_IF_NULL(scale_shape);

  constexpr size_t output_num = 5U;
  for (size_t i = 0U; i < output_num; ++i) {
    auto output_shape = context->GetOutputSymbolShape(i);
    GE_ASSERT(output_shape != nullptr, "BNTrainingUpdate output shape is null, idx %zu", i);
    if (i == 0U) {
      *output_shape = *x_shape;
    } else {
      *output_shape = *scale_shape;
    }
  }

  return ge::GRAPH_SUCCESS;
}


IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BNTrainingUpdate).InferSymbolShape(InferShape4BNTrainingUpdate);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BNTrainingUpdateV3).InferSymbolShape(InferShape4BNTrainingUpdate);
}
} // namespace ge