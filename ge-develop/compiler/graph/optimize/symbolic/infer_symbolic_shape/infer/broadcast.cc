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
graphStatus InferShape4BroadcastCommon(gert::InferSymbolShapeContext *context) {
  auto in_shape1 = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape1);
  auto in_shape2 = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(in_shape2);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  GE_ASSERT_SUCCESS(SymbolicInferUtil::Broadcast({in_shape1->GetDims(),
      in_shape2->GetDims()}, out_shape->MutableDims()));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Add).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Pow).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(AddV2).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Mul).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Less).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Sub).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(RealDiv).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Equal).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(NotEqual).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Greater).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Maximum).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Minimum).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LogicalAnd).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LogicalOr).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Div).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LessEqual).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SquaredDifference).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(GreaterEqual).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(DivNoNan).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LeakyReluGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReluGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ConfusionSoftmaxGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BitwiseAnd).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(FloorDiv).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(FloorMod).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(EluGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SoftmaxGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(TanhGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Axpy).InferSymbolShape(InferShape4BroadcastCommon);
}  // namespace
}  // namespace ge
