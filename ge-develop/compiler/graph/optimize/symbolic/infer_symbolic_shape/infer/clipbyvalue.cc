
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
const size_t kClipByValueInputSize = 3;

graphStatus InferShape4BroadcastInputThree(gert::InferSymbolShapeContext *context) {
  std::vector<std::vector<Expression>> shapes;
  for (size_t i = 0; i < kClipByValueInputSize; i++) {
    auto in_shape = context->GetInputSymbolShape(i);
    GE_UNSUPPORTED_IF_NULL(in_shape);
    shapes.emplace_back(in_shape->GetDims());
  }
  std::vector<bool> need_broadcasts;
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  GE_ASSERT_SUCCESS(SymbolicInferUtil::Broadcast(shapes, out_shape->MutableDims()));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ClipByValue).InferSymbolShape(InferShape4BroadcastInputThree);
} // namespace
} // namespace ge