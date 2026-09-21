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
graphStatus InferShapeSparseSoftmaxCrossEntropyWithLogits(gert::InferSymbolShapeContext* context) {
  const auto features = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(features);
  const auto labels = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(labels);
  auto loss = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(loss);
  auto lback_prop = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(lback_prop);
  *loss = *labels;
  *lback_prop = *features;
  return GRAPH_SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SparseSoftmaxCrossEntropyWithLogits).InferSymbolShape(InferShapeSparseSoftmaxCrossEntropyWithLogits);
}  // namespace
}  // namespace ge
