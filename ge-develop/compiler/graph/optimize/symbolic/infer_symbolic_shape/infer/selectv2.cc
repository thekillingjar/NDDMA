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
/**
 * SelectV2算子 根据condition从两个input中选择元素，输出的shape和输入的shape以及condition广播后的shape一致
 *
 * condition：一个布尔类型的张量，表示选择的条件。
 * t：一个张量，当condition为True时，从这个张量中选择元素。
 * e：一个张量，当condition为False时，从这个张量中选择元素。
 */
graphStatus InferShape4SelectV2(gert::InferSymbolShapeContext *context) {
  const auto condition_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(condition_shape);
  const auto in_shape1 = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(in_shape1);
  const auto in_shape2 = context->GetInputSymbolShape(2);
  GE_UNSUPPORTED_IF_NULL(in_shape2);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  GE_ASSERT_SUCCESS(SymbolicInferUtil::Broadcast({in_shape1->GetDims(),
     in_shape2->GetDims(), condition_shape->GetDims()}, out_shape->MutableDims()));
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SelectV2).InferSymbolShape(InferShape4SelectV2);
}  // namespace
}  // namespace ge
