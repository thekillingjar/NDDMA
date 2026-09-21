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
#include "graph/utils/type_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace ge {
namespace {
const Symbol kSymbolMinusOne(-1);

graphStatus InferShape4BroadcastTo(gert::InferSymbolShapeContext *context) {
  auto shape_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(shape_tensor);
  auto shape_value = shape_tensor->GetSymbolicValue();
  if (shape_value == nullptr) {
    GELOGW("Symbol Infer unsupported, reason get symbolic value is nullptr, node %s[%s]", context->GetNodeName(), context->GetNodeType());
    return ge::UNSUPPORTED;
  }
  auto shape_size = shape_value->size();
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto in_dims_num = in_shape->GetDimNum();
  GE_ASSERT_TRUE(shape_size >= in_dims_num, "tensor shape size %zu should greater than or equal input shape dim"
                 " num %zu.", shape_size, in_dims_num);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  out_shape->MutableDims() = *shape_value;
  auto diff = static_cast<int32_t>(shape_size - in_dims_num);
  for (int32_t i = (shape_size - 1); i >= 0; i--) {
    if (SymbolicUtils::StaticCheckEq(shape_value->at(i), kSymbolMinusOne) == TriBool::kTrue) {
      if (i >= diff) {
        out_shape->MutableDim(i) = in_shape->GetDim(i - diff);
      } else {
        out_shape->MutableDim(i) = kSymbolOne;
      }
    }
    if (i < diff) {
      continue;
    }
    GE_ASSERT_TRUE(EXPECT_SYMBOL_OR(sym::Eq(out_shape->GetDim(i), in_shape->GetDim(i - diff)),
                                    sym::Eq(in_shape->GetDim(i - diff), kSymbolOne)));
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BroadcastTo).InferSymbolShape(InferShape4BroadcastTo);
}  // namespace
}  // namespace ge