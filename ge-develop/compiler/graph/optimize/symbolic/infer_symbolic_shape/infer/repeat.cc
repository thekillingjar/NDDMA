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
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
graphStatus InferShape4Repeat(gert::InferSymbolShapeContext* context) {
  auto input_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_shape);
  if (input_shape->IsScalar()) {
    GELOGW("Symbol Infer unsupported, reason get: input_shape is scalar, node %s[%s]", 
      context->GetNodeName(),
      context->GetNodeType()
      );
    return ge::UNSUPPORTED;
  }
  auto repeat_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(repeat_tensor);
  auto repeat_num_values = repeat_tensor->GetSymbolicValue();
  if (repeat_num_values == nullptr) {
    GELOGW("Symbol Infer unsupported, reason get: symbolic_value_is_nullptr, node %s[%s]", 
      context->GetNodeName(),
      context->GetNodeType()
      );
    return ge::UNSUPPORTED;
  }
  const auto repeat_size = repeat_num_values->size();
  GE_ASSERT(repeat_size > 0U, "Invalid repeat_num, must be non-empty!");
  ge::Expression total_repeat(Symbol(0));
  for (const auto& val : *repeat_num_values) {
    total_repeat = total_repeat + val;
  }
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  *out_shape = *input_shape;
  out_shape->MutableDim(0) = total_repeat;
  return ge::SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Repeat).InferSymbolShape(InferShape4Repeat);
}
}  // namespace ge