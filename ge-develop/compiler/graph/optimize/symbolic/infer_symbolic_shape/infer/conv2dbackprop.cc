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
#include "common/checker.h"
#include "common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"

namespace ge {
namespace {
constexpr size_t kConv2dDimSizeLimit = 4U;
graphStatus InferShapeForConv2DBackpropD(gert::InferSymbolShapeContext *context) {
  const auto op_name = context->GetNodeName();
  auto y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(y_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto attrinput = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(attrinput);
  size_t input_size = attrinput->GetSize();
  if (input_size != kConv2dDimSizeLimit) {
    GELOGI("%s dim num is not %zu", op_name, kConv2dDimSizeLimit);
    return PARAM_INVALID;
  }
  std::vector<int64_t> input_list;
  input_list.resize(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    input_list[i] = attrinput->GetData()[i];
  }
  for (auto &cur_input : input_list) {
    y_shape->AppendDim(Symbol(cur_input));
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Conv2DBackpropInputD).InferSymbolShape(InferShapeForConv2DBackpropD);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Conv2DBackpropFilterD).InferSymbolShape(InferShapeForConv2DBackpropD);
}  // namespace
}  // namespace ge