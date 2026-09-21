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

/**
 * SquareSumV1算子 infer shape函数实现
 *
 */
graphStatus InferShape4SquareSumV1(gert::InferSymbolShapeContext *context) {
  auto input_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_shape);

  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto axes = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(axes);

  std::vector<int64_t> reduce_axes;
  reduce_axes.resize(axes->GetSize());
  for (size_t i = 0U; i < axes->GetSize(); ++i) {
    reduce_axes.at(i) = axes->GetData()[i];
  }

  auto axes_size = axes->GetSize();
  auto keep_dims = attrs->GetBool(1);
  GE_ASSERT_NOTNULL(keep_dims);

  if (*keep_dims) {
    return SymbolicInferUtil::ReduceDimsWithKeepDims<int64_t>(input_shape, reduce_axes, axes_size, out_shape);
  }
  return SymbolicInferUtil::ReduceDimsWithoutKeepDims<int64_t>(input_shape, reduce_axes, axes_size, out_shape);
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SquareSumV1).InferSymbolShape(InferShape4SquareSumV1);
}  // namespace
}  // namespace ge