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
constexpr size_t INPUT_NUM_SEGMENTS_IDX = 2UL;

graphStatus UnsortedSegmentInferShapeImpl(const int64_t &first_dim, const gert::SymbolShape *x_shape,
  const gert::SymbolShape *segment_ids_shape, gert::SymbolShape *output_shape) {
  const auto x_rank = x_shape->GetDimNum();
  const auto segment_ids_rank = segment_ids_shape->GetDimNum();
  const auto output_rank = x_rank - segment_ids_rank + 1UL;
  output_shape->AppendDim(Symbol(first_dim));
  for (size_t i = 1UL; i < output_rank; i++) {
    output_shape->AppendDim(x_shape->GetDim(i + segment_ids_rank - 1UL));
  }

  return GRAPH_SUCCESS;
}

graphStatus InferShape4UnsortedSegment(gert::InferSymbolShapeContext* context) {
  const auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  const auto segment_ids_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(segment_ids_shape);
  const auto num_segments_shape = context->GetInputSymbolShape(INPUT_NUM_SEGMENTS_IDX);
  GE_UNSUPPORTED_IF_NULL(num_segments_shape);
  const auto num_segments_tensor = context->GetInputSymbolTensor(INPUT_NUM_SEGMENTS_IDX);
  GE_UNSUPPORTED_IF_NULL(num_segments_tensor);

  auto output_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(output_shape);
  GE_ASSERT(num_segments_shape->GetDims().size() == 1UL, "The size of num_segments must be 1, it is %zu!",
    num_segments_shape->GetDims().size());
  const auto num_segments_value = num_segments_tensor->GetSymbolicValue();
  GE_ASSERT_NOTNULL(num_segments_value);
  GE_ASSERT_EQ(num_segments_value->size(), 1UL);
  const auto dim_desc = context->GetInputDesc(INPUT_NUM_SEGMENTS_IDX);
  GE_ASSERT_NOTNULL(dim_desc);
  const auto dt = dim_desc->GetDataType();
  int64_t num_segments = 0L;
  GE_ASSERT_GRAPH_SUCCESS(SymbolicInferUtil::GetConstInt(num_segments_tensor, dt, num_segments));
  return UnsortedSegmentInferShapeImpl(num_segments, x_shape, segment_ids_shape, output_shape);
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(UnsortedSegmentMax).InferSymbolShape(InferShape4UnsortedSegment);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(UnsortedSegmentMin).InferSymbolShape(InferShape4UnsortedSegment);
}  // namespace
}  // namespace ge
