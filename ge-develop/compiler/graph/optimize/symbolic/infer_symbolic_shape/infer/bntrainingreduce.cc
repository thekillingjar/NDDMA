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
const size_t kFormatNHWCDimSize = 4U;
const size_t kFormatNHWCCDimIdx = 3U;
const size_t kFormatNCHWDimMinSize = 2U;
const size_t kFormatNCHWCDimIdx = 1U;
const size_t kFormatNCHWDimMaxSize = 4U;
const size_t kFormatNDHWCDimSize = 5U;
const size_t kFormatNDHWCCDimIdx = 4U;
const size_t kFormatNDC1HWC0DimSize = 6U;
const size_t kFormatNDC1HWC0C1DimIdx = 2U;
const size_t kFormatNDC1HWC0C0DimIdx = 5U;
const size_t kFormatNCDHWDimSize = 5U;
const size_t kFormatNCDHWCDimIdx = 1U;

graphStatus GetOutputShapeFromOriginFormat(size_t dim_num, ge::Format origin_format, const gert::SymbolShape *x_shape,
                                           gert::SymbolShape *sum_shape, gert::SymbolShape *squaresum_shape) {
  if (origin_format == FORMAT_NHWC) {
    GE_ASSERT(dim_num == kFormatNHWCDimSize, "Input x rank[%zu] can only support 4 when NHWC", dim_num);
    sum_shape->AppendDim(x_shape->GetDim(kFormatNHWCCDimIdx));  // index of C is 3
    squaresum_shape->MutableDims() = sum_shape->GetDims();
  } else if (origin_format == FORMAT_NCHW) {
    GE_ASSERT(dim_num >= kFormatNCHWDimMinSize && dim_num <= kFormatNCHWDimMaxSize,
              "Input x rank[%zu] can only support 2 3 4 when NCHW", dim_num);
    sum_shape->AppendDim(x_shape->GetDim(kFormatNCHWCDimIdx));  // index of C is 1
    squaresum_shape->MutableDims() = sum_shape->GetDims();
  } else if (origin_format == FORMAT_NDHWC) {
    GE_ASSERT(dim_num == kFormatNDHWCDimSize, "Input x rank[%zu] can only support 5 when NDHWC", dim_num);
    sum_shape->AppendDim(x_shape->GetDim(kFormatNDHWCCDimIdx));  // index of C is 4
    squaresum_shape->MutableDims() = sum_shape->GetDims();
  } else if (origin_format == FORMAT_NDC1HWC0) {
    GE_ASSERT(dim_num == kFormatNDC1HWC0DimSize, "Input x rank[%zu] can only support 6 when NDC1HWC0", dim_num);
    sum_shape->AppendDim(ge::kSymbolOne);
    sum_shape->AppendDim(ge::kSymbolOne);
    sum_shape->AppendDim(x_shape->GetDim(kFormatNDC1HWC0C1DimIdx));  // index of C1 is 2
    sum_shape->AppendDim(ge::kSymbolOne);
    sum_shape->AppendDim(ge::kSymbolOne);
    sum_shape->AppendDim(x_shape->GetDim(kFormatNDC1HWC0C0DimIdx));  // index of C0 is 5
    squaresum_shape->MutableDims() = sum_shape->GetDims();
  } else if (origin_format == FORMAT_NCDHW) {
    GE_ASSERT(dim_num == kFormatNCDHWDimSize, "Input x rank[%zu] can only support 5 when NCDHW", dim_num);
    sum_shape->AppendDim(x_shape->GetDim(kFormatNCDHWCDimIdx));  // index of C is 1
    squaresum_shape->MutableDims() = sum_shape->GetDims();
  } else {
    GELOGE(GRAPH_FAILED, "origin_format only support NCHW, NHWC, NDHWC, NCDHW, NDC1HWC0, current origin_format %d", origin_format);
    return GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4BNTrainingReduce(gert::InferSymbolShapeContext *context) {
  auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  auto sum_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(sum_shape);
  auto squaresum_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(squaresum_shape);
  sum_shape->Clear();
  squaresum_shape->Clear();
  auto input_x_desc = context->GetInputDesc(0);
  GE_ASSERT_NOTNULL(input_x_desc);
  auto origin_format = input_x_desc->GetOriginFormat();
  size_t dim_num = x_shape->GetDimNum();

  return GetOutputShapeFromOriginFormat(dim_num, origin_format, x_shape, sum_shape, squaresum_shape);
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BNTrainingReduce).InferSymbolShape(InferShape4BNTrainingReduce);
}
} // namespace ge