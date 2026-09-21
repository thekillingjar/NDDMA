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
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

static const int64_t kZero = 0;
static const int64_t kOne = 1;
static const int64_t kTwo = 2;
static const int64_t kThree = 3;
static const int64_t kFour = 4;

namespace ge {
namespace {
graphStatus InferShape4LayerNormXBackpropCommon(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  auto out1_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(out1_shape);

  *out_shape = *in_shape;
  *out1_shape = *in_shape;
  return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4SoftmaxCrossEntropyWithLogits(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(kZero);
  auto in1_shape = context->GetInputSymbolShape(kOne);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  GE_UNSUPPORTED_IF_NULL(in1_shape);
  auto out_shape = context->GetOutputSymbolShape(kZero);
  auto out1_shape = context->GetOutputSymbolShape(kOne);
  GE_ASSERT_NOTNULL(out_shape);
  GE_ASSERT_NOTNULL(out1_shape);

  auto check_dim = (in_shape->GetDimNum() == kTwo && in1_shape->GetDimNum() == kTwo) ||
      (in_shape->GetDimNum() == kFour && in1_shape->GetDimNum() == kFour);
  GE_ASSERT_TRUE(check_dim, "[Node:%s(%s)] input invalid, input1 dim:%zu, input2 dim:%zu. Inputdims must be two or four",
    context->GetNodeName(), context->GetNodeType(), in_shape->GetDimNum(), in1_shape->GetDimNum());

  auto feature_0 = in_shape->GetDim(kZero);
  auto labels_0 = in1_shape->GetDim(kZero);
  auto feature_1 = in_shape->GetDim(kOne);
  auto labels_1 = in1_shape->GetDim(kOne);
  auto dim_0 = in1_shape->GetDim(kZero);
  if (EXPECT_SYMBOL_GE(feature_0, labels_0)) {
    dim_0 = in_shape->GetDim(kZero);
  }
  auto dim_1 = in1_shape->GetDim(kOne);
  if (EXPECT_SYMBOL_GE(feature_1, labels_1)) {
    dim_1 = in_shape->GetDim(kOne);
  }
  out_shape->AppendDim(dim_0);
  out1_shape->AppendDim(dim_0);
  out1_shape->AppendDim(dim_1);

  if (in_shape->GetDimNum() == kFour && in1_shape->GetDimNum() == kFour) {
    auto feature_2 = in_shape->GetDim(kTwo);
    auto labels_2 = in1_shape->GetDim(kTwo);
    auto feature_3 = in_shape->GetDim(kThree);
    auto labels_3 = in1_shape->GetDim(kThree);
    out_shape->AppendDim(dim_1);

    auto dim_2 = in1_shape->GetDim(kTwo);
    if (EXPECT_SYMBOL_GE(feature_2, labels_2)) {
      dim_2 = in_shape->GetDim(kTwo);
    }
    auto dim_3 = in1_shape->GetDim(kThree);
    if (EXPECT_SYMBOL_GE(feature_3, labels_3)) {
      dim_3 = in_shape->GetDim(kThree);
    }
    out_shape->AppendDim(dim_2);
    out1_shape->AppendDim(dim_2);
    out_shape->AppendDim(dim_3);
    out1_shape->AppendDim(dim_3);
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SoftmaxCrossEntropyWithLogits).InferSymbolShape(InferShape4SoftmaxCrossEntropyWithLogits);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LayerNormXBackpropV3).InferSymbolShape(InferShape4LayerNormXBackpropCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LayerNormXBackpropV2).InferSymbolShape(InferShape4LayerNormXBackpropCommon);
}  // namespace
}  // namespace ge