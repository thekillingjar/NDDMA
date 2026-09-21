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

constexpr size_t kMatmulDimNum = 2U;
constexpr size_t kZero = 0U;
constexpr size_t kOne = 1U;

graphStatus InferShape4MatMul(gert::InferSymbolShapeContext *context) {
  auto in_shape1 = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape1);
  auto in_shape2 = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(in_shape2);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const bool *trans_a = attrs->GetAttrPointer<bool>(0);
  const bool *trans_b = attrs->GetAttrPointer<bool>(1);
  GE_ASSERT_NOTNULL(trans_a);
  GE_ASSERT_NOTNULL(trans_b);
  // todo: matmul的infershape中出现了补维现象，例如shape[128]会补成[1, 128]，暂时我们不支持
  GE_ASSERT_EQ(in_shape1->GetDimNum(), kMatmulDimNum);
  GE_ASSERT_EQ(in_shape2->GetDimNum(), kMatmulDimNum);
  size_t idx_m = kZero;
  size_t idx_k_a = kOne;
  size_t idx_k_b = kZero;
  size_t idx_n = kOne;
  if (*trans_a) {
    idx_m = kOne;
    idx_k_a = kZero;
  }

  if (*trans_b) {
    idx_k_b = kOne;
    idx_n = kZero;
  }

  auto k_a = in_shape1->GetDim(idx_k_a);
  auto k_b = in_shape2->GetDim(idx_k_b);
  ASSERT_SYMBOL_EQ(k_a, k_b);

  // 设置输出shape
  out_shape->MutableDims().clear();
  out_shape->MutableDims().emplace_back(in_shape1->GetDim(idx_m));
  out_shape->MutableDims().emplace_back(in_shape2->GetDim(idx_n));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MatMul).InferSymbolShape(InferShape4MatMul);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MatMulV2).InferSymbolShape(InferShape4MatMul);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MatMulV3).InferSymbolShape(InferShape4MatMul);
}  // namespace
}  // namespace ge