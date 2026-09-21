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
#include "common/checker.h"
#include "common/types.h"
#include "graph/compute_graph.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"

namespace ge {
namespace {

constexpr size_t kMatmulDimNum = 2U;
const size_t kBatchMatMulMaxDimNum = 8;
const size_t kMatMulMIdx = 2;
const size_t kMatMulNIdx = 1;

void GetMatMulDim(size_t dim_num, bool adj, size_t &dim_m, size_t &dim_n) {
  if (!adj) {
    dim_m = dim_num - kMatMulMIdx;
    dim_n = dim_num - kMatMulNIdx;
  } else {
    dim_m = dim_num - kMatMulNIdx;
    dim_n = dim_num - kMatMulMIdx;
  }
}

Status CalBatchOutShape(gert::SymbolShape *out_shape, const gert::SymbolShape &shape_long,
                      const gert::SymbolShape &shape_short) {
  auto valid_offset = shape_long.GetDimNum() - shape_short.GetDimNum();

  for (size_t i = 0U; i < valid_offset; ++i) {
    out_shape->AppendDim(shape_long.GetDim(i));
  }

  const auto sym_1 = kSymbolOne;
  for (size_t i = valid_offset; i < shape_long.GetDimNum() - kMatmulDimNum; ++i) {
    const auto &dim_long = shape_long.GetDim(i);
    const auto &dim_short = shape_short.GetDim(i - valid_offset);
    // 添加guard 如果dim_long和dim_short需要相等或者为1
    if (EXPECT_SYMBOL_EQ(dim_long, dim_short)) {
      auto dim_out = dim_long.IsConstExpr() ? dim_long : dim_short;
      out_shape->AppendDim(dim_out);
    } else if (EXPECT_SYMBOL_EQ(dim_long, sym_1)) {
      out_shape->AppendDim(dim_short);
    } else if (EXPECT_SYMBOL_EQ(dim_short, sym_1)) {
      out_shape->AppendDim(dim_long);
    } else {
      GELOGE(ge::FAILED, "Symbol input0[%zu]:%s is not equal to input1[%zu]:%s which cannot broadcast.",
          i, dim_long.Serialize().get(), (i - valid_offset), dim_short.Serialize().get());
      return FAILED;
    }
  }
  return SUCCESS;
}
// todo tensorflow只有x1, x2, adj_x1, adj_x2四个入参，没有bias参数，因此暂不考虑
graphStatus InferShape4BatchMatMulV2(gert::InferSymbolShapeContext *context) {
  auto shape_x1 = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(shape_x1);
  auto dim_num_x1 = shape_x1->GetDimNum();
  GE_ASSERT_TRUE(dim_num_x1 >= 1 && dim_num_x1 <= kBatchMatMulMaxDimNum,
                 "X1 invalid, dim_num: %zu must in must in [1, 8]", dim_num_x1);

  auto shape_x2 = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(shape_x2);
  auto dim_num_x2 = shape_x2->GetDimNum();
  GE_ASSERT_TRUE(dim_num_x2 >= 1 && dim_num_x2 <= kBatchMatMulMaxDimNum,
                 "X2 invalid, dim_num: %zu must in must in [1, 8]", dim_num_x2);

  auto shape_out = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(shape_out);

  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto adj_x1 = attrs->GetAttrPointer<bool>(0);
  GE_ASSERT_NOTNULL(adj_x1);
  const auto adj_x2 = attrs->GetAttrPointer<bool>(1);
  GE_ASSERT_NOTNULL(adj_x2);

  auto shape_x1_new = *shape_x1;
  if (shape_x1_new.GetDimNum() == 1) {
    auto ori_dim = shape_x1_new.GetDim(0);
    shape_x1_new.Clear();
    shape_x1_new.AppendDim(kSymbolOne);
    shape_x1_new.AppendDim(ori_dim);
  }

  auto shape_x2_new = *shape_x2;
  if (shape_x2_new.GetDimNum() == 1) {
    shape_x2_new.AppendDim(kSymbolOne);
  }
  size_t idx_m;
  size_t idx_k_a;
  GetMatMulDim(shape_x1_new.GetDimNum(), *adj_x1, idx_m, idx_k_a);

  size_t idx_k_b;
  size_t idx_n;
  GetMatMulDim(shape_x2_new.GetDimNum(), *adj_x2, idx_k_b, idx_n);

  // 添加guard idx_k_a和idx_k_b应相等
  ASSERT_SYMBOL_EQ(shape_x1_new.GetDim(idx_k_a), shape_x2_new.GetDim(idx_k_b));
  if (shape_x1_new.GetDimNum() > shape_x2_new.GetDimNum()) {
    GE_ASSERT_SUCCESS(CalBatchOutShape(shape_out, shape_x1_new, shape_x2_new));
  } else {
    GE_ASSERT_SUCCESS(CalBatchOutShape(shape_out, shape_x2_new, shape_x1_new));
  }

  shape_out->AppendDim(shape_x1_new.GetDim(idx_m));
  shape_out->AppendDim(shape_x2_new.GetDim(idx_n));
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BatchMatMulV2).InferSymbolShape(InferShape4BatchMatMulV2);
}  // namespace
}  // namespace ge