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
/**
 * TransposeD算子符号化推导，交换一个张量的维度
 * input_shape0：输入张量的shape
 * perm：转置后张量的维度在原张量上的映射，整型列表，列表长度跟输入张量的维度相同，其中的元素要求>=0，且小于输入张量的最大维度
 * perm_num：perm 列表长度
 * 例如
 * 输入张量为{1, 2, 3}
 * perm为[2, 0, 1]， perm_num=3
 * 转置后的张量为{3, 1, 2}
 */
graphStatus InferShape4TransposeD(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto perm = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(perm);
  int64_t perm_size = perm->GetSize();
  size_t input_dim_size = in_shape->GetDimNum();
  GE_ASSERT_EQ(static_cast<int64_t>(input_dim_size), perm_size);
  std::vector<int64_t> perm_list;
  perm_list.resize(perm_size);
  for (int64_t i = 0; i < perm_size; ++i) {
    perm_list[i] = perm->GetData()[i];
  }
  for (auto &p : perm_list) {
    int64_t perm_v = p;
    perm_v = perm_v >= 0 ? perm_v : perm_v + input_dim_size;
    if (perm_v < 0 || perm_v >= static_cast<int64_t>(input_dim_size)) {
      return ge::GRAPH_FAILED;
    }
    out_shape->AppendDim(in_shape->GetDim(perm_v));
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(TransposeD).InferSymbolShape(InferShape4TransposeD);
}  // namespace
}  // namespace ge