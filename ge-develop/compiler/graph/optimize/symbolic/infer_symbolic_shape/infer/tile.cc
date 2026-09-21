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
 * Tile算子符号化推导，输出的shape等于输入shape*重复次数
 *
 * 1. input shape维度数大于multiples数组大小, 先以当前维度 * 1； 例如 {2, 3, 4} * [3, 2] --> {2, 3, 4} * [1, 3, 2]
 * 2. input shape维度数小于multiples数组大小, 在shape前添加一个维度1； 例如 {2, 3} * [3, 2, 1] --> {1, 2, 3} * [3, 2, 1]
 */
graphStatus InferShape4Tile(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto multiples = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(multiples);
  if (multiples->GetSymbolicValue() == nullptr) {
    GELOGW("Symbol Infer unsupported, get symbolic value is nullptr, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const size_t in_shape_len = in_shape->GetDimNum();
  const size_t multiples_len = multiples->GetSymbolicValue()->size();
  const size_t max_len = in_shape_len > multiples_len ? in_shape_len : multiples_len;
  int64_t diff_len = static_cast<int64_t>(in_shape_len) - static_cast<int64_t>(multiples_len);
  out_shape->Clear();
  const auto multiples_value = *multiples->GetSymbolicValue();
  int32_t in_shape_index = 0;
  int32_t multiples_index = 0;
  for (size_t i = 0; i < max_len; ++i) {
    if (diff_len > 0) {
      // input shape维度数大于multiples数组大小, 先以当前维度 * 1
      out_shape->AppendDim(in_shape->GetDim(in_shape_index++) * ge::Symbol(1));
      diff_len--;
    } else if (diff_len < 0) {
      // input shape维度数小于multiples数组大小, 在shape前添加一个维度1
      out_shape->AppendDim(ge::Symbol(1) * multiples_value[multiples_index++]);
      diff_len++;
    } else {
      out_shape->AppendDim(in_shape->GetDim(in_shape_index++) * multiples_value[multiples_index++]);
    }
  }
  return ge::GRAPH_SUCCESS;
}
constexpr size_t kTileAttrIndexMultiples = 0U;
constexpr size_t kZero = 0U;
constexpr size_t kMaxDimNum = 8U;

graphStatus InferShape4TileD(gert::InferSymbolShapeContext *context) {
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto multiples = attrs->GetListInt(kTileAttrIndexMultiples);
  GE_ASSERT_NOTNULL(multiples);
  const int64_t *multiples_data = multiples->GetData();
  size_t multiples_len = multiples->GetSize();
  auto in_shape = context->GetInputSymbolShape(kZero);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto out_shape = context->GetOutputSymbolShape(kZero);
  GE_ASSERT_NOTNULL(out_shape);
  auto in_shape_len = in_shape->GetDimNum();
  GE_ASSERT_TRUE(multiples_len <= kMaxDimNum, "the tile multiples len %zu is more than MaxDimNum %zu", multiples_len,
                 kMaxDimNum);
  const size_t max_len = in_shape_len > multiples_len ? in_shape_len : multiples_len;
  int64_t diff_len = static_cast<int64_t>(in_shape_len) - static_cast<int64_t>(multiples_len);
  out_shape->Clear();
  int32_t in_shape_index = 0;
  int32_t multiples_index = 0;
  for (size_t i = 0U; i < max_len; ++i) {
    if (diff_len > 0) {
      // input shape维度数大于multiples数组大小, 输出就等于输入维度
      out_shape->AppendDim(in_shape->GetDim(in_shape_index++));
      diff_len--;
    } else if (diff_len < 0) {
      // input shape维度数小于multiples数组大小, 先padding1，再做multi
      out_shape->AppendDim(ge::Symbol(multiples_data[multiples_index++]));
      diff_len++;
    } else {
      out_shape->AppendDim((in_shape->GetDim(in_shape_index++) * ge::Symbol(multiples_data[multiples_index++])));
    }
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Tile).InferSymbolShape(InferShape4Tile);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(TileD).InferSymbolShape(InferShape4TileD);
}  // namespace
}  // namespace ge