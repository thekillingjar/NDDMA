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
 * MultisliceConcat算子符号化推导，每个output shape根据concat_size指示的个数，对slice_size列表进行累加得到
 *
 * 1. input shape目前只支持二维，输入为{batchsize，14768},
 * 2. 如现在out个数为2，concatSize = {2,4}，slice_size = {4,5,12,1,1,1}
 * 3. 则output(0)的shape为{batchsize，9}，其中9=slice_size(0)+slice_size(1)
 * 4. 则output(0)的shape为{batchsize，15}，其中15=slice_size(2)+slice_size(3)+slice_size(4)+slice_size(5)
 *
 */
constexpr size_t CONCAT_SIZE_LIST_IDX = 0U;
constexpr size_t SLICE_SIZE_IDX = 2U;
constexpr size_t CONCAT_NUM_IDX = 3U;
constexpr size_t DIM_NUM = 2U;
graphStatus InferShape4MultisliceConcat(gert::InferSymbolShapeContext *context) {
  GE_CHECK_NOTNULL(context);
  auto input_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_shape);
  auto dim_num = input_shape->GetDimNum();
  GE_ASSERT_EQ(dim_num, DIM_NUM);
  auto input_row = input_shape->GetDim(0);
  auto input_col = input_shape->GetDim(1);

  GE_ASSERT_NOTNULL(context->GetAttrs());
  auto concat_num_ptr = context->GetAttrs()->GetInt(CONCAT_NUM_IDX);
  GE_ASSERT_NOTNULL(concat_num_ptr);
  auto concat_num = *(concat_num_ptr);
  GE_ASSERT_TRUE(concat_num > 0L);
  auto concat_size = context->GetAttrs()->GetListInt(CONCAT_SIZE_LIST_IDX);
  GE_ASSERT_NOTNULL(concat_size);
  GE_ASSERT_EQ(concat_size->GetSize(), static_cast<size_t>(concat_num));
  auto slice_size = context->GetAttrs()->GetListInt(SLICE_SIZE_IDX);
  GE_ASSERT_NOTNULL(slice_size);
  int64_t addr_offset = 0L;

  for (int64_t i = 0L; i < concat_num; i++) {
    auto size = concat_size->GetData()[i];
    ge::Expression out_col_size{Symbol(0)};
    for (int64_t j = 0L; j < size; j++) {
      auto offset = addr_offset + j;
      GE_ASSERT_TRUE(static_cast<size_t>(offset) < slice_size->GetSize());
      auto slice_num = slice_size->GetData()[offset];
      GE_ASSERT_TRUE(slice_num > 0L);
      out_col_size = out_col_size + Symbol(slice_num);
    }
    addr_offset += size;
    auto out_shape = context->GetOutputSymbolShape(i);
    GE_ASSERT_NOTNULL(out_shape);
    out_shape->Clear();
    out_shape->AppendDim(input_row);
    out_shape->AppendDim(out_col_size);
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MultisliceConcat).InferSymbolShape(InferShape4MultisliceConcat);
}  // namespace
}  // namespace ge