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
 * Pack算子 算子的符号化shape推导
 * 【算子功能】将一系列张量沿着一个制定的新维度堆叠起来生成一个新的张量
 * 【算子约束】
 *      1. 每个输入的轴数(或维度dim_num)相同
 *      2. 输入数必须等于设置在Attr中的N值
 *      3. 沿着制定维度拼接的轴数必须合法（在已有轴数范围内）
 *      举例：
 * 输入1=(3,8,10)，输入2=(3,8,10)，拼接轴=1，则上述输入的所有轴需要一致，且拼接轴范围在【-3，2】。输出=(3,2，8,10)
 * 【推导逻辑】
 *      1. 按照算子约束校验各个输入shape
 *      2. 以第一个输入的shape为基础，校验其他输入合法
 *      3. 设置拼接轴所在维度的值为输入个数（代表以该轴合并输入的所有张量）
 */
graphStatus InferShape4Pack(gert::InferSymbolShapeContext *context) {
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto *axis_ptr = attrs->GetAttrPointer<int64_t>(0);  // 拼接维度
  GE_ASSERT_NOTNULL(axis_ptr);
  auto axis = *axis_ptr;
  const auto *N = attrs->GetAttrPointer<int64_t>(1);  // 几个输入拼接
  GE_ASSERT_NOTNULL(N);
  const auto num = context->GetComputeNodeInputNum();
  GE_ASSERT(*N == static_cast<long int>(num) || *N == 1, "The N should be equal to input_size or default value 1.");
  auto fst_in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(fst_in_shape);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  *out_shape = *fst_in_shape;
  for (size_t relative_index = 1; relative_index < num; ++relative_index) {
    auto input_i_shape = context->GetInputSymbolShape(relative_index);
    GE_UNSUPPORTED_IF_NULL(input_i_shape);
    // input shape i is not equal input shape 0
    // 如果输入和输出(第一个输入)的维度不同，直接报错
    GE_ASSERT_TRUE(input_i_shape->GetDimNum() == fst_in_shape->GetDimNum(),
        "input_%d_dim_num(%d) != first_input_dim_num(%d)",
        relative_index, input_i_shape->GetDimNum(), fst_in_shape->GetDimNum());
      // 若输入和输出的维度相同时，还需要校验轴的维数需要一致，否则没法拼接。
    for (size_t i = 0UL; i < input_i_shape->GetDimNum(); i++) {
      ASSERT_SYMBOL_EQ(input_i_shape->GetDim(i), fst_in_shape->GetDim(i));
    }
  }
  if (out_shape->IsScalar()) {
    // scalar to shape[1]
    out_shape->AppendDim(Symbol(num));
    return GRAPH_SUCCESS;
  }
  size_t output_dim_size = out_shape->GetDimNum() + 1;
  if (!SymbolicInferUtil::IsDimValid(output_dim_size, axis)) {
    GELOGE(PARAM_INVALID, "axis=%d  but output_dim_size is %d", axis, output_dim_size);
    return PARAM_INVALID;
  }
  if (axis < 0) {
    axis += output_dim_size;
  }
  auto dims = &out_shape->MutableDims();
  dims->insert(dims->begin() + axis, Symbol(num));
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Pack).InferSymbolShape(InferShape4Pack);
}  // namespace
}  // namespace ge