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
 * ConcatV2D 和 ConcatD 算子的符号化shape推导
 * 【算子功能】将给定的N个输入，按照给定的拼接轴concat_dim，拼接成一个输出
 * 【算子约束】
 *      1. 每个输入的轴数(或维度dim_num)相同
 *      2. 每个输入除了拼接轴外，其他的轴维数(或dim)相同
 *      举例： 输入1=(3,8,10)，输入2=(3,10,10)，输入3=(3,1,10)，拼接轴=1，则上述输入的轴0和轴2需要一致。输出=(3,19,10)
 * 【推导逻辑】
 *      1. 按照算子约束校验各个输入shape
 *      2. 以第一个输入的shape为基础，将其他输入的拼接轴维数dim往第一个输入的拼接轴上累加
 *      3. 将累加结果作为输出
 */
graphStatus InferShape4ConcatV2Common(gert::InferSymbolShapeContext *context, int64_t concat_dim, bool is_d,
                                      size_t start_shape_idx = 0U) {
  auto fst_in_shape = context->GetInputSymbolShape(start_shape_idx);  // 第一个输入的shape
  GE_UNSUPPORTED_IF_NULL(fst_in_shape);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  auto concat_axis = concat_dim;
  auto concat_num = context->GetComputeNodeInputNum();
  if (!is_d) {
    concat_num -= 1;
  }
  // 把第1个输入的shape赋值给输出shape
  *out_shape = *fst_in_shape;

  // 如果只有一个输入，那么直接返回即可，输出shape 就等于 输入shape
  if (concat_num == 1) {
    return ge::GRAPH_SUCCESS;
  }

  // 多输入场景下，再判断输入维度是否为标量。因为如果是标量，其shape为空，但拼接后的shape有1个dim，因此需要为其增加一个dim。
  if (out_shape->IsScalar()) {
    out_shape->AppendDim(Symbol(1));
  }
  auto out_dim_num = static_cast<int64_t>(out_shape->GetDimNum());
  GE_ASSERT(concat_axis >= -out_dim_num && concat_axis < out_dim_num, "concat_dim=%d must be in [%d, %d)", concat_axis,
            -out_dim_num, out_dim_num);
  if (concat_axis < 0) {
    concat_axis += (out_dim_num);  // 如果拼接轴是负数，表示是倒数第N个轴，将其转变为正数的轴。
  }

  // 把剩下输入的拼接轴dim累加到输出shape中对应的dim上
  auto total_concat_dim_size = out_shape->GetDim(concat_axis);
  for (int64_t i = 1; i < static_cast<int64_t>(concat_num); ++i) {
    auto input_i_shape = context->GetInputSymbolShape(i + start_shape_idx);
    GE_UNSUPPORTED_IF_NULL(input_i_shape);
    // 输入和输出(第一个输入)的维度相同 且 都是标量，直接+1即可
    if (input_i_shape->IsScalar() && out_dim_num == 1) {
      total_concat_dim_size = total_concat_dim_size + Symbol(1);
      continue;
    }
    // 如果输入和输出(第一个输入)的维度不同，直接报错
    GE_ASSERT(input_i_shape->GetDimNum() == static_cast<size_t>(out_dim_num),
              "input_%d_dim_num(%d) != first_input_dim_num(%d)", i, input_i_shape->GetDimNum(), out_dim_num);
    // 若输入和输出的维度相同时，还需要校验除了拼接轴之外，其他轴的维数需要一致，否则没法拼接。
    for (int64_t check_dim = 0; check_dim < out_dim_num; ++check_dim) {
      if (check_dim == concat_axis) {
        continue;
      }
      ASSERT_SYMBOL_EQ(input_i_shape->GetDim(check_dim), out_shape->GetDim(check_dim));
      if (!out_shape->GetDim(check_dim).IsConstExpr() && input_i_shape->GetDim(check_dim).IsConstExpr()) {
        out_shape->MutableDims()[check_dim] = input_i_shape->GetDim(check_dim);
      }
    }

    // 满足上述约束后，则累加拼接轴的维数
    total_concat_dim_size = total_concat_dim_size + input_i_shape->GetDim(concat_axis);
  }

  // 将累计的拼接轴维数设置回输出shape中，结束。
  out_shape->MutableDims()[concat_axis] = total_concat_dim_size;
  return GRAPH_SUCCESS;
}

graphStatus GetConcatDimValue(const gert::InferSymbolShapeContext *context, size_t dim_index, int64_t &concat_dim) {
  auto concat_dim_tensor = context->GetInputSymbolTensor(dim_index);
  GE_UNSUPPORTED_IF_NULL(concat_dim_tensor);
  auto concat_dim_value = concat_dim_tensor->GetSymbolicValue();
  if (concat_dim_value == nullptr) {
    GELOGW("Symbol Infer unsupported, reason get symbolic value is nullptr, node %s[%s]", context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  GE_ASSERT_EQ(static_cast<int32_t>(concat_dim_value->size()), 1);
  auto dim_desc = context->GetInputDesc(dim_index);
  GE_ASSERT_NOTNULL(dim_desc);
  auto dt = dim_desc->GetDataType();
  GE_ASSERT_GRAPH_SUCCESS(SymbolicInferUtil::GetConstInt(concat_dim_tensor, dt, concat_dim));
  return GRAPH_SUCCESS;
}

graphStatus InferShape4ConcatV2D(gert::InferSymbolShapeContext *context) {
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto *concat_dim = attrs->GetAttrPointer<int64_t>(0);  // 拼接维度
  GE_ASSERT_NOTNULL(concat_dim);

  return InferShape4ConcatV2Common(context, *concat_dim, true);
}

graphStatus InferShape4ConcatV2(gert::InferSymbolShapeContext *context) {
  const auto num = context->GetComputeNodeInputNum();
  auto dim_index = num - 1;

  int64_t concat_dim;
  auto status = GetConcatDimValue(context, dim_index, concat_dim);
  if (status == UNSUPPORTED) {
    return UNSUPPORTED;
  }
  GE_ASSERT_GRAPH_SUCCESS(status);
  return InferShape4ConcatV2Common(context, concat_dim, false);
}

graphStatus InferShape4Concat(gert::InferSymbolShapeContext *context) {
  int64_t concat_dim;
  auto status = GetConcatDimValue(context, 0U, concat_dim);
  if (status == UNSUPPORTED) {
    return UNSUPPORTED;
  }
  GE_ASSERT_GRAPH_SUCCESS(status);
  return InferShape4ConcatV2Common(context, concat_dim, false, 1U);
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Concat).InferSymbolShape(InferShape4Concat);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ConcatD).InferSymbolShape(InferShape4ConcatV2D);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ConcatV2).InferSymbolShape(InferShape4ConcatV2);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ConcatV2D).InferSymbolShape(InferShape4ConcatV2D);
}  // namespace
}  // namespace ge