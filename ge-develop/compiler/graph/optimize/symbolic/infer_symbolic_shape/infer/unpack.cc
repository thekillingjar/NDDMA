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
 * UnPack算子 算子的符号化shape推导
 * 【算子功能】将一个张量按制定维度反解成多个张量的操作，属于pack的还原
 * 【算子约束】
 *         1. 输出数必须等于设置在Attr中的N值
 *         2. 沿着制定维度拼接的轴数必须合法（在已有轴数范围内）
 *         举例： 输入1=(2,8,10)，拆分轴=0，拆分轴范围在【-3，2】。输出：out_shape1=(8,10),out_shape2=(8,10)
 * 【推导逻辑】
 *      1. 校验输出shape数量是否等于属性num
 *      2. 删除拆分轴所在的维度并设置输出shape的除axis轴外的dim为输入
 */
graphStatus InferShape4Unpack(gert::InferSymbolShapeContext *context) {
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto *num = attrs->GetAttrPointer<int64_t>(0);  // 拆分成几个输出
  GE_ASSERT_NOTNULL(num);
  const auto *axis_ptr = attrs->GetAttrPointer<int64_t>(1);  // 拆分维度
  GE_ASSERT_NOTNULL(axis_ptr);
  if (!context->GetOutputSymbolShape(*num - 1) || context->GetOutputSymbolShape(*num)) {
    // output shapes size != num
    GELOGE(PARAM_INVALID, "invalid num or out_shape_size");
    return GRAPH_FAILED;
  }
  auto input_x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_x_shape);
  size_t input_x_dim_size = input_x_shape->GetDimNum();
  int64_t real_axis = (*axis_ptr >= 0) ? (*axis_ptr) : *axis_ptr + input_x_dim_size;
  if (real_axis < 0 || real_axis >= static_cast<int64_t>(input_x_dim_size)) {
    GELOGE(PARAM_INVALID, "invalid axis=%d  but input_x_shape is %d", *axis_ptr, input_x_shape);
    return PARAM_INVALID;
  }
  for (size_t i = 0; i < static_cast<size_t>(*num); ++i) {
    auto out_y_i_shape = context->GetOutputSymbolShape(i);
    GE_ASSERT_NOTNULL(out_y_i_shape);
    for (size_t j = 0; j < input_x_dim_size; ++j) {
      if (static_cast<int64_t>(j) < real_axis) {
        out_y_i_shape->AppendDim(input_x_shape->GetDim(j));
      } else if (static_cast<int64_t>(j) > real_axis) {
        out_y_i_shape->AppendDim(input_x_shape->GetDim(j));
      }
    }
  }
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Unpack).InferSymbolShape(InferShape4Unpack);
}  // namespace
}  // namespace ge