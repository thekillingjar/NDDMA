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
 * Select算子 根据condition从两个input中选择元素，输出的shape和两个输入的shape一致
 *
 * condition：一个布尔类型的张量，表示选择的条件。
 * t：一个张量，当condition为True时，从这个张量中选择元素。
 * e：一个张量，当condition为False时，从这个张量中选择元素。
 */
graphStatus InferShape4Select(gert::InferSymbolShapeContext *context) {
  const auto condition_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(condition_shape);
  const auto in_shape1 = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(in_shape1);
  const auto in_shape2 = context->GetInputSymbolShape(2);
  GE_UNSUPPORTED_IF_NULL(in_shape2);
  // 添加guard校验in_shape1和in_shape2相等
  GE_ASSERT_TRUE(in_shape1->GetDimNum() == in_shape2->GetDimNum(),
                 "Input1 dim num %zu shpuld equal to Input2 dim num: %zu of Select node", in_shape1->GetDimNum(),
                 in_shape2->GetDimNum());
  for (size_t i = 0UL; i < in_shape1->GetDimNum(); i++) {
    ASSERT_SYMBOL_EQ(in_shape1->GetDim(i), in_shape2->GetDim(i));
  }
  // 校验condition和输入完全相等这一场景
  if (condition_shape->GetDimNum() == in_shape1->GetDimNum()) {
    for (size_t i = 0UL; i < condition_shape->GetDimNum(); i++) {
      ASSERT_SYMBOL_EQ(in_shape1->GetDim(i), condition_shape->GetDim(i));
    }
  } else if (condition_shape->GetDimNum() == 1U && in_shape1->GetDimNum() >= 1U) {
    // 校验condion只有一维，且和输入的第一维相等这一合法场景
    ASSERT_SYMBOL_EQ(in_shape1->GetDim(0), condition_shape->GetDim(0));
  } else if (condition_shape->GetDimNum() != 0U) {
    // 当进入这里说明他不是condion为标量，即维度数量为0，这一合法场景；报错退出
    GELOGE(PARAM_INVALID,
           "condition dim num is %zu not 0 or 1 and not equal input_shape dim num, input_shape dim num is %zu",
           condition_shape->GetDimNum(), in_shape1->GetDimNum());
    return PARAM_INVALID;
  }
  // 当维度为0,且已校验in_shape1和in_shape2相等，属于合法场景，继续进行推导
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  *out_shape = *in_shape1;
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Select).InferSymbolShape(InferShape4Select);
}  // namespace
}  // namespace ge