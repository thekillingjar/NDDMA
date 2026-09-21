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
#include <graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h>

namespace ge {
namespace {
constexpr size_t kMatmulDimNum = 2U;
constexpr size_t kZero = 0U;
constexpr size_t kOne = 1U;
/**
 * Range 算子的符号化shape推导
 * 【算子功能】按照指定的步长(delta)，在给定的数据范围[start, limit)内，生成一组数字序列。
 * 【算子约束】
 *      1. 起点(start)允许为空，默认值为0
 *      2. 终点(limit)不允许为空
 *      3. 步长(delta)允许为空，默认值为1
 * 【推导逻辑】
 *      1. 三个输入均为数据依赖，若参数为空，则按照算子约束设置默认值；若每个参数有多个元素，则只取第0个元素。
 *      2. 计算取值范围: range = limit - start
 *      3. 计算取值个数: range_num = CEIL(range / delta)  -- 范围除以步长并下取整，因为取值范围不包含limit
 *  [场景处理]
 *      1. 当有且只有第一个输入有值，第一个值视作limit，其余采用默认值
 *      2. 当有且只有第一个与第二个输入有值，第一个值视作start，第二个值视作limit，delta采用默认值
 *      3. 当三个输入都有值，按照正常情况计算
 *      4. 非上述三种情况，Tensorflow视为异常场景报错，而我们将返回UnSupport
 */

bool CheckValid(const gert::SymbolTensor *symbol_tensor) {
  if (symbol_tensor == nullptr) {
    return false;
  }
  auto value_symbol = symbol_tensor->GetSymbolicValue();
  if (value_symbol == nullptr || value_symbol->empty()) {
    return false;
  }
  return true;
}

graphStatus InferShape4Range(gert::InferSymbolShapeContext *context) {
  auto start_tensor = context->GetInputSymbolTensor(0);
  GE_UNSUPPORTED_IF_NULL(start_tensor);
  auto limit_tensor = context->GetInputSymbolTensor(1);
  auto delta_tensor = context->GetInputSymbolTensor(2);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  Expression start{Symbol(0)};
  Expression limit = start;
  Expression delta{Symbol(1)};
  bool valid_start = CheckValid(start_tensor);
  bool valid_limit = CheckValid(limit_tensor);
  bool valid_delta = CheckValid(delta_tensor);
  if (valid_start && valid_limit && valid_delta) {
    // 当三个输入都有值，按照正常情况计算
    start = start_tensor->GetSymbolicValue()->at(0);
    limit = limit_tensor->GetSymbolicValue()->at(0);
    delta = delta_tensor->GetSymbolicValue()->at(0);
  } else if (valid_start && valid_limit && !valid_delta) {
    // 当有且只有第一个与第二个输入有值，第一个值视作start，第二个值视作limit，delta采用默认值
    start = start_tensor->GetSymbolicValue()->at(0);
    limit = limit_tensor->GetSymbolicValue()->at(0);
  } else if (valid_start && !valid_limit && !valid_delta) {
    // 当有且只有第一个输入有值，第一个值视作limit，其余采用默认值
    limit = start_tensor->GetSymbolicValue()->at(0);
  } else {
    GELOGW("SymbolicKernel compute unsupported, reason: Get symbolic value [%d%d%d], node %s[%s].", valid_start,
           valid_limit, valid_delta, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  if (EXPECT_SYMBOL_GT(limit, start)) {
    ASSERT_SYMBOL_GT(delta, kSymbolZero);
  } else if (EXPECT_SYMBOL_LT(limit, start)) {
    ASSERT_SYMBOL_LT(delta, kSymbolZero);
  }
  ASSERT_SYMBOL_NE(delta, kSymbolZero);
  auto range_length = limit - start;
  auto range_num = sym::Ceiling(range_length / delta);
  out_shape->Clear();
  out_shape->AppendDim(range_num);
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Range).InferSymbolShape(InferShape4Range);
}  // namespace
}  // namespace ge