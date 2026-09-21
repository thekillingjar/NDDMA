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
 * Fill算子 infer shape函数实现, dims中的值就是输出shape的维度值，设置到output的shape上
 *
 * 数据依赖 dims：一个1-D张量，表示输出张量的形状，类型为int32或int64，非负整数 ()
 */
graphStatus InferShape4Fill(gert::InferSymbolShapeContext *context) {
  const auto dims = context->GetInputSymbolTensor(0);
  GE_UNSUPPORTED_IF_NULL(dims);
  const auto dims_value = dims->GetSymbolicValue();
  if (dims_value == nullptr) {
    GELOGW("Symbol Infer unsupported, get symbolic value is nullptr, node %s[%s]", context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  const auto dims_size = dims_value->size();
  GE_ASSERT_TRUE(dims_size > 0, "Invalid dims %d, must be non-empty", dims_size);

  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  out_shape->MutableDims().clear();
  for (int i = 0; i < static_cast<int>(dims_size); i++) {
    auto dim = dims_value->at(i);
    out_shape->MutableDims().emplace_back(dim);
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Fill).InferSymbolShape(InferShape4Fill);
}  // namespace
}  // namespace ge