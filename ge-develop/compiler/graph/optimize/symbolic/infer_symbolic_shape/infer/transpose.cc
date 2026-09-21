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
template <typename T>
bool TransposeInferCommon(gert::InferSymbolShapeContext *context, const gert::SymbolShape *in_shape,
                          std::vector<Expression> &exprs, gert::SymbolShape *out_shape) {
  auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return false;
  }
  int64_t inserted_by_fe = 0;
  if (attrs->GetAttrNum() > 0) {
    auto inserted_by_fe_flag = attrs->GetInt(0);
    if (inserted_by_fe_flag != nullptr) {
      inserted_by_fe = *inserted_by_fe_flag;
    }
  }
  size_t input_dim_size = in_shape->GetDimNum();
  if (inserted_by_fe == 0) {
    for (auto &expr : exprs) {
      T perm_v;
      expr.GetConstValue<T>(perm_v);
      perm_v = perm_v >= 0 ? perm_v : perm_v + input_dim_size;
      if (perm_v < 0 || static_cast<size_t>(perm_v) >= input_dim_size) {
        return false;
      }
      out_shape->AppendDim(in_shape->GetDim(perm_v));
    }
  } else {
    for (size_t i = 0; i < exprs.size(); i++) {
      out_shape->AppendDim(in_shape->GetDim(i));
    }
  }
  return true;
}

/**
 * Transpose算子符号化推导，交换一个张量的维度
 * input_shape0：输入张量的shape
 * perm：转置后张量的维度在原张量上的映射，整型列表，列表长度跟输入张量的维度相同，其中的元素要求>=0，且小于输入张量的最大维度
 * 例如
 * 输入张量为{1, 2, 3}
 * perm为[2, 0, 1]
 * 转置后的张量为{3, 1, 2}
 */
graphStatus InferShape4Transpose(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto perm_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(perm_tensor);
  if (perm_tensor->GetSymbolicValue() == nullptr) {
    GELOGW("Symbol Infer unsupported, get perm_tensor symbolic value is nullptr, node %s[%s]", context->GetNodeName(),
       context->GetNodeType());
    return UNSUPPORTED;
  }
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  auto perm_tensor_exprs = *perm_tensor->GetSymbolicValue();
  int64_t perm_size = perm_tensor_exprs.size();
  size_t input_dim_size = in_shape->GetDimNum();
  GE_ASSERT_EQ(static_cast<int64_t>(input_dim_size), perm_size);
  auto perm_desc = context->GetInputDesc(1);
  GE_ASSERT_NOTNULL(perm_desc);
  auto dt = perm_desc->GetDataType();
  if (dt == DT_INT64) {
    GE_ASSERT_TRUE(TransposeInferCommon<int64_t>(context, in_shape, perm_tensor_exprs, out_shape));
  } else if (dt == DT_INT32) {
    GE_ASSERT_TRUE(TransposeInferCommon<int32_t>(context, in_shape, perm_tensor_exprs, out_shape));
  } else {
    return ge::PARAM_INVALID;
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Transpose).InferSymbolShape(InferShape4Transpose);
}  // namespace
}  // namespace ge