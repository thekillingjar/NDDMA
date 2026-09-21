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
graphStatus GetConstInt(const Expression &expr, DataType dt, int64_t &value) {
  if (dt == DT_INT32) {
    int32_t tmp_value = 0;
    if (expr.GetConstValue<int32_t>(tmp_value) == false) {
      return UNSUPPORTED;
    }
    value = static_cast<int64_t>(tmp_value);
  } else if (dt == DT_INT64) {
    if (expr.GetConstValue<int64_t>(value) == false) {
      return UNSUPPORTED;
    }
  } else {
    GELOGE(PARAM_INVALID, "dt must in [int32, int64]");
    return ge::PARAM_INVALID;
  }
  return ge::GRAPH_SUCCESS;
}

graphStatus ReshapeInferCommon(const gert::InferSymbolShapeContext *context, const gert::SymbolShape *in_shape, gert::SymbolShape *out_shape,
                               const gert::SymbolTensor *shape_tensor, DataType dt) {
  auto reshape_dim_num = shape_tensor->GetSymbolicValue()->size();
  size_t unknown_dim_idx = std::numeric_limits<size_t>::max();
  // expr可能是常量或者符号
  for (size_t i = 0; i < reshape_dim_num; i++) {
    auto dim_expr = shape_tensor->GetSymbolicValue()->at(i);
    if (dim_expr.IsConstExpr()) {
      // 如果dim是常量，只能是int32或者int64类型
      int64_t dim = -2;
      if (GetConstInt(dim_expr, dt, dim) == UNSUPPORTED) {
        GELOGW("Symbol Infer unsupported, get dim at index[%zu] is not constvalue, node %s[%s]", i,
               context->GetNodeName(), context->GetNodeType());
        return UNSUPPORTED;
      }
      if (dim == 0) {
        // 输入为0表示使用输入的维度
        GE_ASSERT_TRUE(i < in_shape->GetDimNum());
        out_shape->AppendDim(in_shape->GetDim(i));
      } else if (dim == -1) {
        // 输入为-1表示该维度不确定需要等其它维度确定后最后计算，先用1占位，并记录该维度
        GE_ASSERT_TRUE(unknown_dim_idx == std::numeric_limits<size_t>::max());
        out_shape->AppendDim(Symbol(1));
        unknown_dim_idx = i;
      } else {
        out_shape->AppendDim(dim_expr);
      }
    } else {
      out_shape->AppendDim(dim_expr);
    }
  }
  Expression in_shape_size = in_shape->GetSymbolShapeSize();
  Expression out_shape_size = out_shape->GetSymbolShapeSize();
  if (unknown_dim_idx == std::numeric_limits<size_t>::max()) {
    // 添加guard out_shape_size == in_shape_size
    ASSERT_SYMBOL_EQ(in_shape_size, out_shape_size);
  } else {
    // todo 添加guard dynamic_dim为整数, 依赖mod运算
    // 计算不确定的维度
    auto dynamic_dim = in_shape_size / out_shape_size;
    out_shape->MutableDims()[unknown_dim_idx] = dynamic_dim;
  }
  return ge::GRAPH_SUCCESS;
}

/**
 * Reshape算子符号化推导，该算子能改变输入数据的形状，根据指定的形状参数对输入的数据进行重排列，输出重新排列后的数据。
 * data：输入数据
 * shape：一个列表，用来定义输出数据的维度，0表示该维度跟输入数据一致，-1表示剩下的数据放入该维度
 * 例如
 * data: {2, 3, 4}
 * shape [1, 0, -1, 2]
 * output:
 * dim1 = 1
 * dim2 = 3
 * dim4 = 2
 * dim3 = (2 * 3 * 4) / (1 * 3 * 2) = 4
 * {1, 3, 4, 2}
 */
graphStatus InferShape4Reshape(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto shape_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(shape_tensor);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  // 输入为空列表不需要进行reshape
  if (shape_tensor->GetSymbolicValue() == nullptr || shape_tensor->GetSymbolicValue()->empty()) {
    GELOGW("Symbol Infer unsupported, get symbolic value is nullptr or empty, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  auto shape_desc = context->GetInputDesc(1);
  GE_ASSERT_NOTNULL(shape_desc);
  auto dt = shape_desc->GetDataType();
  return ReshapeInferCommon(context, in_shape, out_shape, shape_tensor, dt);
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Reshape).InferSymbolShape(InferShape4Reshape);
}  // namespace
}  // namespace ge