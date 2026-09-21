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
#include "graph/utils/type_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {

template <typename T>
graphStatus ExpandDimsInferShapeImpl(const gert::InferSymbolShapeContext *context, const gert::SymbolShape *x_shape, const gert::SymbolTensor *axis_tensor,
                                     gert::SymbolShape *output_shape) {
  T axis_data;  // ExpandDims only support one axis
  if (axis_tensor->GetSymbolicValue()->at(0).GetConstValue<T>(axis_data) == false) {
    GELOGW("Symbol Infer unsupported, axis value is not constvalue, node %s[%s]", context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  axis_data = (axis_data < 0) ? (axis_data + static_cast<T>(x_shape->GetDimNum() + 1)) : axis_data;
  if (axis_data < 0 || axis_data > static_cast<int64_t>(x_shape->GetDimNum())) {
    GELOGE(PARAM_INVALID, "axis[%d] is not in [-%d,%d]", static_cast<int64_t>(axis_data), x_shape->GetDimNum() + 1,
           x_shape->GetDimNum());
    return GRAPH_FAILED;
  }
  *output_shape = *x_shape;
  auto &dims = output_shape->MutableDims();
  dims.insert(dims.begin() + axis_data, Symbol(1));
  return GRAPH_SUCCESS;
}

/**
 * ExpandDims算子 算子的符号化shape推导
 * 【算子功能】在多维数组（张量）中指定的位置插入一个新的维度，其大小为1
 * 【算子约束】
 *      沿着制定维度拼接的轴数必须合法（在已有轴数范围内 -input_shape.dimNums.size-1<=axis<=input_shape.dimNums.size）
 *      举例：输入=(3,8,10)，axis=1，则上述输入的所有轴需要一致，且拼接轴范围在【-4，3】。输出=(3,1，8,10)
 * 【推导逻辑】
 *      1. 按照算子约束校验各个输入输出非空、axis值合法
 *      2. 设置拼接轴所在维度的值为1
 */
graphStatus InferShape4ExpandDims(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto axes_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(axes_tensor);
  if (axes_tensor->GetSymbolicValue() == nullptr) {
    GELOGW("Symbol Infer unsupported, get symbolic value is nullptr, node %s[%s]", context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  // check axis shape is valid. Only scalar or single element tensor is allowed.
  auto axes_size = static_cast<int32_t>(axes_tensor->GetSymbolicValue()->size());
  if (axes_size != 1) {
    GELOGE(PARAM_INVALID, "expand dims failed, axis input must be a tensor with a single value, actually rank = %d",
           axes_size);
    return GRAPH_FAILED;
  }
  GE_ASSERT_NOTNULL(context->GetInputDesc(1));
  auto dtype = context->GetInputDesc(1)->GetDataType();
  GE_ASSERT(dtype == DT_INT32 || dtype == DT_INT64, "axes datatype %s, must in (DT_INT32, DT_INT64)",
            TypeUtils::DataTypeToSerialString(dtype).c_str());
  if (dtype == DT_INT32) {
    return ExpandDimsInferShapeImpl<int32_t>(context, in_shape, axes_tensor, out_shape);
  }
  return ExpandDimsInferShapeImpl<int64_t>(context, in_shape, axes_tensor, out_shape);
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ExpandDims).InferSymbolShape(InferShape4ExpandDims);
}  // namespace
}  // namespace ge