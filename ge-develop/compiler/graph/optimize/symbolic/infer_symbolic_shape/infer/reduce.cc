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
#include "common/checker.h"
#include "common/types.h"
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
graphStatus InferShape4ReduceCommon(gert::InferSymbolShapeContext *context) {
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
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const bool *keep_dims = attrs->GetAttrPointer<bool>(0);
  GE_ASSERT_NOTNULL(keep_dims);
  if (axes_tensor->GetSymbolicValue()->empty()) {
    *out_shape = *in_shape;
    GELOGD("axes is empty tensor, will ignore infer, set output shape = input shape");
    return ge::GRAPH_SUCCESS;
  }
  auto axes_size = axes_tensor->GetSymbolicValue()->size();
  auto input1_desc = context->GetInputDesc(1);
  GE_ASSERT_NOTNULL(input1_desc);
  auto dtype = input1_desc->GetDataType();
  GE_ASSERT(dtype == DT_INT32 || dtype == DT_INT64, "axes datatype %s, must in (DT_INT32，DT_INT64)",
            TypeUtils::DataTypeToSerialString(dtype).c_str());
  if (dtype == DT_INT32) {
    return SymbolicInferUtil::ReduceDims<int32_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
  }
  return SymbolicInferUtil::ReduceDims<int64_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
}

graphStatus InferShape4ReduceDCommon(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto axes = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(axes);
  const bool *keep_dims = attrs->GetAttrPointer<bool>(1);
  GE_ASSERT_NOTNULL(keep_dims);
  auto axes_size = axes->GetSize();
  if (axes_size == 0) {
    *out_shape = *in_shape;
    GELOGD("axes is empty tensor, will ignore infer, set output shape = input shape");
    return ge::GRAPH_SUCCESS;
  }
  std::vector<int64_t> reduce_axes;
  reduce_axes.resize(axes->GetSize());
  for (size_t i = 0; i < axes->GetSize(); ++i) {
    reduce_axes.at(i) = axes->GetData()[i];
  }
  if (*keep_dims) {
    return SymbolicInferUtil::ReduceDimsWithKeepDims<int64_t>(in_shape, reduce_axes, axes_size, out_shape);
  }
  return SymbolicInferUtil::ReduceDimsWithoutKeepDims<int64_t>(in_shape, reduce_axes, axes_size, out_shape);
}

graphStatus InferShape4BiasAddGrad(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto data_format = attrs->GetStr(0);
  GE_ASSERT_NOTNULL(data_format);
  std::vector<int64_t> reduce_axes;
  if (strcmp(data_format, "NCHW") != 0 && strcmp(data_format, "NHWC") != 0) {
    GELOGE(PARAM_INVALID, "data_format should be NHWC or NCHW");
    return ge::PARAM_INVALID;
  }
  // 降维到只有特征维度，NHWC特征维度在最后一个， NCHW特征维度在倒数第三个
  const int64_t nhwc_min_dim_size = 1;
  int64_t dim_size = static_cast<int64_t>(in_shape->GetDimNum());
  GE_ASSERT_TRUE(dim_size >= nhwc_min_dim_size);
  int64_t ignore_reduce_axes = dim_size - nhwc_min_dim_size;
  if (strcmp(data_format, "NCHW") == 0) {
    const int64_t nchw_min_dim_size = 3;
    GE_ASSERT_TRUE(dim_size >= nchw_min_dim_size);
    ignore_reduce_axes = dim_size - nchw_min_dim_size;
  }
  for (int64_t i = 0; i < dim_size; i++) {
    if (ignore_reduce_axes != i) {
      reduce_axes.emplace_back(i);
    }
  }

  return SymbolicInferUtil::ReduceDimsWithoutKeepDims<int64_t>(in_shape, reduce_axes, reduce_axes.size(), out_shape);
}

graphStatus InferShape4LayerNormBetaGammaBackpropV2(gert::InferSymbolShapeContext *context) {
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto dy_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(dy_shape);
  auto res_for_gamma_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(res_for_gamma_shape);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  auto out1_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(out1_shape);

  //shape_gamma
  auto cv = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(cv);
  std::vector<int64_t> dims;
  for (size_t i = 0; i < cv->GetSize(); i++) {
    int64_t expr_value = 0;
    expr_value = cv->GetData()[i];
    dims.push_back(expr_value);
  }

  auto it_found = std::find(dims.begin(), dims.end(), -1);
  std::vector<int64_t> reduce_axes;
  if (it_found == dims.end()) {
    //shape is shape as shape_gamma
    out_shape->Clear();
    for (auto dim : dims) {
      out_shape->AppendDim(Symbol(dim));
      out1_shape->AppendDim(Symbol(dim));
    }
    return ge::GRAPH_SUCCESS;
  }

  // dy_shape[s0, s1, s2] dims[-1, 0, -1] reduce_axs[0, 2]  output[1, s1, 1]
  for (int64_t i = 0; i < static_cast<int64_t>(dims.size()); i++) {
    if (dims[i] == -1) {
      reduce_axes.emplace_back(i);
    }
  }
  GE_ASSERT_GRAPH_SUCCESS(SymbolicInferUtil::Broadcast({dy_shape->GetDims(), res_for_gamma_shape->GetDims()}, out_shape->MutableDims()));
  GE_ASSERT_GRAPH_SUCCESS(SymbolicInferUtil::ReduceDimsWithKeepDims<int64_t>(out_shape, reduce_axes, reduce_axes.size(), out_shape));
  GE_ASSERT_GRAPH_SUCCESS(SymbolicInferUtil::ReduceDimsWithKeepDims<int64_t>(dy_shape, reduce_axes, reduce_axes.size(), out1_shape));
  return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4LayerNormV3(gert::InferSymbolShapeContext *context) {
  auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  auto gamma_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(gamma_shape);
  auto beta_shape = context->GetInputSymbolShape(2);
  GE_UNSUPPORTED_IF_NULL(beta_shape);
  auto out_y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_y_shape);
  auto out_mean_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(out_mean_shape);
  auto out_rstd_shape = context->GetOutputSymbolShape(2);
  GE_ASSERT_NOTNULL(out_rstd_shape);
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);

  size_t real_dim_num = x_shape->GetDimNum();
  auto begin_norm_axis_ptr = attrs->GetInt(0);
  GE_ASSERT_NOTNULL(begin_norm_axis_ptr);
  int64_t begin_norm_axis = *(begin_norm_axis_ptr);
  if (begin_norm_axis < 0) {
    begin_norm_axis += static_cast<int64_t>(real_dim_num);
  }
  if (begin_norm_axis < 0 || static_cast<size_t>(begin_norm_axis) >= real_dim_num) {
    GELOGE(PARAM_INVALID, "the op layernormv3 does not support beginNormAxis"
                          "(%ld) large than shape dims(%lu)", begin_norm_axis, real_dim_num);
    return ge::PARAM_INVALID;
  }
  // the shape of y is equal with input x
  *out_y_shape = *x_shape;

  vector<int64_t> reduce_axes;
  for (int64_t i = begin_norm_axis; i < static_cast<int64_t>(real_dim_num); i++) {
    reduce_axes.emplace_back(i);
  }

  GE_ASSERT_GRAPH_SUCCESS(SymbolicInferUtil::ReduceDimsWithKeepDims<int64_t>(x_shape, reduce_axes, reduce_axes.size(), out_mean_shape));
  GE_ASSERT_GRAPH_SUCCESS(SymbolicInferUtil::ReduceDimsWithKeepDims<int64_t>(x_shape, reduce_axes, reduce_axes.size(), out_rstd_shape));
  return ge::GRAPH_SUCCESS;
}

graphStatus InferShapeForBNTrainingUpdateGrad(gert::InferSymbolShapeContext *context) {
  const auto input_batch_mean_shape = context->GetInputSymbolShape(2);
  GE_UNSUPPORTED_IF_NULL(input_batch_mean_shape);
  auto output_diff_scale_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(output_diff_scale_shape);
  auto output_diff_offset_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(output_diff_offset_shape);
  *output_diff_scale_shape = *input_batch_mean_shape;
  *output_diff_offset_shape = *input_batch_mean_shape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BNTrainingUpdateGrad).InferSymbolShape(InferShapeForBNTrainingUpdateGrad);

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceSum).InferSymbolShape(InferShape4ReduceCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceMax).InferSymbolShape(InferShape4ReduceCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceMin).InferSymbolShape(InferShape4ReduceCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceProd).InferSymbolShape(InferShape4ReduceCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceMean).InferSymbolShape(InferShape4ReduceCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceAll).InferSymbolShape(InferShape4ReduceCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceAny).InferSymbolShape(InferShape4ReduceCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceSumD).InferSymbolShape(InferShape4ReduceDCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceMaxD).InferSymbolShape(InferShape4ReduceDCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceMinD).InferSymbolShape(InferShape4ReduceDCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceProdD).InferSymbolShape(InferShape4ReduceDCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceMeanD).InferSymbolShape(InferShape4ReduceDCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceAllD).InferSymbolShape(InferShape4ReduceDCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReduceAnyD).InferSymbolShape(InferShape4ReduceDCommon);

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BiasAddGrad).InferSymbolShape(InferShape4BiasAddGrad);

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LayerNormBetaGammaBackpropV2).InferSymbolShape(InferShape4LayerNormBetaGammaBackpropV2);

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LayerNormV3).InferSymbolShape(InferShape4LayerNormV3);
}  // namespace
}  // namespace ge