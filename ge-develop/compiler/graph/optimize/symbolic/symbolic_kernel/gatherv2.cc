/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/types.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kParamIndex = 0UL;
constexpr size_t kIndiceIndex = 1UL;
constexpr size_t kAxisIndex = 2UL;
constexpr size_t kBatchDimAttrIndex = 0UL;
constexpr size_t kOutputIndex = 0UL;

graphStatus GetInputDimsAndValue(gert::InferSymbolComputeContext *context, const size_t index,
    std::vector<int64_t> &input_dims, std::vector<ge::Expression> &values) {
  GE_ASSERT_NOTNULL(context);
  if (!context->GetConstInputDims(index, input_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: Get input[%zu] dim failed, node %s[%s].",
        index, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto input_tensor = context->GetInputSymbolTensor(index);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  auto value_symbols = input_tensor->GetSymbolicValue();
  if (value_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: Get input[%zu] symbolic value failed, node %s[%s].",
        index, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  values = *value_symbols;
  return GRAPH_SUCCESS;
}

graphStatus GetIndicesValue(gert::InferSymbolComputeContext *context,
    const std::vector<ge::Expression> &indice_symbol_values,
    std::vector<int64_t> &indice_values) {
  indice_values.clear();
  for (size_t i = 0UL; i < indice_symbol_values.size(); i++) {
    int64_t indice_value = 0L;
    if (!indice_symbol_values[i].GetConstValue(indice_value)) {
      GELOGW("SymbolicKernel compute unsupported, reason: get dim sym const value failed, node %s[%s].",
          context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    indice_values.emplace_back(indice_value);
  }
  return GRAPH_SUCCESS;
}

graphStatus GetAxis(gert::InferSymbolComputeContext *context, int64_t &axis) {
  auto axis_tensor = context->GetInputSymbolTensor(kAxisIndex);
  GE_UNSUPPORTED_IF_NULL(axis_tensor);
  auto axis_symbols = axis_tensor->GetSymbolicValue();
  if (axis_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: Get axis symbolic value failed, node %s[%s].",
        context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  GE_ASSERT_TRUE(axis_symbols->size() == 1UL,
      "SymbolicKernel compute failed, reason: axis num should be 1, but get:%zu, node %s[%s].",
      axis_symbols->size(), context->GetNodeName(), context->GetNodeType());
  if (!axis_symbols->front().GetConstValue(axis)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get dim sym const value failed, node %s[%s].",
        context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  return GRAPH_SUCCESS;
}

graphStatus NormalizeInput(gert::InferSymbolComputeContext *context,
    const std::vector<int64_t> &indice_dims,
    std::vector<int64_t> &param_dims,
    int64_t &axis, int64_t &batch_dims) {
  if (param_dims.empty()) {
    GELOGI("GatherV2 in_shape is scalar, set it's shape to (1,)");
    param_dims.emplace_back(1);
  }
  GE_ASSERT_TRUE((axis >= (-1) * static_cast<int64_t>(param_dims.size())) &&
      (axis < static_cast<int64_t>(param_dims.size())),
      "SymbolicKernel compute failed, reason: axis should in the range [-%zu, %zu), but got %ld, node %s[%s].",
      param_dims.size(), param_dims.size(), axis, context->GetNodeName(), context->GetNodeType());
  axis = axis < 0 ? axis + static_cast<int64_t>(param_dims.size()) : axis;
  GE_ASSERT_TRUE((batch_dims >= (-1) * static_cast<int64_t>(indice_dims.size())) &&
      (batch_dims <= static_cast<int64_t>(indice_dims.size())),
      "SymbolicKernel compute failed, reason: batch_dims should in the range [-%zu, %zu), but got %ld, node %s[%s].",
      indice_dims.size(), indice_dims.size(), batch_dims, context->GetNodeName(), context->GetNodeType());
  batch_dims = batch_dims < 0 ? batch_dims + static_cast<int64_t>(indice_dims.size()) : batch_dims;
  GE_ASSERT_TRUE(batch_dims <= axis,
      "SymbolicKernel compute failed, reason: batch_dims:%ld is must be less or equal to axis:%ld, node %s[%s].",
      batch_dims, axis, context->GetNodeName(), context->GetNodeType());
  return GRAPH_SUCCESS;
}

graphStatus CalOutputSymbolValue(gert::InferSymbolComputeContext *context,
    const std::vector<int64_t> &param_dims,
    const std::vector<ge::Expression> &param_values,
    const std::vector<int64_t> &indice_values, const int64_t axis,
    const int64_t batch_dims, std::vector<ge::Expression> &output_values) {
  int64_t outer_loop_num = 1L;
  for (int64_t i = 0L; i < batch_dims; i++) {
    outer_loop_num *= param_dims[static_cast<size_t>(i)];
  }
  int64_t block_size = 1L;
  for (int64_t i = axis + 1L; i < static_cast<int64_t>(param_dims.size()); i++) {
    block_size *= param_dims[static_cast<size_t>(i)];
  }
  const int64_t indice_block_size = static_cast<int64_t>(indice_values.size()) / outer_loop_num;
  const int64_t block_num = static_cast<int64_t>(param_values.size()) /
      outer_loop_num / block_size / param_dims[static_cast<size_t>(axis)];
  const int64_t outer_block_size = static_cast<int64_t>(param_values.size()) / outer_loop_num;
  GELOGD("param total size: %zu, indice total size:%zu, axis dim num: %lld, node %s[%s].",
      param_values.size(), indice_values.size(), param_dims[static_cast<size_t>(axis)],
      context->GetNodeName(), context->GetNodeType());
  for (int64_t i = 0L; i < outer_loop_num; i++) {
    for (int64_t j = 0L; j < block_num; j++) {
      for (int64_t k = 0L; k < indice_block_size; k++) {
        int64_t gather_index = indice_values[static_cast<size_t>(k + indice_block_size * i)];
        GE_ASSERT_TRUE(gather_index < param_dims[static_cast<size_t>(axis)],
            "SymbolicKernel compute failed, reason: indice index:%lld should less than axis:%lld dim:%lld, node %s[%s].",
            gather_index, axis, param_dims[axis], context->GetNodeName(), context->GetNodeType());
        const auto start_iter = param_values.begin() +
            (i * outer_block_size + (j * param_dims[static_cast<size_t>(axis)] + gather_index) * block_size);
        const auto end_iter = param_values.begin() +
            (i * outer_block_size + (j * param_dims[static_cast<size_t>(axis)] + gather_index + 1L) * block_size);
        output_values.insert(output_values.end(), start_iter, end_iter);
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus CalOutputSymbolShape(gert::InferSymbolComputeContext *context,
    const std::vector<int64_t> &param_dims,
    const std::vector<int64_t> &indice_dims, const int64_t axis, const int64_t batch_dims,
    gert::SymbolShape &output_shape) {
  output_shape.Clear();
  for (size_t i = 0UL; i < static_cast<size_t>(axis); i++) {
    output_shape.AppendDim(Symbol(param_dims[i]));
  }

  for (size_t i = 0UL; i < static_cast<size_t>(batch_dims); i++) {
    GE_ASSERT_TRUE(param_dims[i] == indice_dims[i],
        "SymbolicKernel compute failed, reason: param[%zu] dim size: %lld should equal to indices[%zu] dim size: %lld, node %s[%s].",
        i, param_dims[i], i, indice_dims[i], context->GetNodeName(), context->GetNodeType());
  }
  for (size_t i = static_cast<size_t>(batch_dims); i < indice_dims.size(); i++) {
    output_shape.AppendDim(Symbol(indice_dims[i]));
  }
  for (size_t i = static_cast<size_t>(axis + 1UL); i < param_dims.size(); i++) {
    output_shape.AppendDim(Symbol(param_dims[i]));
  }
  return GRAPH_SUCCESS;
}
}  // namespace

static graphStatus GatherV2SymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("GatherV2 Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  // 获取param的shape
  std::vector<int64_t> param_dims;
  std::vector<ge::Expression> param_values;
  auto ret = GetInputDimsAndValue(context, kParamIndex, param_dims, param_values);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  // 获取indice的shape
  std::vector<int64_t> indice_dims;
  std::vector<ge::Expression> indice_symbol_values;
  ret = GetInputDimsAndValue(context, kIndiceIndex, indice_dims, indice_symbol_values);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  // 获取indice_value
  std::vector<int64_t> indice_values;
  ret = GetIndicesValue(context, indice_symbol_values, indice_values);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  // 获取axis
  int64_t axis = 0L;
  ret = GetAxis(context, axis);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  // 获取batchdim
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto batch_dims_attr = attrs->GetAttrPointer<int64_t>(kBatchDimAttrIndex);
  GE_ASSERT_NOTNULL(batch_dims_attr);
  int64_t batch_dims = *batch_dims_attr;
  GE_ASSERT_SUCCESS(NormalizeInput(context, indice_dims, param_dims, axis, batch_dims));
  auto symbolic_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(symbolic_tensor);
  GE_ASSERT_SUCCESS(CalOutputSymbolShape(context, param_dims, indice_dims, axis,
      batch_dims, symbolic_tensor->MutableOriginSymbolShape()));
  auto output_symbol_values = symbolic_tensor->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_symbol_values);
  GE_ASSERT_SUCCESS(CalOutputSymbolValue(context, param_dims, param_values,
      indice_values, axis, batch_dims, *output_symbol_values));
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
      SymbolicInferUtil::DumpSymbolTensor(*symbolic_tensor).c_str());
  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(GatherV2, GatherV2SymbolicKernelCompute);
}  // namespace ge
