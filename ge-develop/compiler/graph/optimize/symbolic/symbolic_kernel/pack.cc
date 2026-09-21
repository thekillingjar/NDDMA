/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kOutputIndex = 0UL;
constexpr size_t kMaxSupportDim = 8UL;

graphStatus GetInputSymbolsValue(const gert::InferSymbolComputeContext *context,
                                 std::vector<std::vector<Expression>> &input_symbols_value) {
  for (size_t i = 0UL; i < context->GetComputeNodeInputNum(); i++) {
    auto symbol_tensor = context->GetInputSymbolTensor(i);
    GE_UNSUPPORTED_IF_NULL(symbol_tensor);
    auto symbolic_value = symbol_tensor->GetSymbolicValue();
    if (symbolic_value == nullptr) {
      GELOGW("SymbolicKernel compute unsupported, reason: Get input symbolic value failed, node %s[%s].",
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    if (!input_symbols_value.empty() && input_symbols_value[0].size() != symbolic_value->size()) {
      GELOGW(
          "SymbolicKernel compute unsupported, reason: Get input symbolic value failed, node %s[%s], size not match "
          "%zu vs %zu",
          context->GetNodeName(), context->GetNodeType(), input_symbols_value[0].size(), symbolic_value->size());
      return UNSUPPORTED;
    }
    input_symbols_value.emplace_back(*symbolic_value);
  }
  return SUCCESS;
}

void CalcOutputShapeSymbols(const std::vector<int64_t> &input_dims, const int64_t axis_dim, const int64_t input_num,
                            std::vector<Expression> &output_symbols_shape) {
  output_symbols_shape.resize(input_dims.size() + 1U);
  size_t input_index = 0UL;
  for (size_t i = 0UL; i < output_symbols_shape.size(); i++) {
    if (i == static_cast<size_t>(axis_dim)) {
      output_symbols_shape[i] = Symbol(input_num);
      continue;
    }
    output_symbols_shape[i] = Symbol(input_dims[input_index]);
    input_index++;
  }
}

graphStatus PackInputSymbolsValue(const std::vector<std::vector<Expression>> &input_symbols_value,
                                  const std::vector<int64_t> &input_dims, const int64_t axis_dim,
                                  std::vector<Expression> &output_symbols_value) {
  GE_ASSERT_TRUE(!input_symbols_value.empty());
  int64_t block_size = 1L;
  for (int64_t i = static_cast<int64_t>(input_dims.size()) - 1; i >= axis_dim; i--) {
    block_size *= input_dims[static_cast<size_t>(i)];
  }
  GE_ASSERT_TRUE(block_size > 0);
  int64_t block_num = input_symbols_value[0].size() / block_size;
  GELOGD("block_size=%u, block_num=%u", block_size, block_num);
  for (int64_t cur_index = 0UL; cur_index < block_num; cur_index++) {
    for (size_t i = 0UL; i < input_symbols_value.size(); i++) {
      auto start_iter = input_symbols_value[i].begin() + cur_index * block_size;
      auto end_iter = input_symbols_value[i].begin() + (cur_index + 1) * block_size;
      GELOGD("cur_index=%d, i=%u, vec_size=%u, start_iter=%u, end_iter=%u", cur_index, i, input_symbols_value[i].size(),
             cur_index * block_size, (cur_index + 1) * block_size);
      output_symbols_value.insert(output_symbols_value.end(), start_iter, end_iter);
    }
  }
  return SUCCESS;
}
}  // namespace

static graphStatus PackSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Pack Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());

  // 获取输入shape
  std::vector<int64_t> input_dims;
  if (!context->GetConstInputDims(0, input_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get const input dim failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  // 获取轴属性
  int64_t axis_dim = 0L;
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto axis_dim_ptr = attrs->GetInt(0);
  GE_ASSERT_NOTNULL(axis_dim_ptr);
  axis_dim = *axis_dim_ptr;

  axis_dim = axis_dim < 0 ? axis_dim + static_cast<int64_t>(input_dims.size()) + 1 : axis_dim;
  GE_ASSERT_TRUE((axis_dim >= 0) && (axis_dim <= static_cast<int64_t>(input_dims.size())),
                 "axis_dim=%lld, input_dims.size=%zu", axis_dim, input_dims.size());

  // 获取InputSymbolsValue
  std::vector<std::vector<Expression>> input_symbols_value;
  auto ret = GetInputSymbolsValue(context, input_symbols_value);
  if (ret != SUCCESS) {
    GELOGW("SymbolicKernel compute unsupported, reason: get input symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  // 计算输出的symbolsValue
  auto output_symbol_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output_symbol_tensor);
  auto output_symbols_value = output_symbol_tensor->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_symbols_value);
  GE_ASSERT_SUCCESS(PackInputSymbolsValue(input_symbols_value, input_dims, axis_dim, *output_symbols_value));

  // 计算输出shape的symbols
  std::vector<Expression> output_symbols_shape;
  CalcOutputShapeSymbols(input_dims, axis_dim, static_cast<int64_t>(input_symbols_value.size()), output_symbols_shape);
  output_symbol_tensor->MutableOriginSymbolShape().MutableDims() = output_symbols_shape;
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output_symbol_tensor).c_str());
  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(Pack, PackSymbolicKernelCompute);
}  // namespace ge
