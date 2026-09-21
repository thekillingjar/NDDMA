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
#include "common/util/mem_utils.h"
#include "common/util.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kInputIndex = 0UL;
constexpr size_t kMaxSupportDim = 8UL;

graphStatus UnpackInputSymbolsValue(const std::vector<int64_t> &input_dims,
                                    const std::vector<Expression> &input_symbols, const int64_t axis,
                                    std::vector<std::vector<Expression>> &output_symbols) {
  output_symbols.resize(input_dims[static_cast<size_t>(axis)]);
  int64_t block_size = 1L;
  for (int64_t i = static_cast<int64_t>(input_dims.size()) - 1; i >= 0; i--) {
    if (i == axis) {
      break;
    }
    block_size *= input_dims[i];
  }
  GE_ASSERT_TRUE(block_size > 0);
  size_t block_num = input_symbols.size() / static_cast<size_t>(block_size);
  size_t output_index = 0UL;
  for (size_t i = 0UL; i < block_num; i++) {
    auto start_iter = input_symbols.begin() + block_size * i;
    auto end_iter = input_symbols.begin() + block_size * (i + 1);
    output_symbols[output_index].insert(output_symbols[output_index].end(), start_iter, end_iter);
    output_index = (output_index + 1U) % output_symbols.size();
  }
  return SUCCESS;
}

graphStatus CalcOutputShapeSymbol(const std::vector<int64_t> &input_dims, const int64_t axis,
                                  std::vector<Expression> &output_shape_symbols) {
  for (size_t i = 0UL; i < input_dims.size(); i++) {
    if (i != static_cast<size_t>(axis)) {
      output_shape_symbols.emplace_back(Symbol(input_dims[i]));
    }
  }
  return SUCCESS;
}
}  // namespace

static graphStatus UnpackSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Unpack Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  // 获取输入的dims
  std::vector<int64_t> input_dims;
  if (!context->GetConstInputDims(kInputIndex, input_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get const input dim failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  // 获取输入的值
  auto input_symbols = context->GetInputSymbolTensor(kInputIndex)->GetSymbolicValue();
  if (input_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get input symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  // 获取轴信息
  int64_t axis_dim = 0L;
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto axis_dim_ptr = attrs->GetInt(static_cast<size_t>(1));
  GE_ASSERT_NOTNULL(axis_dim_ptr);
  axis_dim = *axis_dim_ptr;
  axis_dim = axis_dim < 0 ? axis_dim + static_cast<int64_t>(input_dims.size()) : axis_dim;
  GE_ASSERT_TRUE((axis_dim >= 0) && (axis_dim < static_cast<int64_t>(input_dims.size())),
                 "axis_dim=%lld, input_dims.size=%zu", axis_dim, input_dims.size());
  // 计算输出shape
  std::vector<Expression> output_shape_symbols;
  GE_ASSERT_SUCCESS(CalcOutputShapeSymbol(input_dims, axis_dim, output_shape_symbols));
  // 计算输出
  std::vector<std::vector<Expression>> output_symbols;
  GE_ASSERT_SUCCESS(UnpackInputSymbolsValue(input_dims, *input_symbols, axis_dim, output_symbols));
  // 刷新输出shape
  GE_ASSERT_TRUE(output_symbols.size() == context->GetComputeNodeOutputNum(),
                 "Dim num:%zu of index: %lld is not equal to output num: %zu", output_symbols.size(), axis_dim,
                 context->GetComputeNodeOutputNum());
  for (size_t i = 0UL; i < output_symbols.size(); i++) {
    auto symbolic_tensor = context->GetOutputSymbolTensor(i);
    GE_ASSERT_NOTNULL(symbolic_tensor);
    auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression>>(output_symbols[i]);
    if (symbolic_value_unique != nullptr) {
      symbolic_tensor->SetSymbolicValue(std::move(symbolic_value_unique));
    }

    symbolic_tensor->MutableOriginSymbolShape().MutableDims() = output_shape_symbols;
    GELOGD("%s[%s] kernel success, output[%zu] %s", context->GetNodeName(), context->GetNodeType(), i,
           SymbolicInferUtil::DumpSymbolTensor(*symbolic_tensor).c_str());
  }

  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(Unpack, UnpackSymbolicKernelCompute);
}  // namespace ge
