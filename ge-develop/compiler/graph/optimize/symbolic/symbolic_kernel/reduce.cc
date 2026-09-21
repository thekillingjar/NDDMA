/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <list>
#include <numeric>
#include "framework/common/types.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kXInputIndex = 0UL;
constexpr size_t kAxisInputIndex = 1UL;
constexpr size_t kOutputIndex = 0UL;

graphStatus GetAxisDims(const gert::InferSymbolComputeContext *context, std::vector<int64_t> &axis_dims) {
  std::vector<int64_t> axis_input_shape;
  context->GetConstInputDims(kAxisInputIndex, axis_input_shape);
  if (axis_input_shape.size() != 1) {
    GELOGW("SymbolicKernel compute unsupported, reason: axis_input_shape(%zu) not equal 1, node %s[%s].",
           axis_input_shape.size(), context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto axis_tensor = context->GetInputSymbolTensor(kAxisInputIndex);
  GE_UNSUPPORTED_IF_NULL(axis_tensor);
  auto axis_symbols = axis_tensor->GetSymbolicValue();
  if (axis_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get %zu symbolic value failed, node %s[%s].", kAxisInputIndex,
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  for (size_t i = 0UL; i < axis_symbols->size(); i++) {
    int64_t dim = 0L;
    if (!(*axis_symbols)[i].GetConstValue(dim)) {
      GELOGW("SymbolicKernel compute unsupported, reason: get %zu input const value failed, node %s[%s].", i,
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    axis_dims.emplace_back(dim);
  }
  return SUCCESS;
}

Status NormalizeAxisDims(const int64_t input_dims_size, std::vector<int64_t> &axis_dims) {
  for (size_t i = 0UL; i < axis_dims.size(); i++) {
    axis_dims[i] = axis_dims[i] < 0 ? axis_dims[i] + input_dims_size : axis_dims[i];
    GE_ASSERT_TRUE((axis_dims[i] >= 0) && (axis_dims[i] < input_dims_size), "axis: %lld should in range [0, %lld)",
                   axis_dims[i], input_dims_size);
  }
  std::sort(axis_dims.begin(), axis_dims.end());
  GELOGI("Axis dims: %s", SymbolicInferUtil::VectorToStr(axis_dims).c_str());
  return SUCCESS;
}
Status CalcOutputShape(const std::vector<int64_t> &input_dims, const std::vector<int64_t> &axis_dims,
                       const bool keep_dims, std::vector<Expression> &output_symbol_shape) {
  std::set<int64_t> axis_set(axis_dims.begin(), axis_dims.end());
  for (size_t i = 0UL; i < input_dims.size(); i++) {
    if (axis_set.count(i) > 0) {
      if (keep_dims) {
        output_symbol_shape.emplace_back(Symbol(1));
      }
      continue;
    }
    output_symbol_shape.emplace_back(Symbol(input_dims[i]));
  }
  return SUCCESS;
}
Status ReduceProdOutputSymbolValue(const std::vector<Expression> &input_symbols, const std::vector<int64_t> &input_dims,
                                   const std::vector<int64_t> &axis_dims, std::vector<Expression> &output_symbols) {
  std::vector<Expression> last_output_symbols = input_symbols;
  for (size_t axis_pos = 0UL; axis_pos < axis_dims.size(); axis_pos++) {
    output_symbols.clear();
    int64_t reduce_axis = axis_dims[axis_pos];
    // 累乘得到blocksize
    int64_t block_size =
        std::accumulate(input_dims.begin() + reduce_axis + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    int64_t block_num = static_cast<int64_t>(last_output_symbols.size()) / block_size / input_dims[reduce_axis];
    GELOGI("block num: %lld, block size: %lld, index: %lld", block_num, block_size, reduce_axis);
    for (int64_t i = 0UL; i < block_num; i++) {
      std::vector<Expression> temp_result(last_output_symbols.begin() + i * input_dims[reduce_axis] * block_size,
                                          last_output_symbols.begin() + (i * input_dims[reduce_axis] + 1) * block_size);
      for (int64_t j = 1L; j < input_dims[reduce_axis]; j++) {
        auto start_iter = last_output_symbols.begin() + (i * input_dims[reduce_axis] + j) * block_size;
        std::transform(temp_result.begin(), temp_result.end(), start_iter, temp_result.begin(),
                       [](const Expression &a, const Expression &b) -> Expression { return a * b; });
      }
      output_symbols.insert(output_symbols.end(), temp_result.begin(), temp_result.end());
    }
    last_output_symbols = output_symbols;
  }
  return SUCCESS;
}
}  // namespace

static graphStatus ReduceProdSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("ReduceProd Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());

  // 获取输入shape
  std::vector<int64_t> input_x_dims;
  if (!context->GetConstInputDims(kXInputIndex, input_x_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get const input dim failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  // 获取轴属性
  bool keep_dims = false;
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto keep_dims_ptr = attrs->GetBool(0);
  GE_ASSERT_NOTNULL(keep_dims_ptr);
  keep_dims = *keep_dims_ptr;

  // 获取InputSymbolsValue
  // 获取输入的值
  auto input_x_symbols = context->GetInputSymbolTensor(kXInputIndex)->GetSymbolicValue();
  if (input_x_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get input symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  // 获取axis
  std::vector<int64_t> axis_dims;
  auto ret = GetAxisDims(context, axis_dims);
  if (ret != SUCCESS) {
    return ret;
  }

  GE_ASSERT_TRUE(axis_dims.size() <= input_x_dims.size(), "Axis num: %zu should not more than input shape dims: %zu",
                 axis_dims.size(), input_x_dims.size());

  // 归一化轴信息
  GE_ASSERT_SUCCESS(NormalizeAxisDims(static_cast<int64_t>(input_x_dims.size()), axis_dims));
  std::vector<Expression> output_symbol_shape;
  GE_ASSERT_SUCCESS(CalcOutputShape(input_x_dims, axis_dims, keep_dims, output_symbol_shape));

  auto out_symbolic_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(out_symbolic_tensor);
  out_symbolic_tensor->MutableOriginSymbolShape().MutableDims() = output_symbol_shape;

  // 扩展x_symbols
  auto output_symbols = out_symbolic_tensor->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_symbols);
  GE_ASSERT_SUCCESS(ReduceProdOutputSymbolValue(*input_x_symbols, input_x_dims, axis_dims, *output_symbols));
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*out_symbolic_tensor).c_str());
  return SUCCESS;
}

REGISTER_SYMBOLIC_KERNEL(ReduceProd, ReduceProdSymbolicKernelCompute);
}  // namespace ge
