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
#include "common/util.h"
#include "common/checker.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
graphStatus BuildOutWithOneInput(const gert::InferSymbolComputeContext *context, gert::SymbolTensor &out_tesnor) {
  std::vector<int64_t> input_dims;
  auto symbol_tensor = context->GetInputSymbolTensor(0);
  GE_UNSUPPORTED_IF_NULL(symbol_tensor);
  auto symbolic_value = symbol_tensor->GetSymbolicValue();
  if (symbolic_value == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: no symbolic value, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  if (!context->GetConstInputDims(0, input_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: Get input dim failed, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  std::vector<Expression> out_dims;
  for (auto dim : input_dims) {
    out_dims.push_back(Symbol(dim));
  }
  out_tesnor.SetSymbolicValue(MakeUnique<std::vector<Expression>>(*symbolic_value));
  out_tesnor.MutableOriginSymbolShape().MutableDims() = out_dims;
  return SUCCESS;
}

graphStatus GetInputsValueAndShape(const gert::InferSymbolComputeContext *context, const int64_t N,
                                   std::vector<std::vector<Expression>> &inputs_values,
                                   std::vector<std::vector<int64_t>> &inputs_dims) {
  size_t dim_num = 0;
  for (int64_t i = 0; i < N; ++i) {
    std::vector<int64_t> input_dims;
    GE_UNSUPPORTED_IF_NULL(context->GetInputSymbolTensor(i));
    auto input_sym_values = context->GetInputSymbolTensor(i)->GetSymbolicValue();
    if (input_sym_values == nullptr) {
      GELOGW("SymbolicKernel compute unsupported, reason: no symbolic value, node %s[%s].", context->GetNodeName(),
             context->GetNodeType());
      return UNSUPPORTED;
    }
    if (!context->GetConstInputDims(i, input_dims)) {
      GELOGW("SymbolicKernel compute unsupported, reason: get input dims failed, node %s[%s].", context->GetNodeName(),
             context->GetNodeType());
      return UNSUPPORTED;
    }
    GELOGD("input_dims = %s", SymbolicInferUtil::VectorToStr(input_dims).c_str());
    GELOGD("input_sym_values = %s", SymbolicInferUtil::VectorExpressionToStr(*input_sym_values).c_str());
    if (i == 0) {
      dim_num = input_dims.size();
    } else {
      GE_ASSERT_EQ(dim_num, input_dims.size());
    }
    inputs_values.push_back(*input_sym_values);
    inputs_dims.push_back(input_dims);
  }
  return SUCCESS;
}

graphStatus BuildOutWithScalarInput(const std::vector<std::vector<Expression>> &inputs_values, const int64_t attr_N,
                                    gert::SymbolTensor &out_tesnor) {
  std::vector<Expression> out_values;
  for (auto input_values : inputs_values) {
    out_values.insert(out_values.end(), input_values.begin(), input_values.end());
  }
  std::vector<Expression> out_dims = {Symbol(attr_N)};
  out_tesnor.SetSymbolicValue(MakeUnique<std::vector<Expression>>(out_values));
  out_tesnor.MutableOriginSymbolShape().MutableDims() = out_dims;
  return SUCCESS;
}

graphStatus CalOutShape(const std::vector<std::vector<int64_t>> &inputs_dims, const int64_t concat_dim,
                        gert::SymbolTensor &out_tensor) {
  auto out_dims = inputs_dims[0];
  GE_ASSERT_TRUE(static_cast<size_t>(concat_dim) < out_dims.size());
  auto concat_value = out_dims[concat_dim];
  for (size_t i = 1; i < inputs_dims.size(); ++i) {
    auto input_i = inputs_dims[i];
    // 除拼接轴外其它轴要求相等
    for (int64_t j = 0; j < static_cast<int64_t>(input_i.size()); ++j) {
      if (j == concat_dim) {
        continue;
      }
      GE_ASSERT_EQ(out_dims[j], input_i[j]);
    }
    concat_value = concat_value + input_i[concat_dim];
  }
  out_dims[concat_dim] = concat_value;

  std::vector<Expression> out_sym_dims;
  for (auto dim : out_dims) {
    out_sym_dims.push_back(Symbol(dim));
  }
  out_tensor.MutableOriginSymbolShape().MutableDims() = out_sym_dims;
  return SUCCESS;
}

Status CalOutValue(const std::vector<std::vector<Expression>> &inputs_values,
                   const std::vector<std::vector<int64_t>> &inputs_dims, const int64_t concat_dim,
                   gert::SymbolTensor &out_tensor) {
  auto first_dims = inputs_dims[0];
  int64_t batch = 1L;
  for (int64_t i = 0; i < concat_dim; ++i) {
    batch *= first_dims[i];
  }
  std::vector<int64_t> steps;
  for (auto input_dims : inputs_dims) {
    int64_t step = 1L;
    for (int64_t i = concat_dim; i < static_cast<int64_t>(first_dims.size()); ++i) {
      step *= input_dims[i];
    }
    steps.push_back(step);
  }
  int64_t out_size = 0L;
  for (auto input_values : inputs_values) {
    out_size += inputs_values.size();
  }
  std::vector<Expression> out_dims;
  out_dims.reserve(out_size);
  for (int64_t i = 0; i < batch; ++i) {
    for (size_t j = 0; j < inputs_dims.size(); ++j) {
      int64_t step = steps[j];
      GE_ASSERT_TRUE(j < inputs_values.size(), "j = %u, inputs_values.size() = %d", j, inputs_values.size());
      auto input_values = inputs_values[j];
      auto begin_idx = i * step;
      auto end_idx = (i + 1) * step;
      if (input_values.size() < static_cast<size_t>(end_idx)) {
        return GRAPH_FAILED;
      }
      out_dims.insert(out_dims.end(), input_values.begin() + begin_idx, input_values.begin() + end_idx);
    }
  }
  out_tensor.SetSymbolicValue(MakeUnique<std::vector<Expression>>(out_dims));
  return GRAPH_SUCCESS;
}
}  // namespace

static graphStatus ConcatV2DSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_CHECK_NOTNULL(context);
  GELOGD("ConcatV2D Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());

  int64_t attr_N = context->GetComputeNodeInputNum();

  auto symbolic_tensor = context->GetOutputSymbolTensor(0);
  GE_ASSERT_NOTNULL(symbolic_tensor);
  // 如果只有一个输入则输出跟输入相同
  if (attr_N == 1) {
    return BuildOutWithOneInput(context, *symbolic_tensor);
  }

  int64_t concat_dim;
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto concat_dim_ptr = attrs->GetInt(0);
  GE_ASSERT_NOTNULL(concat_dim_ptr);
  concat_dim = *concat_dim_ptr;

  std::vector<std::vector<Expression>> inputs_values;
  std::vector<std::vector<int64_t>> inputs_dims;

  auto ret = GetInputsValueAndShape(context, attr_N, inputs_values, inputs_dims);
  if (ret != SUCCESS) {
    return ret;
  }
  GE_ASSERT_TRUE(inputs_dims.size() > 0, "inputs_dims size not match, is %zu", inputs_dims.size());
  std::vector<int64_t> first_shape = inputs_dims[0];
  if (first_shape.empty()) {
    GE_ASSERT_TRUE(concat_dim == 0 || concat_dim == -1, "concat_dim=%lld must equal 0 or -1", concat_dim);
  } else {
    if (concat_dim < 0) {
      concat_dim += first_shape.size();
    }
    GE_ASSERT_TRUE(concat_dim >= 0 && concat_dim < static_cast<int64_t>(first_shape.size()),
                   "concat_dim=%lld must be in [0, %d)", concat_dim, first_shape.size());
  }

  if (first_shape.empty()) {
    return BuildOutWithScalarInput(inputs_values, attr_N, *symbolic_tensor);
  }

  ret = CalOutShape(inputs_dims, concat_dim, *symbolic_tensor);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = CalOutValue(inputs_values, inputs_dims, concat_dim, *symbolic_tensor);
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*symbolic_tensor).c_str());
  return ret;
}

REGISTER_SYMBOLIC_KERNEL(ConcatV2D, ConcatV2DSymbolicKernelCompute);
}  // namespace ge
