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
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
const size_t kXInputIndex = 0;
const size_t kAxisInputIndex = 1;
const size_t kOutputIndex = 0;
const size_t kExpandDimsOutputNum = 1;
const size_t kExpandDimsInputNum = 2;

graphStatus BuildOutTensor(const std::vector<Expression> &input_symbol_value, const gert::SymbolShape &input_shape,
                           int64_t axis, gert::SymbolTensor &out_tensor) {
  std::vector<Expression> out_dims;
  for (int64_t i = 0; i < axis; ++i) {
    out_dims.push_back(input_shape.GetDim(i));
  }
  out_dims.push_back(Symbol(1));
  for (size_t i = axis; i < input_shape.GetDimNum(); ++i) {
    out_dims.push_back(input_shape.GetDim(i));
  }
  auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression> >(input_symbol_value);
  if (symbolic_value_unique != nullptr) {
    out_tensor.SetSymbolicValue(std::move(symbolic_value_unique));
  }
  out_tensor.MutableOriginSymbolShape().MutableDims() = out_dims;
  return SUCCESS;
}
}  // namespace

static graphStatus ExpandDimsSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("ExpandDims Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT(context->GetComputeNodeInputNum() == kExpandDimsInputNum);
  GE_ASSERT(context->GetComputeNodeOutputNum() == kExpandDimsOutputNum);

  auto input_symbol_tensor = context->GetInputSymbolTensor(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(input_symbol_tensor);
  const std::vector<Expression> *input_symbol_value = input_symbol_tensor->GetSymbolicValue();
  if (input_symbol_value == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get input symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  // 获取并校验axis的值
  int64_t axis_value;
  auto axis_symbol_tensor = context->GetInputSymbolTensor(kAxisInputIndex);
  GE_UNSUPPORTED_IF_NULL(axis_symbol_tensor);
  auto axis_symbols = axis_symbol_tensor->GetSymbolicValue();
  if (axis_symbols == nullptr || !axis_symbols->begin()->GetConstValue(axis_value)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get Axis input symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  GELOGD("axis_symbols size = %zu", axis_symbols->size());
  GE_ASSERT_EQ(axis_symbols->size(), static_cast<size_t>(1));

  auto input_shape = context->GetInputSymbolShape(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(input_shape);
  // 如果axis为负数修正axis
  auto input_shape_num = static_cast<int64_t>(input_shape->GetDimNum());
  if (axis_value < 0) {
    if (input_shape_num == 0) {
      // input是一个标量，如果axis为负只能为-1，修正后为0
      GE_ASSERT_EQ(axis_value, -1);
      axis_value = 0;
    } else {
      axis_value += input_shape_num;
    }
  }
  GE_ASSERT_TRUE(axis_value >= 0 && axis_value <= input_shape_num, "axis_value=%lld, input_shape_num=%lld", axis_value,
                 input_shape_num);

  auto symbolic_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(symbolic_tensor);
  auto ret = BuildOutTensor(*input_symbol_value, *input_shape, axis_value, *symbolic_tensor);
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*symbolic_tensor).c_str());
  return ret;
}

REGISTER_SYMBOLIC_KERNEL(ExpandDims, ExpandDimsSymbolicKernelCompute);
}  // namespace ge
