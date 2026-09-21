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
constexpr size_t kDimsInputIndex = 0UL;
constexpr size_t kValueInputIndex = 1UL;
constexpr size_t kOutputIndex = 0UL;
constexpr size_t kMaxSupportDim = 25UL;
constexpr int64_t kMaxSupportOutputSize = 1024L;

bool GetElementNum(const std::vector<ge::Expression> &dims_symbols, int64_t &element_num,
                   const gert::InferSymbolComputeContext *context) {
  for (const auto &dim_sym : dims_symbols) {
    int64_t dim_value = 0L;
    if (!dim_sym.GetConstValue(dim_value)) {
      GELOGW("SymbolicKernel compute unsupported, reason: get dim sym const value failed, node %s[%s].",
             context->GetNodeName(), context->GetNodeType());
      return false;
    }
    if (ge::MulOverflow(element_num, dim_value, element_num)) {
      GELOGW("SymbolicKernel compute unsupported, reason: output element num over flow, node %s[%s].",
             context->GetNodeName(), context->GetNodeType());
      return false;
    }
  }

  if (element_num > kMaxSupportOutputSize) {
    GELOGW("SymbolicKernel compute unsupported, reason: output element num[%lld] is over limit [%lld], node %s[%s].",
           element_num, kMaxSupportOutputSize, context->GetNodeName(), context->GetNodeType());
    return false;
  }
  GELOGD("Node: %s value element: %lld", context->GetNodeName(), element_num);
  return true;
}
}  // namespace

static graphStatus FillSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Fill Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());

  // 获取dim的值
  const auto dim_input_shape = context->GetInputSymbolShape(kDimsInputIndex);
  GE_UNSUPPORTED_IF_NULL(dim_input_shape);
  if (dim_input_shape->GetDimNum() != 1) {
    GELOGW("SymbolicKernel compute unsupported, reason: dim num %zu not equal to 1, node %s[%s].",
           dim_input_shape->GetDimNum(), context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  auto dims_input_tensor = context->GetInputSymbolTensor(kDimsInputIndex);
  GE_UNSUPPORTED_IF_NULL(dims_input_tensor);
  auto dims_symbols = dims_input_tensor->GetSymbolicValue();
  if (dims_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: no symbolic value, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  if (dims_symbols->size() > kMaxSupportDim) {
    GELOGW("SymbolicKernel compute unsupported, reason: dim size %zu over limit %zu, node %s[%s].",
           dims_symbols->size(), kMaxSupportDim, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  auto symbolic_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(symbolic_tensor);
  symbolic_tensor->MutableOriginSymbolShape().MutableDims() = *dims_symbols;
  // 获取value个数
  int64_t element_num = 1L;
  if (!GetElementNum(*dims_symbols, element_num, context)) {
    return UNSUPPORTED;
  }

  // 获取value的值
  auto value_input_tensor = context->GetInputSymbolTensor(kValueInputIndex);
  GE_UNSUPPORTED_IF_NULL(value_input_tensor);
  auto value_symbols = value_input_tensor->GetSymbolicValue();
  if (value_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get %zu symbolic value failed, node %s[%s].", kValueInputIndex,
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  GE_ASSERT_TRUE(value_symbols->size() == 1UL,
                 "Value of input: %zu of node: %s should only have one element, but get: %zu", kValueInputIndex,
                 context->GetNodeName(), value_symbols->size());
  auto symbol_value = std::vector<Expression>(element_num, value_symbols->front());
  auto symbol_value_unique = ge::MakeUnique<std::vector<ge::Expression> >(symbol_value);
  if (symbol_value_unique != nullptr) {
    symbolic_tensor->SetSymbolicValue(std::move(symbol_value_unique));
  }
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*symbolic_tensor).c_str());
  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(Fill, FillSymbolicKernelCompute);
}  // namespace ge
