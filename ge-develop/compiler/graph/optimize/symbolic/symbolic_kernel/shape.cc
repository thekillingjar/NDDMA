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
#include "graph_metadef/common/ge_common/util.h"
#include "framework/common/types.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
constexpr size_t kShapeInputOutputSize = 1U;
static graphStatus ShapeSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_CHECK_NOTNULL(context);
  GELOGD("Shape Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT(context->GetComputeNodeInputNum() == kShapeInputOutputSize, "InputNum=%zu",
            context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == kShapeInputOutputSize, "OutputNum=%zu",
            context->GetComputeNodeOutputNum());

  auto input_tensor = context->GetInputSymbolTensor(0U);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  const auto dims = input_tensor->GetOriginSymbolShape();
  auto symbolic_tensor = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(symbolic_tensor);
  auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression> >(dims.GetDims());
  if (symbolic_value_unique != nullptr) {
    symbolic_tensor->SetSymbolicValue(std::move(symbolic_value_unique));
  }
  symbolic_tensor->MutableOriginSymbolShape().MutableDims() = {ge::Symbol(dims.GetDimNum())};

  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*symbolic_tensor).c_str());
  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(Shape, ShapeSymbolicKernelCompute);
}  // namespace ge
