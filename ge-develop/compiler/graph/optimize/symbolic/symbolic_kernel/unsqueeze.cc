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
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
const size_t kXInputIndex = 0;
const size_t kOutputIndex = 0;
const size_t kUnsqueezeOutputNum = 1;
const size_t kUnsqueezeInputNum = 1;

graphStatus CalOutShape(const gert::SymbolShape &input_shape, const std::vector<int64_t> &axes,
                        std::vector<Expression> &output_shape) {
  int64_t output_dim_num = input_shape.GetDimNum() + axes.size();
  output_shape.reserve(output_dim_num);
  for (int64_t i = 0; i < output_dim_num; ++i) {
    output_shape.push_back(Symbol(0));
  }

  for (int64_t axis : axes) {
    axis = axis >= 0 ? axis : axis + static_cast<int64_t>(output_dim_num);
    GE_ASSERT(axis >= 0 && axis < output_dim_num, "Unsqueeze failed, as axes val[%zu] is out of range[-%zu, %zu].",
              axis, output_dim_num, output_dim_num);
    GE_ASSERT(output_shape[axis] != 1, "Unsqueeze failed, axis repeated");
    output_shape[axis] = Symbol(1);
  }
  size_t in_index = 0;
  for (int64_t i = 0; i < output_dim_num; i++) {
    if (output_shape[i] != 1 && in_index < input_shape.GetDimNum()) {
      output_shape[i] = input_shape.GetDim(in_index);
      in_index++;
    }
  }
  return SUCCESS;
}
}  // namespace

static graphStatus UnsqueezeSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Unsqueeze Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT(context->GetComputeNodeInputNum() == kUnsqueezeInputNum);
  GE_ASSERT(context->GetComputeNodeOutputNum() == kUnsqueezeOutputNum);

  std::vector<int64_t> axes;
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  GE_ASSERT_NOTNULL(attrs->GetListInt(0));
  for (size_t i = 0; i < attrs->GetListInt(0)->GetSize(); i++) {
    axes.push_back(attrs->GetListInt(0)->GetData()[i]);
  }

  auto symbol_tensor = context->GetInputSymbolTensor(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(symbol_tensor);
  if (symbol_tensor->GetSymbolicValue() == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get input symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto out_symbol_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(out_symbol_tensor);
  std::vector<Expression> output_shape;
  GE_ASSERT_SUCCESS(CalOutShape(symbol_tensor->GetOriginSymbolShape(), axes, output_shape));
  auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression> >(*symbol_tensor->GetSymbolicValue());
  if (symbolic_value_unique != nullptr) {
    out_symbol_tensor->SetSymbolicValue(std::move(symbolic_value_unique));
  }
  out_symbol_tensor->MutableOriginSymbolShape().MutableDims() = output_shape;
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*out_symbol_tensor).c_str());

  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(Unsqueeze, UnsqueezeSymbolicKernelCompute);
}  // namespace ge
