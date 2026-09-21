/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "attribute_group/attr_group_symbolic_desc.h"
#include "exe_graph/runtime/storage_shape.h"
#include "common/util/mem_utils.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
const size_t kXInputIndex = 0;
const size_t kOutputIndex = 0;
const size_t kSqueezeOutputNum = 1;
const size_t kSqueezeInputNum = 1;
}  // namespace

static graphStatus CalSqueezeOutShape(const gert::SymbolShape &input_shape, const std::vector<int64_t> &axes,
                                      std::vector<Expression> &output_shape) {
  if (axes.empty()) {
    for (size_t i = 0; i < input_shape.GetDimNum(); i++) {
      if (EXPECT_SYMBOL_NE(input_shape.GetDim(i), Symbol(1))) {
        output_shape.push_back(input_shape.GetDim(i));
      }
    }
    return SUCCESS;
  }

  bool squeeze_dims[gert::Shape::kMaxDimNum] = {false};
  const auto dim_num = static_cast<int64_t>(input_shape.GetDimNum());
  for (auto raw_axis : axes) {
    const int64_t real_axis = raw_axis >= 0 ? raw_axis : raw_axis + dim_num;
    GE_ASSERT(real_axis >= 0 && real_axis < dim_num, "Squeeze failed, as axes val[%ld] is out of range[-%ld, %ld].",
              raw_axis, dim_num, dim_num);
    GE_ASSERT(real_axis < static_cast<int64_t>(gert::Shape::kMaxDimNum));
    squeeze_dims[real_axis] = true;
  }
  for (size_t i = 0; i < input_shape.GetDimNum(); i++) {
    if (!squeeze_dims[i]) {
      output_shape.push_back(input_shape.GetDim(i));
    } else {
      ASSERT_SYMBOL_EQ(input_shape.GetDim(i), Symbol(1));
    }
  }
  return SUCCESS;
}

static graphStatus SqueezeSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Squeeze Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT(context->GetComputeNodeInputNum() == kSqueezeInputNum, "InputNum=%zu", context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == kSqueezeOutputNum, "OutputNum=%zu",
            context->GetComputeNodeOutputNum());

  std::vector<int64_t> axis;
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto attr_index = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(attr_index);
  for (size_t i = 0; i < attr_index->GetSize(); i++) {
    axis.push_back(attr_index->GetData()[i]);
  }
  auto symbol_tensor = context->GetInputSymbolTensor(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(symbol_tensor);
  auto symbol_values = symbol_tensor->GetSymbolicValue();
  if (symbol_values == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get input symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto input_shape = context->GetInputSymbolTensor(kXInputIndex)->GetOriginSymbolShape();
  auto output_desc = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output_desc);
  std::vector<Expression> output_shape;
  auto ret = CalSqueezeOutShape(input_shape, axis, output_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression>>(*symbol_values);
  if (symbolic_value_unique != nullptr) {
    output_desc->SetSymbolicValue(std::move(symbolic_value_unique));
  }
  output_desc->MutableOriginSymbolShape().MutableDims() = output_shape;

  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output_desc).c_str());
  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(Squeeze, SqueezeSymbolicKernelCompute);
}  // namespace ge