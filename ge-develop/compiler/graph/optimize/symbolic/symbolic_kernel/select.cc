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
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
const size_t kSelectInputNum = 3U;
const size_t kSelectOutputNum = 1U;
const std::vector<ge::Expression> *GetInputSymbolicValue(const gert::InferSymbolComputeContext *context, size_t index) {
  auto symbol_tensor = context->GetInputSymbolTensor(index);
  return symbol_tensor == nullptr ? nullptr : symbol_tensor->GetSymbolicValue();
}

bool GetShapeValue(const std::vector<Expression> &symbol_shape, std::vector<int64_t> &shape_value) {
  shape_value.reserve(symbol_shape.size());
  for (const auto &expr : symbol_shape) {
    int64_t value = 0;
    GE_ASSERT_TRUE(expr.GetConstValue(value));
    shape_value.emplace_back(value);
  }
  return true;
}

bool CalOutputValue(const std::vector<Expression> &condition, const std::vector<Expression> &x1,
                    const std::vector<Expression> &x2, std::vector<Expression> &output) {
  output.reserve(condition.size());
  for (size_t i = 0U; i < condition.size(); ++i) {
    int32_t cond = 0;
    GE_ASSERT_TRUE(condition[i].GetConstValue(cond));
    if (cond > 0) {
      output.emplace_back(x1[i]);
    } else {
      output.emplace_back(x2[i]);
    }
  }
  return true;
}

bool CheckInputsValue(const std::vector<ge::Expression> *condition_value, const std::vector<ge::Expression> *x1_value,
                      const std::vector<ge::Expression> *x2_value) {
  return condition_value != nullptr && !condition_value->empty() && x1_value != nullptr && !x1_value->empty() &&
         x2_value != nullptr && !x2_value->empty();
}

std::vector<int64_t> AlignShape(const std::vector<int64_t> &src, const std::vector<int64_t> &dst) {
  std::vector<int64_t> aligned = src;
  while (aligned.size() < dst.size()) {
    aligned.insert(aligned.begin(), 1);
  }
  return aligned;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

bool Broadcast(const std::vector<Expression> &src_data, const std::vector<int64_t> &src_shape, const std::vector<int64_t> &dst_shape,
               std::vector<Expression> &dst_data) {
  std::vector<int64_t> aligned_src = AlignShape(src_shape, dst_shape);
  if (aligned_src.size() != dst_shape.size()) {
    GELOGW("Can not broadcast, after aligned shape size is not equal, aligned_size: %zu , dst_size: %zu",
           aligned_src.size(), dst_shape.size());
    return false;
  }
  for (size_t i = 0U; i < dst_shape.size(); ++i) {
    if (aligned_src[i] != 1 && aligned_src[i] != dst_shape[i]) {
      GELOGW("Dim: %zu can not broadcast, src_value: %lld , dst_value: %lld", i, aligned_src[i], dst_shape[i]);
      return false;
    }
  }
  int64_t total_elements = 1;
  for (auto dim : dst_shape) {
    total_elements *= dim;
  }

  if (static_cast<int64_t>(src_data.size()) == total_elements) {
    dst_data = src_data;
    return true;
  }

  std::vector<int64_t> src_strides = ComputeStrides(aligned_src);
  std::vector<int64_t> dst_strides = ComputeStrides(dst_shape);
  GE_ASSERT_TRUE((aligned_src.size() == src_strides.size()) && (dst_shape.size() == dst_strides.size()));
  dst_data.reserve(total_elements);

  for (int64_t pos = 0; pos < total_elements; ++pos) {
    int64_t src_linear = 0;
    for (size_t i = 0U; i < dst_shape.size(); ++i) {
      int64_t dim = aligned_src[i];
      int64_t idx = (pos / dst_strides[i]) % dst_shape[i];
      src_linear += (idx % dim) * src_strides[i];
    }
    dst_data.push_back(src_data[src_linear]);
  }
  return true;
}
}  // namespace

static graphStatus SelectSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Select Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT_EQ(context->GetComputeNodeInputNum(), kSelectInputNum);
  GE_ASSERT_EQ(context->GetComputeNodeOutputNum(), kSelectOutputNum);

  const std::vector<ge::Expression> *condition_value = GetInputSymbolicValue(context, 0);
  const std::vector<ge::Expression> *x1_value = GetInputSymbolicValue(context, 1);
  const std::vector<ge::Expression> *x2_value = GetInputSymbolicValue(context, 2);
  if (!CheckInputsValue(condition_value, x2_value, x2_value)) {
    GELOGW("Select not support, inputs symbol value is empty.");
    return UNSUPPORTED;
  }

  auto condition_shape = context->GetInputSymbolShape(0U);
  auto x1_shape = context->GetInputSymbolShape(1U);
  auto x2_shape = context->GetInputSymbolShape(2U);
  GE_UNSUPPORTED_IF_NULL(condition_shape);
  GE_UNSUPPORTED_IF_NULL(x1_shape);
  GE_UNSUPPORTED_IF_NULL(x2_shape);
  if (*x1_shape != *x2_shape) {
    GELOGW("Select not support, check inputs shape failed, x1_shape: %s, x2_shape: %s.",
           SymbolicInferUtil::VectorExpressionToStr(x1_shape->GetDims()).c_str(),
           SymbolicInferUtil::VectorExpressionToStr(x2_shape->GetDims()).c_str());
    return UNSUPPORTED;
  }

  std::vector<int64_t> cond_shape_value;
  std::vector<int64_t> x1_shape_value;
  if (!(GetShapeValue(condition_shape->GetDims(), cond_shape_value)) ||
      !(GetShapeValue(x1_shape->GetDims(), x1_shape_value))) {
    GELOGW("Select not support, get condition_shape value: %s or x1_shape value: %s failed.",
           SymbolicInferUtil::VectorExpressionToStr(condition_shape->GetDims()).c_str(),
           SymbolicInferUtil::VectorExpressionToStr(x1_shape->GetDims()).c_str());
    return UNSUPPORTED;
  }
  std::vector<Expression> after_bc_condition;
  if (!Broadcast(*condition_value, cond_shape_value, x1_shape_value, after_bc_condition)) {
    GELOGW("Select not support, broadcast failed.");
    return UNSUPPORTED;
  }

  std::vector<Expression> output_value;
  if (!(CalOutputValue(after_bc_condition, *x1_value, *x2_value, output_value))) {
    GELOGW("Select not support, cal output value failed.");
    return UNSUPPORTED;
  }

  auto output_desc = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(output_desc);
  auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression>>(output_value);
  if (symbolic_value_unique != nullptr) {
    output_desc->SetSymbolicValue(std::move(symbolic_value_unique));
  }
  output_desc->MutableOriginSymbolShape().MutableDims() = x1_shape->GetDims();

  GELOGD("%s[%s] kernel success, output ori shape: %s, output value %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::VectorExpressionToStr(output_desc->GetOriginSymbolShape().GetDims()).c_str(),
         SymbolicInferUtil::VectorExpressionToStr(*output_desc->GetSymbolicValue()).c_str());

  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(Select, SelectSymbolicKernelCompute);
}  // namespace ge