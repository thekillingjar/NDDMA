/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "asc_overrides.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "graph/node.h"
#include "graph/symbolizer/symbolic.h"
#include "symbolizer/symbolic_utils.h"
#include "loop_common.h"

namespace ge {
namespace loop {
std::string BufferName(const Anchor *anchor) {
  auto *buffer = const_cast<Anchor *>(anchor);
  if (buffer == nullptr || buffer->GetOwnerNode() == nullptr) {
    return "nullptr";
  }
  std::string node_name = buffer->GetOwnerNode()->GetName();
  int32_t index = buffer->GetIdx();
  if (Anchor::DynamicAnchorPtrCast<OutDataAnchor>(buffer)) {
    return node_name + "_out" + std::to_string(index);
  }
  if (Anchor::DynamicAnchorPtrCast<InDataAnchor>(buffer)) {
    return node_name + "_in" + std::to_string(index);
  }
  if (Anchor::DynamicAnchorPtrCast<OutControlAnchor>(buffer)) {
    return node_name + "_out_ctrl";
  }
  if (Anchor::DynamicAnchorPtrCast<InControlAnchor>(buffer)) {
    return node_name + "_in_ctrl";
  }
  return node_name + "_unknown" + std::to_string(index);
}

std::string BufferName(const AnchorPtr &buffer) {
  return BufferName(buffer.get());
}

ConstGeTensorDescPtr GetBufferDesc(const OutDataAnchor *buffer) {
  GE_WARN_ASSERT(buffer != nullptr);
  GE_WARN_ASSERT(buffer->GetOwnerNode() != nullptr);
  GE_WARN_ASSERT(buffer->GetOwnerNode()->GetOpDesc() != nullptr);
  return buffer->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(buffer->GetIdx());
}

graphStatus GetBufferShape(const OutDataAnchor *buffer, std::vector<Expression> &dims) {
  const auto desc = GetBufferDesc(buffer);
  GE_WARN_ASSERT(desc != nullptr);
  const auto sym_attr = desc->GetAttrsGroup<SymbolicDescAttr>();
  if (sym_attr == nullptr) {
    GELOGW("GetAttrsGroup attr is null");
    return GRAPH_FAILED;
  }
  dims = sym_attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  return GRAPH_SUCCESS;
}

graphStatus GetBufferShape(const OutDataAnchorPtr &buffer, std::vector<Expression> &dims) {
  return GetBufferShape(buffer.get(), dims);
}

graphStatus GetBufferShape(const InDataAnchorPtr &buffer, std::vector<Expression> &dims) {
  GE_WARN_ASSERT(buffer != nullptr && buffer->GetPeerOutAnchor() != nullptr);
  return GetBufferShape(buffer->GetPeerOutAnchor(), dims);
}

graphStatus GetBufferDataType(const InDataAnchor *buffer, DataType &data_type) {
  GE_WARN_ASSERT(buffer != nullptr);
  auto out = buffer->GetPeerOutAnchor();
  GE_WARN_ASSERT(out != nullptr);
  GE_ASSERT_NOTNULL(out->GetOwnerNode());
  GE_ASSERT_NOTNULL(out->GetOwnerNode()->GetOpDesc());
  auto desc = out->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(out->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  data_type = desc->GetDataType();
  data_type = (data_type == DT_BOOL ? DT_UINT8 : data_type);
  return GRAPH_SUCCESS;
}

void GetStrideAndOffset(const Expression &expr, const std::vector<Expression> &symbols,
                        std::vector<Expression> &strides, Expression &offset) {
  const Expression simplified = expr.Simplify();
  std::vector<std::pair<Expression, Expression>> offset_stubs;
  offset_stubs.reserve(symbols.size());
  for (auto &symbol : symbols) {
    offset_stubs.emplace_back(symbol, Symbol(0));
  }
  offset = simplified.Subs(offset_stubs).Simplify();
  Expression expr_no_offset = (simplified - offset).Simplify();
  for (size_t i = 0U; i < symbols.size(); ++i) {
    std::vector<std::pair<Expression, Expression>> stubs;
    stubs.reserve(symbols.size());
    for (size_t j = 0U; j < symbols.size(); ++j) {
      stubs.emplace_back(symbols[j], j == i ? Symbol(1) : Symbol(0));
    }
    strides.emplace_back(expr_no_offset.Subs(stubs).Simplify());
  }
}

std::vector<Expression> ContiguousStrides(const std::vector<Expression> &dims) {
  if (dims.empty()) {
    return {};
  }
  std::vector<Expression> strides(dims.size(), Symbol(1));
  for (size_t i = dims.size() - 1U; i > 0U; --i) {
    strides[i - 1] = strides[i] * dims[i];
  }
  for (size_t i = 0U; i < dims.size(); ++i) {
    if (SymbolicUtils::StaticCheckEq(dims[i], Symbol(1)) == TriBool::kTrue ||
        SymbolicUtils::StaticCheckEq(dims[i], Symbol(0)) == TriBool::kTrue) {
      strides[i] = Symbol(0);  // 保证dim size为0或1时，stride必定为0，简化stride判定规则，
    }
    strides[i] = strides[i].Simplify();
  }
  return strides;
}
}  // namespace loop
}  // namespace ge