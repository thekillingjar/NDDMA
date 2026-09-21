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
#include "framework/common/types.h"
#include "common/util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kXInputIndex = 0UL;
constexpr size_t kMultiplesInputIndex = 1UL;
constexpr size_t kOutputIndex = 0UL;
constexpr size_t kMaxSupportDim = 8UL;
constexpr int64_t kMaxOutputSize = 1024L;

graphStatus GetMultiplesDims(const gert::InferSymbolComputeContext *context, std::vector<int64_t> &multiples_dims) {
  std::vector<int64_t> input_dims;
  context->GetConstInputDims(kMultiplesInputIndex, input_dims);
  if (input_dims.size() != 1) {
    GELOGW("SymbolicKernel compute unsupported, reason: %zu input dims size (%zu) not equal 1, node %s[%s].",
           kMultiplesInputIndex, input_dims.size(), context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  auto input_tensor = context->GetInputSymbolTensor(kMultiplesInputIndex);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  auto multiples_symbols = input_tensor->GetSymbolicValue();
  if (multiples_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get %zu input symbolic value failed, node %s[%s].",
           kMultiplesInputIndex, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  if (multiples_symbols->size() > kMaxSupportDim) {
    GELOGW("SymbolicKernel compute unsupported, reason: multiples_symbols size(%zu) over limit(%zu), node %s[%s].",
           multiples_symbols->size(), kMaxSupportDim, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  multiples_dims.clear();
  for (size_t i = 0UL; i < multiples_symbols->size(); i++) {
    int64_t dim = 0L;
    if (!(*multiples_symbols)[i].GetConstValue(dim)) {
      GELOGW("SymbolicKernel compute unsupported, reason: get %zu input const value failed, node %s[%s].", i,
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    multiples_dims.emplace_back(dim);
  }
  GELOGD("%s[%s] kernel compute, GetMultiplesDims %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::VectorToStr(multiples_dims).c_str());
  return SUCCESS;
}

void AlignDimNum(std::vector<int64_t> &x_dims, std::vector<int64_t> &y_dims) {
  std::reverse(x_dims.begin(), x_dims.end());
  std::reverse(y_dims.begin(), y_dims.end());
  // 获取维度较小的dims
  std::vector<int64_t> &dims_with_less_size = x_dims.size() < y_dims.size() ? x_dims : y_dims;
  // 给维度较小的dims补维
  dims_with_less_size.insert(dims_with_less_size.end(),
                             abs(static_cast<int64_t>(x_dims.size()) - static_cast<int64_t>(y_dims.size())), 1);
  std::reverse(x_dims.begin(), x_dims.end());
  std::reverse(y_dims.begin(), y_dims.end());
}

Status TileInputXSymbols(const std::vector<int64_t> &x_dims, const std::vector<int64_t> &multiples_dims,
                         const std::vector<Expression> &x_symbols, std::vector<Expression> &output_symbols) {
  GE_ASSERT_TRUE((!x_dims.empty()) && (!multiples_dims.empty()));
  int64_t dim_multis = 1L;
  std::list<Expression> outputs_symbols_list(x_symbols.begin(), x_symbols.end());
  for (int64_t dim_index = static_cast<int64_t>(x_dims.size()) - 1; dim_index >= 0; dim_index--) {
    int64_t block_size = dim_multis * x_dims[static_cast<size_t>(dim_index)];
    auto insert_pos = outputs_symbols_list.begin();
    GE_ASSERT_TRUE(block_size > 0);
    size_t block_num = outputs_symbols_list.size() / static_cast<size_t>(block_size);
    size_t cur_index = 0UL;
    while (cur_index < block_num) {
      for (size_t tile_index = 0UL;
           tile_index < static_cast<size_t>(multiples_dims[static_cast<size_t>(dim_index)] - 1); tile_index++) {
        outputs_symbols_list.insert(insert_pos, insert_pos, std::next(insert_pos, block_size));
      }
      insert_pos = std::next(insert_pos, block_size);
      cur_index++;
    }
    dim_multis = block_size * multiples_dims[static_cast<size_t>(dim_index)];
  }
  output_symbols.assign(outputs_symbols_list.begin(), outputs_symbols_list.end());
  return SUCCESS;
}

Status CalcOutputShapeSymbols(const std::vector<int64_t> &x_dims, const std::vector<int64_t> &multiples_dims,
                              std::vector<Expression> &output_shape_symbols) {
  GE_ASSERT_TRUE(x_dims.size() == multiples_dims.size());
  for (size_t i = 0UL; i < x_dims.size(); i++) {
    output_shape_symbols.emplace_back(Symbol(x_dims[i] * multiples_dims[i]));
  }
  return SUCCESS;
}

bool CheckOutputSize(const std::vector<int64_t> &x_dims, const std::vector<int64_t> &multiples_dims) {
  int64_t output_size = 1L;
  for (size_t i = 0U; i < x_dims.size(); i++) {
    if (ge::MulOverflow(output_size, x_dims[i] * multiples_dims[i], output_size)) {
      return false;
    }
  }
  return output_size <= kMaxOutputSize;
}
}  // namespace

static graphStatus TileSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Tile Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  // 获取multiples
  std::vector<int64_t> multiples_dims;
  auto ret = GetMultiplesDims(context, multiples_dims);
  GE_ASSERT_TRUE(ret == UNSUPPORTED || ret == SUCCESS);
  if (ret == UNSUPPORTED) {
    GELOGW("SymbolicKernel compute unsupported, reason: get const input failed, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  // 获取x的SymShape,shape必须是静态shape
  std::vector<int64_t> x_dims;
  if (!context->GetConstInputDims(kXInputIndex, x_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get const input dim failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  // 对齐multiples_symbols和x_dims的维度
  AlignDimNum(x_dims, multiples_dims);
  if (!CheckOutputSize(x_dims, multiples_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: check output size failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  GELOGI("Input x shape dims: %s, multiples dims: %s.", SymbolicInferUtil::VectorToStr(x_dims).c_str(),
         SymbolicInferUtil::VectorToStr(multiples_dims).c_str());
  // 获取value的值
  auto x_symbols = context->GetInputSymbolTensor(kXInputIndex)->GetSymbolicValue();
  if (x_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get input symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  // 扩展x_symbols
  auto output_symbols = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output_symbols);
  auto output_symbols_value = output_symbols->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_symbols_value);
  GE_ASSERT_SUCCESS(TileInputXSymbols(x_dims, multiples_dims, *x_symbols, *output_symbols_value));
  // 计算输出shape
  std::vector<Expression> output_shape_symbols;
  GE_ASSERT_SUCCESS(CalcOutputShapeSymbols(x_dims, multiples_dims, output_shape_symbols));
  output_symbols->MutableOriginSymbolShape().MutableDims() = output_shape_symbols;

  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output_symbols).c_str());
  return SUCCESS;
}
REGISTER_SYMBOLIC_KERNEL(Tile, TileSymbolicKernelCompute);
}  // namespace ge
