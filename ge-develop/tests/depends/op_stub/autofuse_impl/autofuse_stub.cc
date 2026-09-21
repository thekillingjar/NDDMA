/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autofuse_stub.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace gert;
using namespace ge;

class TilingSymbolEvalContext : public TilingContext {
 public:
  const gert::Tensor *GetInputTensor(size_t data_index) const {
    auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
    if (tensor == nullptr) {
      return nullptr;
    }
    return tensor;
  }
};

class InferShapeSymbolEvalContext : public InferShapeContext {
 public:
  const gert::Tensor *GetInputTensor(size_t data_index) const {
    auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
    if (tensor == nullptr) {
      return nullptr;
    }
    return tensor;
  }
};

class SymbolTilingParseContext : public KernelContext {
 public:
  fe::PlatFormInfos *GetPlatFormInfos() {
    return GetInputValue<fe::PlatFormInfos *>(0);
  }
};

uint32_t TilingFunc(TilingSymbolEvalContext *context) {
  auto kernel_context = reinterpret_cast<KernelContext *>(context);
  auto tiling_data_ptr = kernel_context->GetOutputPointer<TilingData *>(TilingContext::kOutputTilingData);
  int64_t data1 = 1L;
  (*tiling_data_ptr)->Append(data1);
  int64_t data2 = 2L;
  (*tiling_data_ptr)->Append(data2);
  int64_t data3 = 3L;
  (*tiling_data_ptr)->Append(data3);

  auto input_data_num =  kernel_context->GetInputValue<size_t>(0);
  auto tiling_parser = kernel_context->GetInputValue<AfTilingParseData *>(input_data_num + 1);
  auto block_dim = tiling_parser->aiv_num;

  context->SetBlockDim(block_dim);
  *context->GetWorkspaceSizes(1) = 1024;
  return 0;
}

size_t GetTilingDataSize() { return 128; }

graphStatus InferShape(InferShapeSymbolEvalContext *context) {
  auto s0 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  auto s2 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  auto s4 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(2);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  auto s5 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(2);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  auto s6 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(3);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  auto s7 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(3);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();

  context->GetOutputShape(0)->SetDimNum(0);
  context->GetOutputShape(0)->AppendDim(s4 + s7);
  context->GetOutputShape(0)->AppendDim(s5 * s6);
  context->GetOutputShape(0)->AppendDim(s0 + s2);
  std::cout << "InferShape success from stub" << std::endl;
  return GRAPH_SUCCESS;
}

graphStatus GetSymbolTilingCacheKey(TilingSymbolEvalContext *context) {
  auto kernel_context = reinterpret_cast<KernelContext *>(context);
  auto symbol_source_vector = kernel_context->GetOutputPointer<TypedContinuousVector<int64_t>>(0U);

  auto s0 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  auto s1 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  auto s2 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();

  symbol_source_vector->MutableData()[0] = s0;
  symbol_source_vector->MutableData()[1] = s1;
  symbol_source_vector->MutableData()[2] = s2;

  symbol_source_vector->SetSize(3);

  return GRAPH_SUCCESS;
}

graphStatus TilingParse(SymbolTilingParseContext *context) {
  auto *platform = context->GetPlatFormInfos();

  auto kernel_context = reinterpret_cast<KernelContext *>(context);
  auto tiling_parse_av = kernel_context->GetOutput(0);
  auto tiling_parse_data_ptr = new (std::nothrow) uint8_t[sizeof(AfTilingParseData)];
  tiling_parse_av->SetWithDefaultDeleter<uint8_t[]>(tiling_parse_data_ptr);

  auto tiling_parse_data = kernel_context->GetOutputPointer<AfTilingParseData *>(0);
  (*tiling_parse_data)->aiv_num = platform->GetCoreNum();
  (*tiling_parse_data)->ub_size = (184 * 1024);

  return GRAPH_SUCCESS;
}

ge::graphStatus DfxInputSymbolInfo(TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return GRAPH_SUCCESS;
  }
  std::string symbol_info;
  auto s0 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  symbol_info += ("s0: " + std::to_string(s0));

  auto s1 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  symbol_info += (", s1: " + std::to_string(s1));

  auto s2 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  symbol_info += (", s2: " + std::to_string(s2));


  if (symbol_info.empty()) {
    out_symbol_info[0] = '\0';
    return GRAPH_SUCCESS;
  }
  symbol_info += ".";
  if (strncpy_s(out_symbol_info, size, symbol_info.c_str(), std::min(symbol_info.size(), size - 1)) != 0) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
