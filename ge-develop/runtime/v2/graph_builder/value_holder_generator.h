/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_GENERATOR_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_GENERATOR_H_
#include <vector>

#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "common/plugin/ge_make_unique_util.h"

namespace gert {
namespace bg {
std::vector<DevMemValueHolderPtr> CreateDevMemErrorValueHolders(int64_t stream_id, size_t count, const char *fmt, ...);
std::vector<ValueHolderPtr> CreateErrorValueHolders(size_t count, const char *fmt, ...);

template <typename T>
ValueHolderPtr CreateContVecHolder(const std::vector<T> &src_vec) {
  size_t total_size = 0U;
  size_t vec_size = src_vec.size();
  auto vec_holder = ContinuousVector::Create<T>(vec_size, total_size);
  if (vec_holder == nullptr) {
    return nullptr;
  }
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  vec->SetSize(vec_size);
  for (size_t i = 0U; i < vec_size; ++i) {
    *(reinterpret_cast<T *>(vec->MutableData()) + i) = src_vec[i];
  }
  return bg::ValueHolder::CreateConst(vec, total_size);
}

template <typename T>
ValueHolderPtr CreateConstVecHolder(const std::vector<std::vector<T>> &src_vec) {
  size_t total_size = ContinuousVectorVector::GetOverHeadLength(src_vec.size());
  for (const auto &inner_vec : src_vec) {
    size_t inner_vec_length = 0U;
    if (ge::MulOverflow(inner_vec.size(), sizeof(T), inner_vec_length)) {
      return nullptr;
    }
    if (ge::AddOverflow(inner_vec_length, sizeof(ContinuousVector), inner_vec_length)) {
      return nullptr;
    }
    if (ge::AddOverflow(total_size, inner_vec_length, total_size)) {
      return nullptr;
    }
  }
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  auto cvv = new (holder.get()) ContinuousVectorVector();
  if (cvv == nullptr) {
    return nullptr;
  }
  cvv->Init(src_vec.size());
  for (const auto &inner_vec : src_vec) {
    auto cv = cvv->Add<T>(inner_vec.size());
    if (cv == nullptr) {
      return nullptr;
    }
    if (!inner_vec.empty()) {
      const size_t copy_size = inner_vec.size() * sizeof(T);
      if (memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(T), inner_vec.data(), copy_size) != EOK) {
        return nullptr;
      }
    }
  }
  return bg::ValueHolder::CreateConst(cvv, total_size);
}

std::vector<bg::ValueHolderPtr> ConvertDevMemValueHoldersToValueHolders(
    const std::vector<bg::DevMemValueHolderPtr> &dev_mem_holders);
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_GENERATOR_H_
