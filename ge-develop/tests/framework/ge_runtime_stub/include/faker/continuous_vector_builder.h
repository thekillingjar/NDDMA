/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_CONTINUOUS_VECTOR_BUILDER_H
#define AIR_CXX_CONTINUOUS_VECTOR_BUILDER_H
#include "exe_graph/runtime/continuous_vector.h"
namespace gert {
class ContinuousVectorBuilder {
 public:
  template <typename T>
  static std::unique_ptr<uint8_t[]> Create(std::initializer_list<T> value) {
    auto holder = ContinuousVector::Create<T>(value.size());
    auto vec = reinterpret_cast<TypedContinuousVector<T> *>(holder.get());
    vec->SetSize(value.size());
    size_t i = 0U;
    for (auto &v : value) {
      vec->MutableData()[i++] = std::move(v);
    }
    return holder;
  }
};
}
#endif  // AIR_CXX_CONTINUOUS_VECTOR_BUILDER_H
