/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "es_tensor_like.h"
#include <memory>
#include "es_c_tensor_holder.h"
#include "esb_funcs.h"

namespace ge {
namespace es {
class EsTensorLike::EsTensorLikeImpl {
public:
  enum class Tag {
    KTensor,
    KInt64,
    KFloat,
    KVecInt64,
    KVecFloat
  };

  Tag tag_;
  EsTensorHolder tensor_{};
  union {
    int64_t i64;
    float f;
  } v_;
  std::vector<int64_t> vec_i64_;
  std::vector<float> vec_float_;

  explicit EsTensorLikeImpl(const EsTensorHolder &tensor) : tag_(Tag::KTensor), tensor_(tensor) {}
  explicit EsTensorLikeImpl(const std::nullptr_t) : tag_(Tag::KTensor), tensor_(EsTensorHolder(nullptr)) {}
  explicit EsTensorLikeImpl(const int64_t value) : tag_(Tag::KInt64) { v_.i64 = value; }
  explicit EsTensorLikeImpl(const float value) : tag_(Tag::KFloat) { v_.f = value; }
  explicit EsTensorLikeImpl(const std::vector<int64_t> &values) : tag_(Tag::KVecInt64), vec_i64_(values) {}
  explicit EsTensorLikeImpl(const std::vector<float> &values) : tag_(Tag::KVecFloat), vec_float_(values) {}

  EsCGraphBuilder *GetOwnerBuilder() const {
    if (tag_ != Tag::KTensor) {
      return nullptr;
    }
    auto esb_tensor = tensor_.GetCTensorHolder();
    if (esb_tensor == nullptr) {
      return nullptr;
    }
    return &esb_tensor->GetOwnerBuilder();
  }

  EsTensorHolder ToTensorHolder(EsCGraphBuilder *graph) const {
    switch (tag_) {
      case Tag::KTensor:
        return tensor_;
      case Tag::KInt64:
        return EsCreateScalarInt64(graph, v_.i64);
      case Tag::KFloat:
        return EsCreateScalarFloat(graph, v_.f);
      case Tag::KVecInt64:
        return EsCreateVectorInt64(graph, vec_i64_.data(), static_cast<int64_t>(vec_i64_.size()));
      case Tag::KVecFloat:
        return EsCreateVectorFloat(graph, vec_float_.data(), static_cast<int64_t>(vec_float_.size()));
      default:
        return nullptr;
    }
  }
};

EsTensorLike::EsTensorLike(const EsTensorHolder &tensor)
    : impl_(std::make_unique<EsTensorLikeImpl>(tensor)) {}

EsTensorLike::EsTensorLike(const std::nullptr_t)
    : impl_(std::make_unique<EsTensorLikeImpl>(nullptr)) {}

EsTensorLike::EsTensorLike(const int64_t value)
    : impl_(std::make_unique<EsTensorLikeImpl>(value)) {}

EsTensorLike::EsTensorLike(const float value)
    : impl_(std::make_unique<EsTensorLikeImpl>(value)) {}

EsTensorLike::EsTensorLike(const std::vector<int64_t> &values)
    : impl_(std::make_unique<EsTensorLikeImpl>(values)) {}

EsTensorLike::EsTensorLike(const std::vector<float> &values)
    : impl_(std::make_unique<EsTensorLikeImpl>(values)) {}

EsTensorLike::~EsTensorLike() = default;

EsCGraphBuilder *EsTensorLike::GetOwnerBuilder() const {
  return impl_->GetOwnerBuilder();
}

EsTensorHolder EsTensorLike::ToTensorHolder(EsCGraphBuilder *graph) const {
  return impl_->ToTensorHolder(graph);
}
}  // namespace es
}  // namespace ge
