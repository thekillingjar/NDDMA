/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/arg_desc_info.h"
#include "arg_desc_info_impl.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/checker.h"

namespace ge {
ArgDescInfo::~ArgDescInfo() {}
ArgDescInfo::ArgDescInfo(ArgDescType arg_type, int32_t ir_index, bool is_folded) {
  impl_ = ComGraphMakeUnique<ArgDescInfoImpl>(arg_type, ir_index, is_folded);
}

ArgDescInfo::ArgDescInfo(ArgDescInfoImplPtr &&impl) : impl_(std::move(impl)) {}

ArgDescInfo::ArgDescInfo(const ArgDescInfo &other) {
  impl_ = ComGraphMakeUnique<ArgDescInfoImpl>();
  if ((other.impl_ != nullptr) && (impl_ != nullptr)) {
    *impl_ = *other.impl_;
  }
}
ArgDescInfo::ArgDescInfo(ArgDescInfo &&other) noexcept {
  impl_ = std::move(other.impl_);
}
ArgDescInfo &ArgDescInfo::operator=(const ArgDescInfo &other) {
  if (&other != this) {
    impl_ = ComGraphMakeUnique<ArgDescInfoImpl>();
    if ((other.impl_ != nullptr) && (impl_ != nullptr)) {
      *impl_ = *other.impl_;
    }
  }
  return *this;
}
ArgDescInfo &ArgDescInfo::operator=(ArgDescInfo &&other) noexcept {
  if (&other != this) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}
ArgDescInfo ArgDescInfo::CreateCustomValue(uint64_t custom_value) {
  return ArgDescInfo(ArgDescInfoImpl::CreateCustomValue(custom_value));
}
ArgDescInfo ArgDescInfo::CreateHiddenInput(HiddenInputSubType hidden_type) {
  return ArgDescInfo(ArgDescInfoImpl::CreateHiddenInput(hidden_type));
}
ArgDescType ArgDescInfo::GetType() const {
  if (impl_ != nullptr) {
    return impl_->GetType();
  }
  return ArgDescType::kEnd;
}
uint64_t ArgDescInfo::GetCustomValue() const {
  if (impl_ != nullptr) {
    return impl_->GetCustomValue();
  }
  return std::numeric_limits<uint64_t>::max();
}
graphStatus ArgDescInfo::SetCustomValue(uint64_t custom_value) {
  GE_ASSERT_NOTNULL(impl_);
  return impl_->SetCustomValue(custom_value);
}
HiddenInputSubType ArgDescInfo::GetHiddenInputSubType() const {
  if (impl_ != nullptr) {
    return impl_->GetHiddenInputSubType();
  }
  return HiddenInputSubType::kEnd;
}
graphStatus ArgDescInfo::SetHiddenInputSubType(HiddenInputSubType hidden_type) {
  GE_ASSERT_NOTNULL(impl_);
  return impl_->SetHiddenInputSubType(hidden_type);
}

int32_t ArgDescInfo::GetIrIndex() const {
  if (impl_ != nullptr) {
    return impl_->GetIrIndex();
  }
  return -1;
}

void ArgDescInfo::SetIrIndex(int32_t ir_index) {
  if (impl_ != nullptr) {
    impl_->SetIrIndex(ir_index);
  }
}

bool ArgDescInfo::IsFolded() const {
  if (impl_ != nullptr) {
    return impl_->IsFolded();
  }
  return false;
}
void ArgDescInfo::SetFolded(bool is_folded) {
  if (impl_ != nullptr) {
    impl_->SetFolded(is_folded);
  }
}
}