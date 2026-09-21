/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "external_weight_desc_impl.h"

#include "ge/ge_external_weight_desc.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"

namespace ge {
ExternalWeightDesc::~ExternalWeightDesc() = default;

AscendString ExternalWeightDesc::GetLocation() const {
  return data_->GetLocation();
}

size_t ExternalWeightDesc::GetSize() const {
  return data_->GetSize();
}

size_t ExternalWeightDesc::GetOffset() const {
  return data_->GetOffset();
}

AscendString ExternalWeightDesc::GetId() const {
  return data_->GetId();
}

AscendString ExternalWeightDesc::ExternalWeightDescData::GetLocation() const {
  return location_;
}

size_t ExternalWeightDesc::ExternalWeightDescData::GetSize() const {
  return size_;
}

size_t ExternalWeightDesc::ExternalWeightDescData::GetOffset() const {
  return offset_;
}

AscendString ExternalWeightDesc::ExternalWeightDescData::GetId() const {
  return id_;
}

void ExternalWeightDesc::ExternalWeightDescData::SetLocationSizeOffsetId(const AscendString &location, const size_t size,
                                                   const size_t offset, const AscendString &id) {
  location_ = location;
  size_ = size;
  offset_ = offset;
  id_ = id;
}

ExternalWeightDescPtr ExternalWeightDesc::Builder::Build(const AscendString &location, const size_t size,
                                                   const size_t offset, const AscendString &id) {
  ExternalWeightDescPtr external_weight_desc(new (std::nothrow) ExternalWeightDesc());
  GE_ASSERT_NOTNULL(external_weight_desc);

  external_weight_desc->data_ = MakeShared<ExternalWeightDesc::ExternalWeightDescData>();
  GE_ASSERT_NOTNULL(external_weight_desc->data_);
  external_weight_desc->data_->SetLocationSizeOffsetId(location, size, offset, id);
  return external_weight_desc;
}
} // namespace ge
