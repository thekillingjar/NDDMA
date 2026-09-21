/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "feature_memory_impl.h"

#include "ge/ge_feature_memory.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"

namespace ge {
FeatureMemory::~FeatureMemory() = default;

MemoryType FeatureMemory::GetType() const {
  return data_->GetType();
}

size_t FeatureMemory::GetSize() const {
  return data_->GetSize();
}

bool FeatureMemory::IsFixed() const {
  return data_->IsFixed();
}

MemoryType FeatureMemory::FeatureMemoryData::GetType() const {
  return type_;
}

size_t FeatureMemory::FeatureMemoryData::GetSize() const {
  return size_;
}

bool FeatureMemory::FeatureMemoryData::IsFixed() const {
  return is_fixed_;
}

void FeatureMemory::FeatureMemoryData::SetTypeSize(const MemoryType type, const size_t size,
                                                   const FeatureMemoryData::MemoryAttr &mem_attr) {
  type_ = type;
  size_ = size;
  is_fixed_ = mem_attr.is_fixed;
}

FeatureMemoryPtr FeatureMemory::Builder::Build(const MemoryType type, const size_t size,
                                               const FeatureMemoryData::MemoryAttr &mem_attr) {
  FeatureMemoryPtr feature_memory(new (std::nothrow) FeatureMemory());
  GE_ASSERT_NOTNULL(feature_memory);

  feature_memory->data_ = MakeShared<FeatureMemory::FeatureMemoryData>();
  GE_ASSERT_NOTNULL(feature_memory->data_);
  feature_memory->data_->SetTypeSize(type, size, mem_attr);
  return feature_memory;
}
} // namespace ge
