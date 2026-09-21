/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_MEMORY__EXTERNAL_WEIGHT_DESC_IMPL_H
#define GE_COMMON_MEMORY__EXTERNAL_WEIGHT_DESC_IMPL_H

#include "ge/ge_external_weight_desc.h"
#include "graph/ascend_string.h"

namespace ge {
class ExternalWeightDesc::ExternalWeightDescData {
 public:

  ExternalWeightDescData() = default;
  ~ExternalWeightDescData() = default;

  AscendString GetLocation() const;

  size_t GetSize() const;

  size_t GetOffset() const;

  AscendString GetId() const;

  void SetLocationSizeOffsetId(const AscendString &location, const size_t size, const size_t offset, const AscendString &id);

 private:
  AscendString location_;
  size_t size_;
  size_t offset_;
  AscendString id_;
};

class ExternalWeightDesc::Builder {
 public:
  Builder() = default;
  ~Builder() = default;
  static ExternalWeightDescPtr Build(const AscendString &location, const size_t size, const size_t offset, const AscendString &id);
};
} // namespace ge
#endif  // GE_COMMON_MEMORY__EXTERNAL_WEIGHT_DESC_IMPL_H
