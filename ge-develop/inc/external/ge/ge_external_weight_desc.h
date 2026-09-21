/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GE_EXTERNAL_WEIGHT_DESC_H
#define INC_EXTERNAL_GE_GE_EXTERNAL_WEIGHT_DESC_H

#include <memory>
#include <vector>
#include "ge/ge_api_types.h"
#include "ge_error_codes.h"

namespace ge {
class GE_FUNC_VISIBILITY ExternalWeightDesc {
 public:
  class Builder;

  class ExternalWeightDescData;

  ~ExternalWeightDesc();

  ExternalWeightDesc &operator=(const ExternalWeightDesc &) & = delete;
  ExternalWeightDesc(const ExternalWeightDesc &) = delete;

  AscendString GetLocation() const;

  size_t GetSize() const;

  size_t GetOffset() const;

  AscendString GetId() const;

 private:
  ExternalWeightDesc() = default;
  std::shared_ptr<ExternalWeightDescData> data_{nullptr};
};
using ExternalWeightDescPtr = std::shared_ptr<ExternalWeightDesc>;
} // namespace ge
#endif  // INC_EXTERNAL_GE_GE_EXTERNAL_WEIGHT_DESC_H
