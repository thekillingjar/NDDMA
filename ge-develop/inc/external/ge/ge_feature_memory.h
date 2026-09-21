/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GE_FEATURE_MEMORY_H
#define INC_EXTERNAL_GE_GE_FEATURE_MEMORY_H

#include <memory>
#include <vector>
#include "ge/ge_api_types.h"
#include "ge_error_codes.h"

namespace ge {
class GE_FUNC_VISIBILITY FeatureMemory {
 public:
  class Builder;

  class FeatureMemoryData;

  ~FeatureMemory();

  FeatureMemory &operator=(const FeatureMemory &) & = delete;
  FeatureMemory(const FeatureMemory &) = delete;

  ///
  /// @brief get memory type
  /// @return memory type
  ///
  MemoryType GetType() const;

  ///
  /// @brief get memory size
  /// @return memory size
  ///
  size_t GetSize() const;

  ///
  /// @brief is fixed feature memory
  /// @return true if feature memory is fixed, otherwise false
  ///
  bool IsFixed() const;

 private:
  FeatureMemory() = default;
  std::shared_ptr<FeatureMemoryData> data_{nullptr};
};
using FeatureMemoryPtr = std::shared_ptr<FeatureMemory>;
} // namespace ge
#endif  // INC_EXTERNAL_GE_GE_FEATURE_MEMORY_H
