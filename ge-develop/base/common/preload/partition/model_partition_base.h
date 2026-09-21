/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_BASE_COMMON_PRELOAD_MODEL_PARTITION_BASE_H_
#define AIR_BASE_COMMON_PRELOAD_MODEL_PARTITION_BASE_H_
#include "framework/common/ge_types.h"
#include "ge/ge_api_error_codes.h"
#include "common/model/ge_model.h"
#include "common/preload/model/pre_model_partition_utils.h"

namespace ge {
class ModelPartitionBase {
 public:
  ModelPartitionBase() = default;
  virtual ~ModelPartitionBase() = default;
  virtual Status Init(const GeModelPtr &ge_model, const uint8_t type = 0U) = 0;
  virtual std::shared_ptr<uint8_t> Data();
  virtual uint32_t DataSize();

 protected:
  uint32_t buff_size_ = 0U;
  std::shared_ptr<uint8_t> buff_ = nullptr;
};
}  // namespace ge
#endif