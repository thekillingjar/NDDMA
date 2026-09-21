/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_BASE_COMMON_PRELOAD_NANO_TASK_DESC_PARTITION_H_
#define AIR_BASE_COMMON_PRELOAD_NANO_TASK_DESC_PARTITION_H_
#include "common/preload/partition/model_partition_base.h"

namespace ge {
class NanoTaskDescPartition : public ModelPartitionBase {
 public:
  NanoTaskDescPartition() = default;
  virtual ~NanoTaskDescPartition() override = default;
  virtual Status Init(const GeModelPtr &ge_model, const uint8_t type) override;

 protected:
  uint8_t *desc_addr_ = nullptr;
};
}  // namespace ge
#endif