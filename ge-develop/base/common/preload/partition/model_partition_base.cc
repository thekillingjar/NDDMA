/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/partition/model_partition_base.h"

namespace ge {
uint32_t ModelPartitionBase::DataSize() {
  GELOGD("ModelPartitionBase buff_size_:%u", buff_size_);
  return buff_size_;
}

std::shared_ptr<uint8_t> ModelPartitionBase::Data() {
  GE_ASSERT_NOTNULL(buff_, "buff is nullptr, size[%u]", buff_size_);
  return buff_;
}
}
