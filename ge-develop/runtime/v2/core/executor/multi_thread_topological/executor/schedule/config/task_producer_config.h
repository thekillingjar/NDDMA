/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_PRODUCER_CONFIG_H
#define AIR_CXX_RUNTIME_V2_TASK_PRODUCER_CONFIG_H

#include "default_config.h"

namespace gert {
struct TaskProducerConfig {
  TaskProducerType type{kTaskProducerTypeDefault};
  size_t prepared_task_count{kTaskProducerPreparedTaskDefault};
  size_t thread_num{0U};
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_TASK_PRODUCER_CONFIG_H