/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "task_schedule_data.h"
#include "core/execution_data.h"
namespace gert {
TaskScheduleData::TaskScheduleData(const void *exec_data) : execution_data(exec_data), schedule_limit(0) {
  if (execution_data != nullptr) {
    schedule_limit = reinterpret_cast<const ExecutionData *>(execution_data)->base_ed.node_num;
  }
}
}  // namespace gert