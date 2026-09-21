/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/event_record_task_def_faker.h"

namespace gert {
std::vector<domi::TaskDef> EventRecordTaskDefFaker::CreateTaskDef(uint64_t op_index) {
  AddTask({kEvent, kKernelTypeEnd, 0UL});
  return TaskDefFaker::CreateTaskDef(std::numeric_limits<uint64_t>::max());
}
std::unique_ptr<TaskDefFaker> EventRecordTaskDefFaker::Clone() const {
  return std::make_unique<EventRecordTaskDefFaker>(*this);
}
}  // namespace gert
