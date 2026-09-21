/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/label_switch_task_def_faker.h"
namespace gert {
std::unique_ptr<TaskDefFaker> LabelSwitchTaskDefFaker::Clone() const {
  return std::make_unique<LabelSwitchTaskDefFaker>(*this);
}
std::vector<domi::TaskDef> LabelSwitchTaskDefFaker::CreateTaskDef(uint64_t op_index) {
  AddTask({kLabelSwitch, kKernelTypeEnd, 0});
  return TaskDefFaker::CreateTaskDef(op_index);
}
}  // namespace gert
