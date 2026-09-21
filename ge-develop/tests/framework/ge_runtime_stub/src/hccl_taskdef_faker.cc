/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/hccl_taskdef_faker.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
vector<domi::TaskDef> HcclTaskDefFaker::CreateTaskDef(uint64_t op_index) {
  AddTask({kHccl, kHccl_Kernel, 0});
  auto task_def = TaskDefFaker::CreateTaskDef(op_index);
  task_def[0].set_stream_id(0);
  return task_def;
}

std::unique_ptr<TaskDefFaker> HcclTaskDefFaker::Clone()  const {
  return std::unique_ptr<HcclTaskDefFaker>(new HcclTaskDefFaker(*this));
}
}
