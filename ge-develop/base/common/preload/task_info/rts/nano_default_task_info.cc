/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/task_info/rts/nano_default_task_info.h"
#include "common/preload/model/pre_model_utils.h"
#include "common/preload/model/pre_model_types.h"
#include "common/preload/task_info/pre_task_status.h"

namespace ge {
PreTaskResult GenerateDefaultTask(const domi::TaskDef &task_def, const OpDescPtr &op_desc,
                                  const PreTaskInput &pre_task_input) {
  GELOGD("Init generate default task");
  (void)task_def;
  (void)op_desc;
  (void)pre_task_input;

  PreTaskResult result;
  result.status = PreTaskStatus::Success();
  GELOGD("success generate default task");
  return result;
}
REFISTER_PRE_GENERATE_TASK(kPreEngineDefault, GenerateDefaultTask);
}  // namespace ge
