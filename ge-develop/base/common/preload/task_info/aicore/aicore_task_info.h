/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_PRELOAD_TASKINFO_AICORE_AICORE_TASK_INFO_H_
#define GE_COMMON_PRELOAD_TASKINFO_AICORE_AICORE_TASK_INFO_H_
#include "common/preload/task_info/pre_generate_task_registry.h"

namespace ge {
PreTaskResult GenerateAiCoreTask(const domi::TaskDef &task_def, const OpDescPtr &op_desc,
                                 const PreTaskInput &pre_task_input);
}  // namespace ge

#endif  // GE_COMMON_PRELOAD_TASKINFO_AICORE_AICORE_TASK_INFO_H_
