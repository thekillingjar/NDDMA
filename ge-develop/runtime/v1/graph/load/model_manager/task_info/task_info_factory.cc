/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_info_factory.h"
namespace ge {
namespace {
std::shared_ptr<TaskInfoFactory> g_user_defined_ins = nullptr;
}
TaskInfoFactory &TaskInfoFactory::Instance() {
  if (g_user_defined_ins != nullptr) {
    return *g_user_defined_ins;
  } else {
    static TaskInfoFactory instance;
    return instance;
  }
}
void TaskInfoFactory::Replace(std::shared_ptr<TaskInfoFactory> ins) {
  g_user_defined_ins = std::move(ins);
}
}  // namespace ge
