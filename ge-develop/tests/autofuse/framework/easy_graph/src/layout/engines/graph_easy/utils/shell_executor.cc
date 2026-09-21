/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/types.h>
#include <sys/wait.h>
#include <cstdlib>
#include <string>
#include "easy_graph/infra/log.h"
#include "layout/engines/graph_easy/utils/shell_executor.h"

EG_NS_BEGIN

Status ShellExecutor::execute(const std::string &script) {
  EG_DBG("%s", script.c_str());

  pid_t status = system(script.c_str());
  if (-1 == status) {
    EG_ERR("system execute return error!");
    return EG_FAILURE;
  }

  if (WIFEXITED(status) && (0 == WEXITSTATUS(status)))
    return EG_SUCCESS;

  EG_ERR("system execute {%s} exit status value = [0x%x], exit code: %d\n", script.c_str(), status,
         WEXITSTATUS(status));
  return EG_FAILURE;
}

EG_NS_END
