/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling.h"
#include "dump.h"
#include "dbg_main.h"

Status DbgInit(const char *cfg) {
  Status ret = DbgProfInit(cfg);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return DbgDumpInit(cfg);
}

Status DbgDeInit(void) {
  (void)DbgProfDeInit();
  (void)DbgDumpDeInit();
  return SUCCESS;
}

Status DbgLoadModelPostProcess(uint32_t modelId, char *modelName, uint64_t *stepIdAddr, void *dbgHandle) {
  Status ret = DbgProfReportDataProcess(modelId, modelName);
  if (ret != SUCCESS) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return DbgDumpPostProcess(modelId, stepIdAddr, dbgHandle);
}
