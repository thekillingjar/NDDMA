/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump_thread_manager.h"
#include "runtime/rt_model.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
static Status StartDumpThread(void) {
  return InitDataDump();
}

Status InitDumpThread(void) {
  rtError_t rtRet = rtDumpInit();
  if (rtRet != RT_ERROR_NONE) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dump init failed. ret=%d.", rtRet);
    return rtRet;
  }
  return StartDumpThread();
}

void DeInitDumpThread(void) {
    return;
}

void StopDumpThread(void) {
    StopDataDump();
}