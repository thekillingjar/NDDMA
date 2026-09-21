/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include "dump_thread_manager.h"
#include "runtime/rt_model.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
typedef int(*CAC_AICPU_INIT_FUNC)(void);
typedef int(*CAC_AICPU_STOP_FUNC)(void);
static void *g_aicpuHandle = NULL;
static CAC_AICPU_INIT_FUNC cac_aicpu_init_func = NULL;
static CAC_AICPU_STOP_FUNC cac_aicpu_stop_func = NULL;
static Status LoadAicpuDumpSo(void) {
  const char *so_name = "libdatadump.so";
  g_aicpuHandle = dlopen(so_name, RTLD_NOW);
  if (g_aicpuHandle == NULL) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dlopen [%s] failed.", so_name);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  cac_aicpu_init_func = (CAC_AICPU_INIT_FUNC)dlsym(g_aicpuHandle, "InitDataDump");
  if (cac_aicpu_init_func == NULL) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "start dump thread failed.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  cac_aicpu_stop_func = (CAC_AICPU_STOP_FUNC)dlsym(g_aicpuHandle, "StopDataDump");
  if (cac_aicpu_stop_func == NULL) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

static Status StartDumpThread(void) {
  Status ret = LoadAicpuDumpSo();
  if (ret != SUCCESS) {
    return ret;
  }
  return (Status)cac_aicpu_init_func();
}

Status InitDumpThread(void) {
  Status rtRet = (Status)rtDumpInit();
  if (rtRet != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dump init failed. ret=%d.", rtRet);
    return rtRet;
  }
  return StartDumpThread();
}

void DeInitDumpThread(void) {
  if (g_aicpuHandle != NULL) {
    cac_aicpu_init_func = NULL;
    cac_aicpu_stop_func = NULL;
    dlclose(g_aicpuHandle);
    g_aicpuHandle = NULL;
  }
}
void StopDumpThread(void) {
  if (cac_aicpu_stop_func != NULL) {
    cac_aicpu_stop_func();
  }
}