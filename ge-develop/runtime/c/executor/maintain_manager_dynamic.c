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
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "framework/executor_c/ge_executor.h"
#include "maintain_manager.h"

static void *g_handle = NULL;
typedef Status (*CAC_INIT_FUNC)(const char *cfg);
typedef Status (*CAC_DEINIT_FUNC)(void);
typedef Status (*CAC_NOTIFY_DEVICE_FUNC)(uint32_t chipId, uint32_t deviceId);
typedef bool (*CAC_PROF_GET_ENABLE_FUNC)(void);
typedef Status (*LOAD_POST_PROCESS_FUNC)(uint32_t modelId, char *om, uint64_t *stepIdAddr, void *dbgHandle);
typedef void *(*CAC_CREATE_DBG_HANDLE_FUNC)(void);
typedef Status (*CAC_PRE_DUMP_FUNC)(rtMdlLoad_t *mdlLoad, size_t taskDescSize, char *modelName,
                                    void *dbgHandle);
typedef void (*CAC_FREE_LOAD_DUMP_INFO_FUNC)(void *dbgHandle);
typedef void (*CAC_STEPID_CONUTER_FUNC)(void *dbgHandle);

static CAC_INIT_FUNC cac_dbg_init_func = NULL;
static CAC_DEINIT_FUNC cac_dbg_deinit_func = NULL;
static CAC_NOTIFY_DEVICE_FUNC cac_notify_device_func = NULL;
static CAC_PROF_GET_ENABLE_FUNC cac_prof_get_enable_func = NULL;
static LOAD_POST_PROCESS_FUNC load_post_process_func = NULL;
static CAC_STEPID_CONUTER_FUNC cac_stepid_counter_func = NULL;
static CAC_CREATE_DBG_HANDLE_FUNC cac_create_dbg_handle_func = NULL;
static CAC_FREE_LOAD_DUMP_INFO_FUNC cac_free_load_dump_info_func = NULL;
static CAC_PRE_DUMP_FUNC cac_pre_dump_func = NULL;

Status GeDbgInit(const char *configPath) {
  g_handle = dlopen("libge_dbg.so", RTLD_NOW);
  if (g_handle == NULL) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dlopen dbg fail, err is: %s", dlerror());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  cac_dbg_init_func = (CAC_INIT_FUNC)dlsym(g_handle, "DbgInit");
  cac_dbg_deinit_func = (CAC_DEINIT_FUNC)dlsym(g_handle, "DbgDeInit");
  cac_notify_device_func = (CAC_NOTIFY_DEVICE_FUNC)dlsym(g_handle, "DbgNotifySetDevice");
  cac_prof_get_enable_func = (CAC_PROF_GET_ENABLE_FUNC)dlsym(g_handle, "DbgGetprofEnable");
  load_post_process_func = (LOAD_POST_PROCESS_FUNC)dlsym(g_handle, "DbgLoadModelPostProcess");
  cac_create_dbg_handle_func = (CAC_CREATE_DBG_HANDLE_FUNC)dlsym(g_handle, "DbgCreateModelHandle");
  cac_pre_dump_func = (CAC_PRE_DUMP_FUNC)dlsym(g_handle, "DbgDumpPreProcess");
  cac_stepid_counter_func = (CAC_STEPID_CONUTER_FUNC)dlsym(g_handle, "DbgStepIdCounterPlus");
  cac_free_load_dump_info_func = (CAC_FREE_LOAD_DUMP_INFO_FUNC)dlsym(g_handle, "DbgFreeLoadDumpInfo");
  return (cac_dbg_init_func != NULL) ? cac_dbg_init_func(configPath) : SUCCESS;
}

static inline void FreeDlsymHandle(void) {
  cac_dbg_init_func = NULL;
  cac_dbg_deinit_func = NULL;
  cac_notify_device_func = NULL;
  cac_prof_get_enable_func = NULL;
  load_post_process_func = NULL;
  cac_create_dbg_handle_func = NULL;
  cac_pre_dump_func = NULL;
  cac_stepid_counter_func = NULL;
  cac_free_load_dump_info_func = NULL;
}

Status GeDbgDeInit(void) {
  Status ret = (cac_dbg_deinit_func != NULL) ? cac_dbg_deinit_func() : SUCCESS;
  if (g_handle != NULL) {
    FreeDlsymHandle();
    dlclose(g_handle);
    g_handle = NULL;
  }
  return ret;
}

Status GeNofifySetDevice(uint32_t chipId, uint32_t deviceId) {
  return (cac_notify_device_func != NULL) ? cac_notify_device_func(chipId, deviceId) : SUCCESS;
}

bool GetProfEnable(void) {
  return (cac_prof_get_enable_func != NULL) ? cac_prof_get_enable_func() : false;
}

Status LoadModelPostProcess(uint32_t modelId, char *modelName, uint64_t *stepIdAddr, void *dbgHandle) {
  return (load_post_process_func != NULL) ? load_post_process_func(modelId, modelName, stepIdAddr, dbgHandle) : SUCCESS;
}

Status LoadModelPreProcess(rtMdlLoad_t *mdlLoad, size_t taskDescSize, char *modelName,
                           void *dbgHandle) {
  return cac_pre_dump_func != NULL ?
    cac_pre_dump_func(mdlLoad, taskDescSize, modelName, dbgHandle) : SUCCESS;
}

void StepIdConuterPlus(void *dbgHandle) {
  if (cac_stepid_counter_func != NULL) {
    cac_stepid_counter_func(dbgHandle);
  }
}

void DataFreeLoadDumpInfo(void *dbgHandle) {
  if (cac_free_load_dump_info_func != NULL) {
    cac_free_load_dump_info_func(dbgHandle);
  }
}

void *CreatModelDbgHandle(void) {
  return cac_create_dbg_handle_func != NULL ? cac_create_dbg_handle_func() : NULL;
}
