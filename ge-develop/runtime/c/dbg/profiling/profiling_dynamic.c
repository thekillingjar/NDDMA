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
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "json_parser.h"
#include "profiling.h"
#include "toolchain/prof_api.h"

typedef uint32_t (*CAC_MSPROF_INIT_FUNC)(uint32_t dataType, void *data, uint32_t dataLen);
typedef int32_t (*ProfCommandHandle)(uint32_t type, void *data, uint32_t len);
typedef uint32_t (*CAC_MSPROF_REGISTER_CALLBACK_FUNC)(uint32_t moduleId, ProfCommandHandle handle);
typedef uint64_t (*CAC_MSPROF_GETHASHID_FUNC)(const char *hashInfo, size_t length);
typedef uint64_t (*CAC_MSPROF_GETTIME_FUNC)(void);
typedef uint32_t (*CAC_MSPROF_REPORT_DATA_FUNC)(uint32_t agingFlag, const void *data, uint32_t length);
typedef int32_t (*CAC_MSPROF_DEINIT_FUNC)(void);
typedef int32_t (*CAC_MSPROF_NOTIFY_SET_DEVICE_FUNC)(uint32_t chipId, uint32_t deviceId, bool isOpen);

typedef struct {
  CAC_MSPROF_INIT_FUNC cac_msprof_init_func;
  CAC_MSPROF_REGISTER_CALLBACK_FUNC cac_msprof_register_callback_func;
  CAC_MSPROF_GETHASHID_FUNC cac_msprof_hashid_func;
  CAC_MSPROF_GETTIME_FUNC cac_msprof_gettime_func;
  CAC_MSPROF_REPORT_DATA_FUNC cac_msprof_report_data_func;
  CAC_MSPROF_DEINIT_FUNC cac_msprof_deinit_func;
  CAC_MSPROF_NOTIFY_SET_DEVICE_FUNC cac_msprof_notify_set_device_func;
} ProfFuncs;

const char *DBG_PROFILING = "profiler";
static void *gProfHandle = NULL;
static ProfFuncs gProfFuncs = {0};
static bool gMsprofEnable = false;

static Status LoadProfSo(void) {
  const char *soName = "libprofapi.so";
  gProfHandle = dlopen(soName, RTLD_NOW);
  if (gProfHandle == NULL) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "call dlopen failed, so path[%s]. err message: %s", soName, dlerror());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

static Status DataDlopenProf(void) {
  if (LoadProfSo() != SUCCESS) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  gProfFuncs.cac_msprof_init_func = (CAC_MSPROF_INIT_FUNC)dlsym(gProfHandle, "MsprofInit");
  if (gProfFuncs.cac_msprof_init_func == NULL) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  gProfFuncs.cac_msprof_register_callback_func =
      (CAC_MSPROF_REGISTER_CALLBACK_FUNC)dlsym(gProfHandle, "MsprofRegisterCallback");
  if (gProfFuncs.cac_msprof_register_callback_func == NULL) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  gProfFuncs.cac_msprof_hashid_func = (CAC_MSPROF_GETHASHID_FUNC)dlsym(gProfHandle, "MsprofGetHashId");
  if (gProfFuncs.cac_msprof_hashid_func == NULL) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  gProfFuncs.cac_msprof_gettime_func = (CAC_MSPROF_GETTIME_FUNC)dlsym(gProfHandle, "MsprofSysCycleTime");
  if (gProfFuncs.cac_msprof_gettime_func == NULL) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  gProfFuncs.cac_msprof_report_data_func =
      (CAC_MSPROF_REPORT_DATA_FUNC)dlsym(gProfHandle, "MsprofReportAdditionalInfo");
  if (gProfFuncs.cac_msprof_report_data_func == NULL) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  gProfFuncs.cac_msprof_deinit_func = (CAC_MSPROF_DEINIT_FUNC)dlsym(gProfHandle, "MsprofFinalize");
  if (gProfFuncs.cac_msprof_deinit_func == NULL) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  gProfFuncs.cac_msprof_notify_set_device_func =
      (CAC_MSPROF_NOTIFY_SET_DEVICE_FUNC)dlsym(gProfHandle, "MsprofNotifySetDevice");
  if (gProfFuncs.cac_msprof_notify_set_device_func == NULL) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

void SetMsprofEnable(bool flag) {
  gMsprofEnable = flag;
}

bool DbgGetprofEnable(void) {
  return gMsprofEnable;
}

static Status ProcessProfData(void *const data, const uint32_t len) {
  const uint32_t commandLen = sizeof(struct MsprofCommandHandle);
  if (len != commandLen) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Len]len[%u] is invalid, it should be %u", len, commandLen);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  struct MsprofCommandHandle *const profilingCfg = (struct MsprofCommandHandle *)(data);
  SetMsprofEnable(((profilingCfg->profSwitch) & MSPROF_TASK_TIME_L0));
  return SUCCESS;
}

int32_t MsprofCtrlHandleFunc(uint32_t dataType, void *data, uint32_t dataLen) {
  if (data == NULL) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "input data is null");
    return (int32_t)ACL_ERROR_GE_PARAM_INVALID;
  }
  if (dataType == PROF_CTRL_SWITCH) {
    const int32_t ret = (int32_t)ProcessProfData(data, dataLen);
    if (ret != SUCCESS) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Process][ProfSwitch]failed to call ProcessProfData, result is %u", ret);
      return ret;
    }
  } else {
    GELOGI("get unsupported dataType %u while processing profiling data", dataType);
  }
  return SUCCESS;
}

static void FreeProfHandle(void) {
  gProfFuncs.cac_msprof_init_func = NULL;
  gProfFuncs.cac_msprof_register_callback_func = NULL;
  gProfFuncs.cac_msprof_hashid_func = NULL;
  gProfFuncs.cac_msprof_gettime_func = NULL;
  gProfFuncs.cac_msprof_report_data_func = NULL;
  gProfFuncs.cac_msprof_deinit_func = NULL;
  gProfFuncs.cac_msprof_notify_set_device_func = NULL;
  dlclose(gProfHandle);
  gProfHandle = NULL;
}

Status DbgProfInit(const char *cfg) {
  char *strCfg = CJsonFileParseKey(cfg, DBG_PROFILING);
  if (strCfg == NULL) {
    GELOGI("profiling config is off");
    return SUCCESS;
  }

  int32_t ret = (int32_t)DataDlopenProf();
  if (ret != SUCCESS) {
    mmFree(strCfg);
    if (gProfHandle != NULL) {
      FreeProfHandle();
    }
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  do {
    ret = (int32_t)gProfFuncs.cac_msprof_register_callback_func(GE_MODULE_NAME, MsprofCtrlHandleFunc);
    if (ret != SUCCESS) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "prof RegisterCallback failed");
      break;
    }
    uint32_t len = (uint32_t)strlen(strCfg);
    ret = (int32_t)gProfFuncs.cac_msprof_init_func(MSPROF_CTRL_INIT_ACL_JSON, strCfg, len);
    if (ret != SUCCESS) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "prof Init failed");
      break;
    }
  } while (0);

  mmFree(strCfg);
  return ret == SUCCESS ? SUCCESS : ACL_ERROR_GE_PARAM_INVALID;
}

static void InitMsprofAdditionalInfo(uint32_t modelId, char *om_name, struct MsprofAdditionalInfo *profData) {
  profData->magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
  profData->level = MSPROF_REPORT_MODEL_LEVEL;
  profData->type = MSPROF_REPORT_MODEL_EXEOM_TYPE;
  profData->threadId = (uint32_t)syscall(SYS_gettid);
  profData->timeStamp = gProfFuncs.cac_msprof_gettime_func();
  profData->dataLen = sizeof(struct MsprofExeomLoadInfo);
  struct MsprofExeomLoadInfo *modelLoadTag = (struct MsprofExeomLoadInfo *)profData->data;
  modelLoadTag->modelId = modelId;
  modelLoadTag->modelName = gProfFuncs.cac_msprof_hashid_func(om_name, strlen(om_name));
}

Status DbgProfReportDataProcess(uint32_t modelId, char *om) {
  if (!DbgGetprofEnable()) {
    return SUCCESS;
  }
  struct MsprofAdditionalInfo profData = {0};
  InitMsprofAdditionalInfo(modelId, om, &profData);
  uint32_t length = sizeof(profData);
  int32_t ret = (int32_t)gProfFuncs.cac_msprof_report_data_func(false, &profData, length);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "prof Report AdditionalInfo failed, phy_model_id[%d], omName[%s]", modelId, om);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status DbgProfDeInit(void) {
  if (gProfFuncs.cac_msprof_deinit_func == NULL) {
    return SUCCESS;
  }
  SetMsprofEnable(false);
  int32_t ret = gProfFuncs.cac_msprof_deinit_func();
  if (gProfHandle != NULL) {
    FreeProfHandle();
  }
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "prof Finalize failed");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status DbgNotifySetDevice(uint32_t chipId, uint32_t deviceId) {
  if (gProfFuncs.cac_msprof_notify_set_device_func == NULL) {
    return SUCCESS;
  }

  int32_t ret = gProfFuncs.cac_msprof_notify_set_device_func(chipId, deviceId, true);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "prof Notify SetDevice failed");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}
