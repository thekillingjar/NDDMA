/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "json_parser.h"
#include "profiling.h"
#include "toolchain/prof_api.h"

const char *DBG_PROFILING = "profiler";
static bool gMsprofEnable = false;
static bool gProfIsConfig = false;

void SetMsprofEnable(bool flag) {
  gMsprofEnable = flag;
}

bool DbgGetprofEnable() {
  return gMsprofEnable;
}

void SetProfIsConfig(bool flag) {
  gProfIsConfig = flag;
}

bool GetProfIsConfig() {
  return gProfIsConfig;
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

Status DbgProfInit(const char *cfg) {
  char *strCfg = CJsonFileParseKey(cfg, DBG_PROFILING);
  if (strCfg == NULL) {
    GELOGI("profiling config is off");
    return SUCCESS;
  }
  SetProfIsConfig(true);
  int32_t ret = 0;
  do {
    ret = MsprofRegisterCallback(GE_MODULE_NAME, MsprofCtrlHandleFunc);
    if (ret != SUCCESS) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "prof RegisterCallback failed");
      break;
    }
    uint32_t len = strlen(strCfg);
    ret = MsprofInit(MSPROF_CTRL_INIT_ACL_JSON, strCfg, len);
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
  profData->threadId = (uint32_t)mmGetTaskId();
  profData->timeStamp = MsprofSysCycleTime();
  profData->dataLen = sizeof(struct MsprofExeomLoadInfo);
  struct MsprofExeomLoadInfo *modelLoadTag = (struct MsprofExeomLoadInfo *)profData->data;
  modelLoadTag->modelId = modelId;
  modelLoadTag->modelName = MsprofGetHashId(om_name, strlen(om_name));
}

Status DbgProfReportDataProcess(uint32_t modelId, char *om) {
  if (!DbgGetprofEnable()) {
    return SUCCESS;
  }
  struct MsprofAdditionalInfo profData = {0};
  InitMsprofAdditionalInfo(modelId, om, &profData);
  uint32_t length = sizeof(profData);
  int32_t ret = MsprofReportAdditionalInfo(false, &profData, length);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "prof Report AdditionalInfo failed, phy_model_id[%d], omName[%s]", modelId, om);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status DbgProfDeInit(void) {
  if (!GetProfIsConfig()) {
    return SUCCESS;
  }
  SetMsprofEnable(false);
  SetProfIsConfig(false);
  int32_t ret = MsprofFinalize();
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "prof Finalize failed");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status DbgNotifySetDevice(uint32_t chipId, uint32_t deviceId) {
  if (!GetProfIsConfig()) {
    return SUCCESS;
  }
  int32_t ret = MsprofNotifySetDevice(chipId, deviceId, true);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "prof Notify SetDevice failed");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}