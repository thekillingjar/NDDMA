/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include "securec.h"
#include "dump_config.h"
#include "parse_dbg_file.h"
#include "tlv_parse.h"
#include "parse_json_file.h"
#include "dump_thread_manager.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "dump.h"

const char *DBG_FILE_SUFFIX = ".dbg";

Status DbgDumpInit(const char *cfg) {
  Status ret = DumpConfigInit(cfg);
  if (ret != SUCCESS) {
    FreeDumpConfigRes();
    GELOGI("get dump config file failed, result = %u", ret);
    return SUCCESS;
  }
  if (GetDumpEnableFlag()) {
    ret = InitDumpThread();
    if (ret != SUCCESS) {
      FreeDumpConfigRes();
      return ret;
    }
  }
  GELOGI("dbg dump init success");
  return SUCCESS;
}

void *DbgCreateModelHandle(void) {
  if (!GetDumpEnableFlag()) {
    return NULL;
  }
  ModelDbgHandle *dbgHandle = (ModelDbgHandle *)mmMalloc(sizeof(ModelDbgHandle));
  if (dbgHandle == NULL) {
    return NULL;
  }
  memset_s(dbgHandle, sizeof(ModelDbgHandle), 0, sizeof(ModelDbgHandle));
  return (void *)dbgHandle;
}

static void DbgFreeResource(ModelDbgHandle *modelDbgHandle) {
  if (modelDbgHandle->aicpuDumpInfo != NULL) {
    (void)mmFree(modelDbgHandle->aicpuDumpInfo);
    modelDbgHandle->aicpuDumpInfo = NULL;
    modelDbgHandle->aicpuDumpInfoLen = 0UL;
  }
  if (modelDbgHandle->dumpFileContent != NULL) {
    mmFree(modelDbgHandle->dumpFileContent);
    modelDbgHandle->dumpFileContent = NULL;
    modelDbgHandle->dumpFileLen = 0UL;
    modelDbgHandle->offset = 0UL;
  }
  if (modelDbgHandle->modelName != NULL) {
    mmFree(modelDbgHandle->modelName);
    modelDbgHandle->modelName = NULL;
  }
  mmFree(modelDbgHandle);
}

Status DbgDumpPreProcess(rtMdlLoad_t *mdlLoad, size_t taskDescSize, char *modelName, void *dbgHandle) {
  if (!GetDumpEnableFlag()) {
    return SUCCESS;
  }
  if ((dbgHandle == NULL) || (modelName == NULL)) {
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  ModelDbgHandle *modelDbgHandle = (ModelDbgHandle *)dbgHandle;
  modelDbgHandle->taskDescBaseAddr = mdlLoad->taskDescBaseAddr;
  modelDbgHandle->totalTaskNum = mdlLoad->totalTaskNum;
  modelDbgHandle->taskDescSize = taskDescSize;
  char *dumpPath = GetDumpPath();
  uint32_t pathLen = strlen(dumpPath);
  if (dumpPath[pathLen - 1] != '/') {
    pathLen += 1;
  }
  uint32_t nameLen = strlen(modelName);
  uint32_t suffixLen = strlen(DBG_FILE_SUFFIX);
  uint32_t totalLen = pathLen + nameLen + suffixLen + 1; // '\0'
  char *dbgFileName  = (char *)mmMalloc(totalLen);
  if (dbgFileName == NULL) {
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  dbgFileName[pathLen - 1] = '/';
  errno_t pathRet = memcpy_s(dbgFileName, totalLen, dumpPath, pathLen - 1);
  errno_t nameRet = memcpy_s(dbgFileName + pathLen, totalLen - pathLen, modelName, nameLen);
  errno_t suffixRet = strcpy_s(dbgFileName + pathLen + nameLen, totalLen - pathLen - nameLen, DBG_FILE_SUFFIX);
  bool result = ((pathRet != EOK) || (nameRet != EOK) || (suffixRet != EOK));
  if (result) {
    mmFree(dbgFileName);
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  uint8_t *dumpFileContent = ParseDbgFile(dbgFileName, modelDbgHandle);
  mmFree(dbgFileName);
  dbgFileName = NULL;
  if (dumpFileContent == NULL) {
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  Status ret = ParseDbgHead(modelDbgHandle);
  if (ret != SUCCESS) {
    return ret;
  }
  uint32_t dumpStatus = GetDumpStatus();
  if (dumpStatus == DUMP_NORMAL_ENABLE) {
    ret = ParseDbgTlv(modelDbgHandle);
  }
  if (dumpStatus == DUMP_DEBUG_ENABLE) {
    mdlLoad->overflow_en = 1;
  }
  return ret;
}

Status DbgDumpPostProcess(uint32_t modelId, uint64_t *stepIdAddr, void *dbgHandle) {
  if (!GetDumpEnableFlag()) {
    return SUCCESS;
  }
  if (dbgHandle == NULL) {
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  ModelDbgHandle *modelDbgHandle = (ModelDbgHandle *)dbgHandle;
  modelDbgHandle->modelId = modelId;
  modelDbgHandle->stepIdAddr = stepIdAddr;

  uint32_t dumpStatus = GetDumpStatus();
  if (dumpStatus == DUMP_NORMAL_ENABLE) {
    if (modelDbgHandle->cfgMatchedCount == 0) {
      GELOGI("modelName or opName is not matched .the load dump info not need to be sent.");
      return SUCCESS;
    }
  }
  Status ret = SendLoadInfoToAicpu(modelDbgHandle);
  if (ret != SUCCESS) {
    return ret;
  }
  modelDbgHandle->isSendFlag = true;
  return SUCCESS;
}

void DbgStepIdCounterPlus(void *dbgHandle) {
  if (!GetDumpEnableFlag()) {
    return;
  }
  if (dbgHandle == NULL) {
    return;
  }
  ModelDbgHandle *modelDbgHandle = (ModelDbgHandle *)dbgHandle;
  if (!modelDbgHandle->isSendFlag) {
    GELOGI("not sent dump info, stepId not need plus plus.");
    return;
  }
  (*modelDbgHandle->stepIdAddr)++;
  return;
}

void DbgFreeLoadDumpInfo(void *dbgHandle) {
  if (!GetDumpEnableFlag()) {
    return;
  }
  if (dbgHandle == NULL) {
    return;
  }
  ModelDbgHandle *modelDbgHandle = (ModelDbgHandle *)dbgHandle;
  if (modelDbgHandle->isSendFlag) {
    (void)SendUnLoadInfoToAicpu(modelDbgHandle);
  }
  DbgFreeResource(modelDbgHandle);
  return;
}

Status DbgDumpDeInit(void) {
  if (!GetDumpEnableFlag()) {
    return SUCCESS;
  }
  SetDumpEnableFlag(false);
  rtError_t rtRet = rtDumpDeInit();
  StopDumpThread();
  DeInitDumpThread();
  FreeDumpConfigRes();
  GELOGI("dbg dump deinit success");
  return (Status)rtRet;
}