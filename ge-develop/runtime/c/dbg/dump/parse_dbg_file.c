/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include "securec.h"
#include "parse_json_file.h"
#include "tlv_parse.h"
#include "dump_config.h"
#include "runtime/dev.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "framework/executor_c/ge_executor.h"
#include "parse_dbg_file.h"
#define BUFFER_SIZE  32
#define IDMAX        21
static void UpdateTaskDumpEnable(void *taskDescBaseAddr, size_t taskDescSize, uint32_t taskId) {
  rtError_t rtRet = rtSetTaskDescDumpFlag(taskDescBaseAddr, taskDescSize, taskId);
  if (rtRet != RT_ERROR_NONE) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "set taskId[%d]'s dump flag failed, taskDescSize %zu, ret=%d.",
           taskId, taskDescSize, rtRet);
    return;
  }
  GELOGI("update dump flag of taskId %d, taskDescSize %zu.", taskId, taskDescSize);
}

static Status ProcOptTlvList(uint32_t len, uint8_t *tlvValue, void *appInfo) {
  Status ret = SUCCESS;
  struct DbgOpDescTlv1 *desc = (struct DbgOpDescTlv1 *)tlvValue;
  GELOGI("desc->num is %d", desc->num);
  uint8_t *tlvValueTmp = tlvValue + sizeof(struct DbgOpDescTlv1);
  ModelDbgHandle *modelDbgHandle = (ModelDbgHandle *)appInfo;
  uint32_t offset = 0U;
  static uint32_t subTlvList[] = {DBG_L2_TLV_TYPE_OP_NAME, DBG_L2_TLV_TYPE_ORI_OP_NAME};
  static uint32_t subTlvNum = sizeof(subTlvList) / sizeof(subTlvList[0]);
  for (; offset < len; ) {
    struct DbgOpDescParamTlv1 *opDesc = (struct DbgOpDescParamTlv1 *)(tlvValueTmp + offset);
    if (!CheckTlvLenValid(len, offset, sizeof(struct DbgOpDescParamTlv1))) {
      break;
    }
    offset += sizeof(struct DbgOpDescParamTlv1);
    GELOGI("taskId[%d] taskNum[%u].", opDesc->task_id, modelDbgHandle->totalTaskNum);
    if (opDesc->task_id >= modelDbgHandle->totalTaskNum) {
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    struct TlvHead *parseSubTlvList[2] = {NULL};
    if (!CheckTlvLenValid(len, offset, opDesc->l2_tlv_list_len)) {
      ret = ACL_ERROR_GE_INTERNAL_ERROR;
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dbg file tlv len check failed.");
      break;
    }
    uint32_t tlvNum = ParseSubTlvListU16(opDesc->l2_tlv, opDesc->l2_tlv_list_len,
                                         subTlvNum, subTlvList, parseSubTlvList);
    if (tlvNum == 0U) {
      ret = ACL_ERROR_GE_INTERNAL_ERROR;
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dbg file parse tlv failed.");
      break;
    }
    if (tlvNum != subTlvNum) {
      if (IsOpNameMatch(parseSubTlvList[0]->data, (uint16_t)parseSubTlvList[0]->len, modelDbgHandle->modelName)) {
        UpdateTaskDumpEnable(modelDbgHandle->taskDescBaseAddr, modelDbgHandle->taskDescSize, opDesc->task_id);
        modelDbgHandle->cfgMatchedCount++;
      }
    } else {
      if (IsOriOpNameMatch(parseSubTlvList[1]->data, (uint16_t)parseSubTlvList[1]->len, modelDbgHandle->modelName)) {
        UpdateTaskDumpEnable(modelDbgHandle->taskDescBaseAddr, modelDbgHandle->taskDescSize, opDesc->task_id);
        modelDbgHandle->cfgMatchedCount++;
      }
    }
    offset += opDesc->l2_tlv_list_len;
  }
  return ret;
}

static Status ProModelName(uint32_t len, uint8_t *tlvValue, void *appInfo) {
  ModelDbgHandle *modelDbgHandle = (ModelDbgHandle *)appInfo;
  struct DbgModelNameTlv1 *modelName = (struct DbgModelNameTlv1 *)tlvValue;
  char *name = StringDepthCpy(len, (void *)modelName->name);
  if (name == NULL) {
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  modelDbgHandle->modelName = name;
  GELOGI("dbg file modelName is [%s].", modelDbgHandle->modelName);
  return SUCCESS;
}

Status ParseDbgTlv(ModelDbgHandle *dbgHandle) {
  static TlvProcPair tlvProcMap[] = {
    {DBG_L1_TLV_TYPE_MODEL_NAME, ProModelName},
    {DBG_L1_TLV_TYPE_OP_DESC, ProcOptTlvList}
  };
  uint8_t *subTlvList = dbgHandle->dumpFileContent + dbgHandle->offset;
  uint32_t subTlvListLen = (uint32_t)(dbgHandle->dumpFileLen - dbgHandle->offset);
  uint32_t parseTlvNum = sizeof(tlvProcMap) / sizeof(tlvProcMap[0]);
  return ParseAndProcSubTlvListU32(subTlvList, subTlvListLen, parseTlvNum, tlvProcMap, (void *)dbgHandle);
}

Status ParseDbgHead(ModelDbgHandle *dbgHandle) {
  uint32_t magic = ((struct DbgDataHead *)dbgHandle->dumpFileContent)->magic;
  if (magic != DBG_DATA_HEAD_MAGIC) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dbg file check magic failed.");
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  dbgHandle->offset = sizeof(struct DbgDataHead);
  return SUCCESS;
}

uint8_t *ParseDbgFile(char *dbgFilePath, ModelDbgHandle *dbgHandle) {
  uint32_t fileLen = 0U;

  mmFileHandle *fd = mmOpenFile(dbgFilePath, FILE_READ_BIN);
  if (fd == NULL) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dbg file path is %s can not opened.", dbgFilePath);
    return NULL;
  }
  (void)mmSeekFile(fd, 0, MM_SEEK_FILE_END);
  fileLen = (uint32_t)mmTellFile(fd);
  uint8_t *dbgData = (uint8_t *)mmMalloc(fileLen);
  if (dbgData == NULL) {
    mmCloseFile(fd);
    return NULL;
  }
  (void)mmSeekFile(fd, 0, MM_SEEK_FILE_BEGIN);
  if (mmReadFile(dbgData, (int32_t)sizeof(uint8_t), (int32_t)fileLen, fd) == 0) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "dbg file fread failed.");
    mmCloseFile(fd);
    (void)mmFree(dbgData);
    return NULL;
  }
  mmCloseFile(fd);
  dbgHandle->dumpFileContent = dbgData;
  dbgHandle->dumpFileLen = fileLen;
  GELOGI("mapping dbg file success.");
  return dbgData;
}

static Status InitUnLoadDumpInfo(ModelDbgHandle *dbgHandle) {
  Status ret = ACL_ERROR_GE_PARAM_INVALID;
  if (dbgHandle->aicpuDumpInfo != NULL) {
    uint8_t *tlvUnload = (uint8_t *)(dbgHandle->aicpuDumpInfo + dbgHandle->dumpFileLen);
    uint8_t *unLoadDumpInfo = (uint8_t *)(tlvUnload + sizeof(struct TlvHead));
    struct DbgModelDescTlv1 *dbgModelDesc = (struct DbgModelDescTlv1 *)(unLoadDumpInfo);
    dbgModelDesc->flag = 0;
    ret = SUCCESS;
  }
  return ret;
}

static uint32_t IntToStr(size_t num, char *str, uint32_t length) {
#define TEN 10
  char tmp[IDMAX];
  uint32_t i = 0;
  do {
    tmp[i++] = '0' + num % TEN;
    num /= TEN;
  } while (num > 0);
  uint32_t len = 0;
  if (length >= (i + 1)) {
    while (i > 0) {
      str[len++] = tmp[--i];
    }
    str[len] = '\0';
  }
  return len;
}

static char *ConcatDumpPath(char *dumpPath) {
  // process dump_path + timestamp + deviceId
  char *dumPathTemp = NULL;
  char *dumpPathStr = GetDumpPath();
  time_t rawTime;
  char timeStamp[BUFFER_SIZE] = "";
  time(&rawTime);
  struct tm *ptm = localtime(&rawTime);
  if (ptm != NULL) {
    (void)strftime(timeStamp, BUFFER_SIZE, "%Y%m%d%H%M%S", ptm);
  }
  int32_t deviceId = 0;
  (void)rtGetDevice(&deviceId);
  bool result = 0;
  char devIdStr[IDMAX];
  uint32_t devLen = IntToStr((size_t)deviceId, devIdStr, IDMAX);
  uint32_t pathLen = strlen(dumpPathStr);
  uint32_t timeLen = strlen(timeStamp);
  if (dumpPathStr[pathLen - 1] != '/') {
    errno_t retL1 = memcpy_s(dumpPath, PATH_MAX, dumpPathStr, pathLen);
    errno_t retL2 = memcpy_s(dumpPath + pathLen, PATH_MAX - pathLen, "/", 1);
    errno_t retL3 = memcpy_s(dumpPath + pathLen + 1, PATH_MAX - pathLen - 1, timeStamp, timeLen);
    errno_t retL4 = memcpy_s(dumpPath + pathLen + 1 + timeLen, PATH_MAX - pathLen - 1 - timeLen, "/", 1);
    errno_t retL5 = memcpy_s(dumpPath + pathLen + 1 + timeLen + 1, PATH_MAX - pathLen - 1 - timeLen - 1, devIdStr, devLen);
    errno_t retL6 = strcpy_s(dumpPath + pathLen + 1 + timeLen + 1 + devLen, PATH_MAX - pathLen - 1 - timeLen - 1 - devLen, "/");
    result = (retL1 != EOK || retL2 != EOK || retL3 != EOK || retL4 != EOK || retL5 != EOK || retL6 != EOK);
  } else {
    errno_t retL1 = memcpy_s(dumpPath, PATH_MAX, dumpPathStr, pathLen);
    errno_t retL2 = memcpy_s(dumpPath + pathLen, PATH_MAX - pathLen, timeStamp, timeLen);
    errno_t retL3 = memcpy_s(dumpPath + pathLen + timeLen, PATH_MAX - pathLen - timeLen, "/", 1);
    errno_t retL4 = memcpy_s(dumpPath + pathLen + timeLen + 1, PATH_MAX - pathLen - timeLen - 1, devIdStr, devLen);
    errno_t retL5 = strcpy_s(dumpPath + pathLen + timeLen + 1 + devLen, PATH_MAX - pathLen - timeLen - 1 - devLen, "/");
    result = (retL1 != EOK || retL2 != EOK || retL3 != EOK || retL4 != EOK || retL5 != EOK);
  }
  dumPathTemp = dumpPath;
  if (result) {
    dumPathTemp = dumpPathStr;
    GELOGW("concat dumpPath failed.");
  }
  GELOGI("GetDumpPath [%s].", dumPathTemp);
  return dumPathTemp;
}

static Status InitLoadDumpInfo(ModelDbgHandle *dbgHandle) {
  char dumpPath[PATH_MAX] = "";
  char *dumPathTemp = ConcatDumpPath(dumpPath);

  size_t total_len = dbgHandle->dumpFileLen + sizeof(struct TlvHead) + sizeof(struct DbgModelDescTlv1) +
                     sizeof(struct TlvHead) + strlen(dumPathTemp);
  dbgHandle->aicpuDumpInfo = (uint8_t *)mmMalloc(total_len);
  if (dbgHandle->aicpuDumpInfo == NULL) {
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  errno_t ret = memcpy_s(dbgHandle->aicpuDumpInfo, total_len,
                         dbgHandle->dumpFileContent, dbgHandle->dumpFileLen);
  if (ret != EOK) {
      (void)mmFree(dbgHandle->aicpuDumpInfo);
      dbgHandle->aicpuDumpInfo = NULL;
      return ret;
  }
  dbgHandle->aicpuDumpInfoLen = total_len;

  // model desc process
  uint8_t *tlv1 = (uint8_t *)(dbgHandle->aicpuDumpInfo + dbgHandle->dumpFileLen);
  struct TlvHead *tlv = (struct TlvHead *)tlv1;
  tlv->type = DBG_L1_TLV_TYPE_MODEL_DESC;
  tlv->len = sizeof(struct DbgModelDescTlv1);
  uint8_t *memCpyTlv1Addr = (uint8_t *)(tlv1 + sizeof(struct TlvHead));
  struct DbgModelDescTlv1 *dbgModelDesc = (struct DbgModelDescTlv1 *)(memCpyTlv1Addr);
  dbgModelDesc->flag = 1;
  dbgModelDesc->model_id = dbgHandle->modelId;
  dbgModelDesc->step_id_addr = dbgHandle->stepIdAddr;
  dbgModelDesc->iterations_per_loop_addr = (uintptr_t)NULL;
  dbgModelDesc->loop_cond_addr = (uintptr_t)NULL;
  dbgModelDesc->dump_data = GetDumpData();
  dbgModelDesc->dump_mode = GetDumpMode();
  GELOGI("step_id value[%u].", (uint32_t)*dbgModelDesc->step_id_addr);
  // dump path process
  uint8_t *tlv2 = (uint8_t *)(memCpyTlv1Addr + sizeof(struct DbgModelDescTlv1));
  struct TlvHead *tlvDumpPath = (struct TlvHead *)tlv2;
  tlvDumpPath->type = DBG_L1_TLV_TYPE_DUMP_PATH;
  tlvDumpPath->len = (uint32_t)strlen(dumPathTemp);
  uint8_t *memCpyTlv2Addr = (uint8_t *)(tlv2 + sizeof(struct TlvHead));
  ret = memcpy_s(memCpyTlv2Addr, tlvDumpPath->len, dumPathTemp, tlvDumpPath->len);
  if (ret != EOK) {
      (void)mmFree(dbgHandle->aicpuDumpInfo);
      dbgHandle->aicpuDumpInfo = NULL;
      return ret;
  }
  GELOGI("init load dump info success.");
  return SUCCESS;
}

Status SendLoadInfoToAicpu(ModelDbgHandle *dbgHandle) {
  Status ret = InitLoadDumpInfo(dbgHandle);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "init load dump info failed ret[%u].", ret);
    return ret;
  }
  Status rtRet =
    (Status)rtMsgSend((uint32_t)mmGetTaskId(), 0, 0, dbgHandle->aicpuDumpInfo, (uint32_t)dbgHandle->dumpFileLen);
  if (rtRet != SUCCESS) {
    if (dbgHandle->aicpuDumpInfo != NULL) {
      (void)mmFree(dbgHandle->aicpuDumpInfo);
      dbgHandle->aicpuDumpInfo = NULL;
    }
    return rtRet;
  }
  GELOGI("send load info to aicpu success.");
  return SUCCESS;
}

Status SendUnLoadInfoToAicpu(ModelDbgHandle *dbgHandle) {
  Status ret = InitUnLoadDumpInfo(dbgHandle);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "init unload dump info failed ret[%u].", ret);
    return ret;
  }
  Status rtRet =
    (Status)rtMsgSend((uint32_t)mmGetTaskId(), 0, 0, dbgHandle->aicpuDumpInfo, (uint32_t)dbgHandle->dumpFileLen);
  if (rtRet != SUCCESS) {
    return rtRet;
  }
  GELOGI("send unload info to aicpu success.");
  return SUCCESS;
}
