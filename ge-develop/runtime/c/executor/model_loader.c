/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_loader.h"
#include "maintain_manager.h"
#include "model_manager.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"
#include "maintain_manager.h"

typedef struct {
  const ModelData *modelData;
  GeModelDesc *mdlDesc;
  Status ret;
} ModelLoadParam;

static void CreateRtMdlDesc(GeModelDesc *mdlDesc, rtMdlLoad_t *mdlLoad) {
  mdlLoad->taskDescBaseAddr = mdlDesc->part.taskPtr;
  mdlLoad->pcBaseAddr = mdlDesc->part.kernelPtr;
  mdlLoad->paramBaseAddr = mdlDesc->part.paramPtr;
  mdlLoad->weightBaseAddr = mdlDesc->part.weightPtr;

  (void)rtMemcpy(&(mdlLoad->weightPrefetch), sizeof(uint8_t), &(((ModelDesc *)mdlDesc->part.modelDescPtr)->weight_type),
                 sizeof(uint8_t), RT_MEMCPY_DEVICE_TO_HOST);
  (void)rtMemcpy(&(mdlLoad->totalTaskNum), sizeof(uint16_t), &(((ModelDesc *)mdlDesc->part.modelDescPtr)->task_num),
                 sizeof(uint16_t), RT_MEMCPY_DEVICE_TO_HOST);
  return;
}

static Status LoadExeModel(ModelLoadParam *paramPtr) {
  paramPtr->mdlDesc->memType = paramPtr->modelData->memType;
  Status ret = MdlPartitionParse(paramPtr->modelData, paramPtr->mdlDesc);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "parse model partition failed.");
    return ret;
  }

  rtMdlLoad_t mdlLoad = {0};
  CreateRtMdlDesc(paramPtr->mdlDesc, &mdlLoad);
  ModelFileHeader *header = (ModelFileHeader *)paramPtr->modelData->modelData;
  void *dbgHandle = CreatModelDbgHandle();
  if (dbgHandle != NULL) {
    paramPtr->mdlDesc->modelDbgHandle = dbgHandle;
  }
  size_t taskDescSize = paramPtr->mdlDesc->part.taskSize;
  ret = LoadModelPreProcess(&mdlLoad, taskDescSize, (char *)header->name, dbgHandle);
  if (ret != SUCCESS) {
    return ret;
  }
  uint32_t phyMdlId = 0U;
  Status rtRet = (Status)rtNanoModelLoad(&mdlLoad, &phyMdlId);
  if (rtRet != SUCCESS) {
    return rtRet;
  }
  paramPtr->mdlDesc->phyModelId = phyMdlId;
  ret = LoadModelPostProcess(phyMdlId, (char *)header->name, &paramPtr->mdlDesc->stepId, dbgHandle);
  if (ret != SUCCESS) {
    return ret;
  }
  GELOGI("model load success modelId[%d], mid[%d].", paramPtr->mdlDesc->modelId, phyMdlId);
  return SUCCESS;
}

static void *InitModelDescRef(RefObj *refObj) {
  ModelDescRefObj *mdlDescRefObj = GET_MAIN_BY_MEMBER(refObj, ModelDescRefObj, refObj);
  ModelLoadParam *param = (ModelLoadParam *)mdlDescRefObj->appInfo;
  param->ret = LoadExeModel(param);
  if (param->ret == SUCCESS) {
    return param->mdlDesc;
  }
  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "model load failed ret[%d]", param->ret);
  return NULL;
}

static ModelDescRefObj *CreateModelDescRefObj(void) {
  ModelDescRefObj *mdlDescRefObj = (ModelDescRefObj *)mmMalloc(sizeof(ModelDescRefObj));
  if (mdlDescRefObj == NULL) {
    return NULL;
  }
  mdlDescRefObj->modelDesc.innerPtrState = 0;
  Partition part = {0};
  mdlDescRefObj->modelDesc.part = part;
  mdlDescRefObj->modelDesc.phyModelId = INVALID_PHY_MODLE_ID;
  mdlDescRefObj->modelDesc.stepId = 0UL;
  mdlDescRefObj->modelDesc.modelDbgHandle = NULL;
  InitModelInOutInfo(&mdlDescRefObj->modelDesc.ioInfo);
  InitModelFifoInfo(&mdlDescRefObj->modelDesc.fifoInfo);
  mdlDescRefObj->modelDesc.memType = RT_MEMORY_DEFAULT;
  GELOGD("create model desc ref obj success.");
  return mdlDescRefObj;
}

Status LoadOfflineModelFromData(uint32_t *modelId, const ModelData *modelData) {
  Status result = SUCCESS;
  *modelId = GenerateModelId();
  ModelDescRefObj *mdlDescRefObj = CreateModelDescRefObj();
  if (mdlDescRefObj == NULL) {
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  mdlDescRefObj->modelDesc.modelId = *modelId;
  const ModelData *data = modelData;
  ModelLoadParam param = {data, &mdlDescRefObj->modelDesc, result};
  mdlDescRefObj->appInfo = &param;
  InitRefObj(&mdlDescRefObj->refObj);
  if (GetObjRef(&mdlDescRefObj->refObj, InitModelDescRef) != NULL) {
    Status ret = AddModelDescRef(mdlDescRefObj);
    if (ret != SUCCESS) {
      DestroyModelDesc(param.mdlDesc);
      mmFree(mdlDescRefObj);
      return ret;
    }
  } else {
    DestroyModelDesc(param.mdlDesc);
    mmFree(mdlDescRefObj);
    return param.ret;
  }
  GELOGD("load model[%u] success.", *modelId);
  return SUCCESS;
}

Status UnloadOfflineModel(const uint32_t modelId) {
  Status ret = DelModelDescRef(modelId);
  if (ret != SUCCESS) {
    return ret;
  }
  GELOGI("unload model[%u] success.", modelId);
  return SUCCESS;
}