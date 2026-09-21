/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_parse.h"
#include "maintain_manager.h"
#include "runtime/mem.h"
#include "runtime/rt_model.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "sort_vector.h"
#include "model_manager.h"

static mmMutex_t g_mutex;
static SortVector geModelDescManager;
mmAtomicType g_max_model_id = 0;

uint32_t GenerateModelId(void) {
  return mmValueInc(&g_max_model_id, 1);
}

static int32_t ModelDescRefObjCmp(void *a, void *b, void *appInfo) {
  (void)appInfo;
  // GE make sure a b not NULL
  return (int32_t)(*(ModelDescRefObj**)a)->modelDesc.modelId - (int32_t)(*(ModelDescRefObj**)b)->modelDesc.modelId;
}

static void FreeModelDescMem(GeModelDesc *modelDesc) {
  if ((modelDesc->innerPtrState & INNER_TBE_KERNELS_PTR) != 0) {
    (void)rtFree(modelDesc->part.kernelPtr);
    modelDesc->part.kernelPtr = NULL;
  }

  if ((modelDesc->innerPtrState & INNER_WEIGHTS_DATA_PTR) != 0) {
    (void)rtFree(modelDesc->part.weightPtr);
    modelDesc->part.weightPtr = NULL;
  }

  if ((modelDesc->innerPtrState & INNER_STATIC_TASK_DESC_PTR) != 0) {
    (void)rtFree(modelDesc->part.taskPtr);
    modelDesc->part.taskPtr = NULL;
  }

  if ((modelDesc->innerPtrState & INNER_TASK_PARAM_PTR) != 0) {
    (void)rtFree(modelDesc->part.paramPtr);
    modelDesc->part.paramPtr = NULL;
  }

  if ((modelDesc->innerPtrState & INNER_DYNAMIC_TASK_DESC_PTR) != 0) {
    (void)rtFree(modelDesc->part.dynTaskPtr);
    modelDesc->part.dynTaskPtr = NULL;
  }

  if ((modelDesc->innerPtrState & INNER_PRE_MODEL_DESC_PTR) != 0) {
    (void)rtFree(modelDesc->part.modelDescPtr);
    modelDesc->part.modelDescPtr = NULL;
  }

  if ((modelDesc->innerPtrState & INNER_FIFO_PTR) != 0) {
    (void)rtFree(modelDesc->part.fifoPtr);
    modelDesc->part.fifoPtr = NULL;
  }

  DeInitModelInOutInfo(&modelDesc->ioInfo);
  if (modelDesc->modelDbgHandle != NULL) {
    DataFreeLoadDumpInfo(modelDesc->modelDbgHandle);
    modelDesc->modelDbgHandle = NULL;
  }
  DeInitModelFifoInfo(&(modelDesc->fifoInfo));
}

void DestroyModelDesc(GeModelDesc *modelDesc) {
  if (modelDesc->phyModelId != INVALID_PHY_MODLE_ID) {
    (void)rtNanoModelDestroy(modelDesc->phyModelId);
  }
  FreeModelDescMem(modelDesc);
}

static void DestroyModelDescRef(RefObj *refObj) {
  DestroyModelDesc(GetRefObjVal(refObj));
  mmFree(GET_MAIN_BY_MEMBER(refObj, ModelDescRefObj, refObj));
}

static void DestroyModelDescRefObj(void *ptr) {
  ModelDescRefObj *mdlDescRefObj = *(ModelDescRefObj **)ptr;
  ReleaseObjRef(&mdlDescRefObj->refObj, DestroyModelDescRef);
}

void InitGeModelDescManager(void) {
  mmMutexInit(&g_mutex);
  (void)mmMutexLock(&g_mutex);
  InitSortVector(&geModelDescManager, sizeof(ModelDescRefObj *), ModelDescRefObjCmp, NULL);
  SetSortVectorDestroyItem(&geModelDescManager, DestroyModelDescRefObj);
  (void)mmMutexUnLock(&g_mutex);
}

void DeInitGeModelDescManager(void) {
  mmSetData(&g_max_model_id, 0);
  (void)mmMutexLock(&g_mutex);
  DeInitSortVector(&geModelDescManager);
  (void)mmMutexUnLock(&g_mutex);
  mmMutexDestroy(&g_mutex);
}

Status AddModelDescRef(ModelDescRefObj * mdlDescRef) {
  (void)mmMutexLock(&g_mutex);
  if (EmplaceSortVector(&geModelDescManager, &mdlDescRef) == NULL) {
    (void)mmMutexUnLock(&g_mutex);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  (void)mmMutexUnLock(&g_mutex);
  return SUCCESS;
}

void ReleaseModelDescRef(GeModelDesc *modelDesc) {
  ModelDescRefObj *mdlDescRefObj = GET_MAIN_BY_MEMBER(modelDesc, ModelDescRefObj, modelDesc);
  ReleaseObjRef(&mdlDescRefObj->refObj, DestroyModelDescRef);
}

void* FuncNull(RefObj *refObj) {
  (void)refObj;
  return NULL;
}

GeModelDesc *GetModelDescRef(uint32_t modelId) {
  ModelDescRefObj mdlDescRefObj;
  mdlDescRefObj.modelDesc.modelId = modelId;
  ModelDescRefObj *mdlDescRefObjTag = &mdlDescRefObj;
  GeModelDesc *outMdlDesc = NULL;
  (void)mmMutexLock(&g_mutex);
  ModelDescRefObj** refObj = (ModelDescRefObj**)SortVectorAtKey(&geModelDescManager, &mdlDescRefObjTag);
  if (refObj == NULL) {
    (void)mmMutexUnLock(&g_mutex);
    return NULL;
  }
  ModelDescRefObj *mdlDescRefObjTemp = *refObj;
  mdlDescRefObjTemp->appInfo = NULL;
  outMdlDesc = GetObjRef(&mdlDescRefObjTemp->refObj, FuncNull);
  (void)mmMutexUnLock(&g_mutex);
  return outMdlDesc;
}

Status DelModelDescRef(uint32_t modelId) {
  Status ret = SUCCESS;
  ModelDescRefObj mdlDescRefObj;
  mdlDescRefObj.modelDesc.modelId = modelId;
  ModelDescRefObj *mdlDescRefObjTag = &mdlDescRefObj;
  (void)mmMutexLock(&g_mutex);
  size_t i = FindSortVector(&geModelDescManager, &mdlDescRefObjTag);
  if (i < SortVectorSize(&geModelDescManager)) {
    RemoveSortVector(&geModelDescManager, i);
  } else {
    ret = ACL_ERROR_GE_EXEC_MODEL_ID_INVALID;
  }
  (void)mmMutexUnLock(&g_mutex);
  return ret;
}