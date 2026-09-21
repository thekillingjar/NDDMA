/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_config.h"
#include "log_inner.h"
#include "model_desc_internal.h"


typedef aclError (*CheckMdlConfigFunc)(const void *const, const size_t);
typedef aclError (*SetMdlConfigFunc)(aclmdlConfigHandle *const, const void *const);

typedef struct {
  CheckMdlConfigFunc checkFunc;
  SetMdlConfigFunc setFunc;
} SetMdlConfigParamFunc;

typedef struct {
  aclmdlConfigAttr configAttr;
  SetMdlConfigParamFunc configParamFunc;
} SetMdlConfigParamFuncMap;

static aclError CheckMdlLoadType(const void *const attrValue, const size_t valueSize) {
  if (valueSize != sizeof(size_t)) {
    ACL_LOG_INNER_ERROR("valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(size_t));
    return ACL_ERROR_INVALID_PARAM;
  }
  const size_t type = *((const size_t *)attrValue);
  if ((type < (size_t)ACL_MDL_LOAD_FROM_FILE) ||
      (type > (size_t)ACL_MDL_LOAD_FROM_MEM_WITH_MEM)) {
    ACL_LOG_INNER_ERROR("type[%zu] is invalid, it should be in [%d, %d]",
        type, ACL_MDL_LOAD_FROM_FILE, ACL_MDL_LOAD_FROM_MEM_WITH_MEM);
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_LOG_INFO("check loadType[%zu] success.", type);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadType(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t type = *((const size_t *)attrValue);
  handle->mdlLoadType = type;
  handle->attrState = handle->attrState | ACL_MDL_LOAD_TYPE_SIZET_BIT;
  ACL_LOG_INFO("set loadType[%zu] success.", type);
  return ACL_SUCCESS;
}

// compared with CheckMdlLoadPtrAttr, the CheckMdlLoadPtrAttrEx requires the values attrValue point to
// can't be nullptr
static aclError CheckMdlLoadPtrAttrEx(const void *const attrValue, const size_t valueSize) {
  if (valueSize != sizeof(void *)) {
    ACL_LOG_INNER_ERROR("valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(void *));
    return ACL_ERROR_INVALID_PARAM;
  }
  const void *val = *((void *const *)attrValue);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(val);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadPath(aclmdlConfigHandle *const handle, const void *const attrValue) {
  handle->loadPath = *(char *const *)attrValue;
  handle->attrState = handle->attrState | ACL_MDL_PATH_PTR_BIT;
  ACL_LOG_INFO("set loadPath[%s] success.", handle->loadPath);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadMemPtr(aclmdlConfigHandle *const handle, const void *const attrValue) {
  handle->mdlAddr = *(void *const *)attrValue;
  handle->attrState = handle->attrState | ACL_MDL_MEM_ADDR_PTR_BIT;
  return ACL_SUCCESS;
}

static aclError CheckMdlLoadSizeAttr(const void *const attrValue, const size_t valueSize) {
  (void)attrValue;
  if (valueSize != sizeof(size_t)) {
      ACL_LOG_INNER_ERROR("valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(size_t));
      return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_SUCCESS;
}

static aclError SetMdlLoadWeightSize(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t weightSize = *((const size_t *)attrValue);
  handle->exeOMDesc.weightSize = weightSize;
  ACL_LOG_INFO("set weightSize[%zu] success.", weightSize);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadMemSize(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t memSize = *((const size_t *)attrValue);
  handle->mdlSize = memSize;
  handle->attrState = handle->attrState | ACL_MDL_MEM_SIZET_BIT;
  ACL_LOG_INFO("set memSize[%zu] success.", memSize);
  return ACL_SUCCESS;
}

static aclError CheckMdlLoadPtrAttr(const void *const attrValue, const size_t valueSize) {
  (void)attrValue;
  if (valueSize != sizeof(void *)) {
    ACL_LOG_INNER_ERROR("valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(void *));
    return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_SUCCESS;
}

static aclError SetMdlLoadWeightPtr(aclmdlConfigHandle *const handle, const void *const attrValue) {
  handle->exeOMInfo.weightPtr = *(void *const *)attrValue;
  return ACL_SUCCESS;
}

static aclError SetMdlLoadModelDescPtr(aclmdlConfigHandle *const handle, const void *const attrValue) {
  handle->exeOMInfo.modelDescPtr = *(void *const *)attrValue;
  return ACL_SUCCESS;
}

static aclError SetMdlLoadModelDescSize(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t modelDescSize = *((const size_t *)attrValue);
  handle->exeOMDesc.modelDescSize = modelDescSize;
  ACL_LOG_INFO("set mdlDescSize[%zu] success.", modelDescSize);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadKernelPtr(aclmdlConfigHandle *const handle, const void *const attrValue) {
  handle->exeOMInfo.kernelPtr = *(void *const *)attrValue;
  return ACL_SUCCESS;
}

static aclError SetMdlLoadKernelSize(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t kernelSize = *((const size_t *)attrValue);
  handle->exeOMDesc.kernelSize = kernelSize;
  ACL_LOG_INFO("set kernelSize[%zu] success.", kernelSize);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadKernelArgsPtr(aclmdlConfigHandle *const handle, const void *const attrValue) {
  handle->exeOMInfo.kernelArgsPtr = *(void *const *)attrValue;
  return ACL_SUCCESS;
}

static aclError SetMdlLoadKernelArgsSize(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t kernelArgsSize = *((const size_t *)attrValue);
  handle->exeOMDesc.kernelArgsSize = kernelArgsSize;
  ACL_LOG_INFO("set kernelArgsSize[%zu] success.", kernelArgsSize);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadStaticTaskPtr(aclmdlConfigHandle *const handle, const void *const attrValue) {
  handle->exeOMInfo.staticTaskPtr = *(void *const *)attrValue;
  return ACL_SUCCESS;
}

static aclError SetMdlLoadStaticTaskSize(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t staticTaskSize = *((const size_t *)attrValue);
  handle->exeOMDesc.staticTaskSize = staticTaskSize;
  ACL_LOG_INFO("set staticTaskSize[%zu] success.", staticTaskSize);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadDynamicTaskPtr(aclmdlConfigHandle *const handle, const void *const attrValue) {
  handle->exeOMInfo.dynamicTaskPtr = *(void *const *)attrValue;
  return ACL_SUCCESS;
}

static aclError SetMdlLoadDynamicTaskSize(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t dynamicTaskSize = *((const size_t *)attrValue);
  handle->exeOMDesc.dynamicTaskSize = dynamicTaskSize;
  ACL_LOG_INFO("set dynamicTaskSize[%zu] success.", dynamicTaskSize);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadMemType(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t memPolicy = *((const size_t *)attrValue);
  rtMemType_t type = RT_MEMORY_DEFAULT;
  aclError ret = GetMemTypeFromPolicy(memPolicy, &type);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  handle->memType = type;
  ACL_LOG_INFO("set memType[%u] success.", type);
  return ACL_SUCCESS;
}

static aclError SetMdlLoadFifoTaskPtr(aclmdlConfigHandle *const handle, const void *const attrValue) {
  handle->exeOMInfo.fifoTaskPtr = *(void *const *)attrValue;
  return ACL_SUCCESS;
}

static aclError SetMdlLoadFifoTaskSize(aclmdlConfigHandle *const handle, const void *const attrValue) {
  const size_t fifoTaskSize = *((const size_t *)attrValue);
  handle->exeOMDesc.fifoTaskSize = fifoTaskSize;
  ACL_LOG_INFO("set fifoTaskSize[%zu] success.", fifoTaskSize);
  return ACL_SUCCESS;
}

static SetMdlConfigParamFuncMap g_setMdlConfigMap[] = {
  {ACL_MDL_PRIORITY_INT32, {NULL, NULL}},
  {ACL_MDL_LOAD_TYPE_SIZET, {CheckMdlLoadType, SetMdlLoadType}},
  {ACL_MDL_PATH_PTR, {CheckMdlLoadPtrAttrEx, SetMdlLoadPath}},
  {ACL_MDL_MEM_ADDR_PTR, {CheckMdlLoadPtrAttrEx, SetMdlLoadMemPtr}},
  {ACL_MDL_MEM_SIZET, {CheckMdlLoadSizeAttr, SetMdlLoadMemSize}},
  {ACL_MDL_WEIGHT_ADDR_PTR, {CheckMdlLoadPtrAttr, SetMdlLoadWeightPtr}},
  {ACL_MDL_WEIGHT_SIZET, {CheckMdlLoadSizeAttr, SetMdlLoadWeightSize}},
  {ACL_MDL_WORKSPACE_ADDR_PTR, {NULL, NULL}},
  {ACL_MDL_WORKSPACE_SIZET, {NULL, NULL}},
  {ACL_MDL_INPUTQ_NUM_SIZET, {NULL, NULL}},
  {ACL_MDL_INPUTQ_ADDR_PTR, {NULL, NULL}},
  {ACL_MDL_OUTPUTQ_NUM_SIZET, {NULL, NULL}},
  {ACL_MDL_OUTPUTQ_ADDR_PTR, {NULL, NULL}},
  {ACL_MDL_WORKSPACE_MEM_OPTIMIZE, {NULL, NULL}},
  {ACL_MDL_WEIGHT_PATH_PTR, {NULL, NULL}},
  {ACL_MDL_MODEL_DESC_PTR, {CheckMdlLoadPtrAttr, SetMdlLoadModelDescPtr}},
  {ACL_MDL_MODEL_DESC_SIZET, {CheckMdlLoadSizeAttr, SetMdlLoadModelDescSize}},
  {ACL_MDL_KERNEL_PTR, {CheckMdlLoadPtrAttr, SetMdlLoadKernelPtr}},
  {ACL_MDL_KERNEL_SIZET, {CheckMdlLoadSizeAttr, SetMdlLoadKernelSize}},
  {ACL_MDL_KERNEL_ARGS_PTR, {CheckMdlLoadPtrAttr, SetMdlLoadKernelArgsPtr}},
  {ACL_MDL_KERNEL_ARGS_SIZET, {CheckMdlLoadSizeAttr, SetMdlLoadKernelArgsSize}},
  {ACL_MDL_STATIC_TASK_PTR, {CheckMdlLoadPtrAttr, SetMdlLoadStaticTaskPtr}},
  {ACL_MDL_STATIC_TASK_SIZET, {CheckMdlLoadSizeAttr, SetMdlLoadStaticTaskSize}},
  {ACL_MDL_DYNAMIC_TASK_PTR, {CheckMdlLoadPtrAttr, SetMdlLoadDynamicTaskPtr}},
  {ACL_MDL_DYNAMIC_TASK_SIZET, {CheckMdlLoadSizeAttr, SetMdlLoadDynamicTaskSize}},
  {ACL_MDL_MEM_MALLOC_POLICY_SIZET, {CheckMdlLoadSizeAttr, SetMdlLoadMemType}},
  {ACL_MDL_FIFO_PTR, {CheckMdlLoadPtrAttr, SetMdlLoadFifoTaskPtr}},
  {ACL_MDL_FIFO_SIZET, {CheckMdlLoadSizeAttr, SetMdlLoadFifoTaskSize}}
};

static bool CheckMdlLoadConfigFromFile(const aclmdlConfigHandle *const handle) {
  if ((handle->attrState & ACL_MDL_PATH_PTR_BIT) == 0) {
    ACL_LOG_INNER_ERROR("type[%zu]: model path is not set in handle", handle->mdlLoadType);
    return false;
  }
  return true;
}

static bool CheckMdlLoadConfigFromMem(const aclmdlConfigHandle *const handle) {
  if ((handle->attrState & ACL_MDL_MEM_ADDR_PTR_BIT) == 0) {
    ACL_LOG_INNER_ERROR("type[%zu]: model memPtr is not set in handle", handle->mdlLoadType);
    return false;
  }

  if ((handle->attrState & ACL_MDL_MEM_SIZET_BIT) == 0) {
    ACL_LOG_INNER_ERROR("model load type[%zu]: model memSize is not set in handle", handle->mdlLoadType);
    return false;
  }
  return true;
}

bool CheckMdlConfigHandle(const aclmdlConfigHandle *const handle) {
  if ((handle->attrState & ACL_MDL_LOAD_TYPE_SIZET_BIT) == 0) {
    ACL_LOG_INNER_ERROR("load type is not set in handle");
    return false;
  }

  if ((handle->mdlLoadType == (size_t)ACL_MDL_LOAD_FROM_FILE) ||
      (handle->mdlLoadType == (size_t)ACL_MDL_LOAD_FROM_FILE_WITH_MEM)) {
    if (!CheckMdlLoadConfigFromFile(handle)) {
      return false;
    }
  }

  if ((handle->mdlLoadType == (size_t)ACL_MDL_LOAD_FROM_MEM) ||
      (handle->mdlLoadType == (size_t)ACL_MDL_LOAD_FROM_MEM_WITH_MEM)) {
    if ((!CheckMdlLoadConfigFromMem(handle))) {
      return false;
    }
  }
  return true;
}

aclError aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr,
                            const void *attrValue, size_t valueSize) {
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attrValue);
  SetMdlConfigParamFunc paramFunc;
  uint32_t attrCount = sizeof(g_setMdlConfigMap) / sizeof(SetMdlConfigParamFuncMap);
  if (attr >= attrCount) {
    ACL_LOG_INNER_ERROR("attr set invalid.");
    return ACL_ERROR_INVALID_PARAM;
  }
  paramFunc = g_setMdlConfigMap[attr].configParamFunc;

  aclError ret = ACL_SUCCESS;
  if (paramFunc.checkFunc != NULL) {
    ret = paramFunc.checkFunc(attrValue, valueSize);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  } else {
    ACL_LOG_INNER_ERROR("not support this attr.");
    return ACL_ERROR_INVALID_PARAM;
  }

  if (paramFunc.setFunc != NULL) {
    ret = paramFunc.setFunc(handle, attrValue);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ACL_LOG_INFO("success set attr[%d]", (int32_t)attr);
  return ACL_SUCCESS;
}

aclmdlConfigHandle *aclmdlCreateConfigHandle() {
  aclmdlConfigHandle *configHandle = (aclmdlConfigHandle *)mmMalloc(sizeof(aclmdlConfigHandle));
  if (configHandle == NULL) {
    return NULL;
  }
  configHandle->loadPath = NULL;
  configHandle->mdlAddr = NULL;
  configHandle->priority = 0;
  configHandle->mdlLoadType = 0UL;
  configHandle->mdlSize = 0UL;
  configHandle->attrState = 0UL;
  configHandle->memType = RT_MEMORY_DEFAULT;
  aclmdlExeOMInfo info = {0};
  configHandle->exeOMInfo = info;
  aclmdlExeOMDesc desc = {0UL};
  configHandle->exeOMDesc = desc;
  return configHandle;
}

aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle *handle) {
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
  ACL_DELETE_AND_SET_NULL(handle);
  return ACL_SUCCESS;
}
