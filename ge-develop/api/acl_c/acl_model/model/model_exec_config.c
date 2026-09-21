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
#include "model_config.h"
#include "framework/executor_c/ge_executor.h"
#include "log_inner.h"
#include "model_desc_internal.h"

typedef aclError (*CheckMdlExecConfigFunc)(const void *const, const size_t);
typedef aclError (*SetMdlExecConfigFunc)(aclmdlExecConfigHandle *const, const void *const);

typedef struct {
  CheckMdlExecConfigFunc checkExecFunc;
  SetMdlExecConfigFunc setExecFunc;
} SetMdlExecConfigParamFunc;

typedef struct {
  aclmdlExecConfigAttr configAttr;
  SetMdlExecConfigParamFunc configParamFunc;
} SetMdlExecConfigParamFuncMap;

static aclError CheckMdlExecPtrAttr(const void *const attrValue, const size_t valueSize) {
  (void)attrValue;
  if (valueSize != sizeof(void *)) {
    ACL_LOG_INNER_ERROR("valueSize[%zu] is invalid, it should be %zu",
        valueSize, sizeof(void *));
    return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_SUCCESS;
}

static aclError CheckMdlExecSizeAttr(const void *const attrValue, const size_t valueSize) {
  (void)attrValue;
  if (valueSize != sizeof(size_t)) {
    ACL_LOG_INNER_ERROR("valueSize[%zu] is invalid, it should be %zu",
        valueSize, sizeof(size_t));
    return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_SUCCESS;
}

static aclError SetMdlExecWorkSpacePtr(aclmdlExecConfigHandle *const handle, const void *const attrValue) {
  handle->workPtr = *(void *const *)attrValue;
  return ACL_SUCCESS;
}

static aclError SetMdlExecWorkSize(aclmdlExecConfigHandle *const handle, const void *const attrValue) {
  const size_t workSize = *((const size_t *)attrValue);
  handle->workSize = workSize;
  ACL_LOG_INFO("set workSize[%zu] success.", handle->workSize);
  return ACL_SUCCESS;
}

static aclError SetMdlExecMpamIdSize(aclmdlExecConfigHandle *const handle, const void *const attrValue) {
  const size_t mpamId = *((const size_t *)attrValue);
  handle->mpamId = mpamId;
  ACL_LOG_INFO("set mpamId[%zu] success.", handle->mpamId);
  return ACL_SUCCESS;
}

static aclError SetMdlExecAicQosSize(aclmdlExecConfigHandle *const handle, const void *const attrValue) {
  const size_t aicQos = *((const size_t *)attrValue);
  handle->aicQos = aicQos;
  ACL_LOG_INFO("set aicQos[%zu] success.", handle->aicQos);
  return ACL_SUCCESS;
}

static aclError SetMdlExecAicOstSize(aclmdlExecConfigHandle *const handle, const void *const attrValue) {
  const size_t aicOst = *((const size_t *)attrValue);
  handle->aicOst = aicOst;
  ACL_LOG_INFO("set aicOst[%zu] success.", handle->aicOst);
  return ACL_SUCCESS;
}

static aclError SetMdlExeTimeOutSize(aclmdlExecConfigHandle *const handle, const void *const attrValue) {
  const size_t mecTimeThreshHold = *((const size_t *)attrValue);
  handle->mecTimeThreshHold = mecTimeThreshHold;
  ACL_LOG_INFO("set mecTimeThreshHold[%zu] success.", handle->mecTimeThreshHold);
  return ACL_SUCCESS;
}

static SetMdlExecConfigParamFuncMap g_setMdlExecConfigMap[ACL_MDL_MEC_TIMETHR_SIZET + 1] = {
  {ACL_MDL_STREAM_SYNC_TIMEOUT, {NULL, NULL}},
  {ACL_MDL_EVENT_SYNC_TIMEOUT, {NULL, NULL}},
  {ACL_MDL_WORK_ADDR_PTR, {&CheckMdlExecPtrAttr, &SetMdlExecWorkSpacePtr}},
  {ACL_MDL_WORK_SIZET, {&CheckMdlExecSizeAttr, &SetMdlExecWorkSize}},
  {ACL_MDL_MPAIMID_SIZET, {&CheckMdlExecSizeAttr, &SetMdlExecMpamIdSize}},
  {ACL_MDL_AICQOS_SIZET, {&CheckMdlExecSizeAttr, &SetMdlExecAicQosSize}},
  {ACL_MDL_AICOST_SIZET, {&CheckMdlExecSizeAttr, &SetMdlExecAicOstSize}},
  {ACL_MDL_MEC_TIMETHR_SIZET, {&CheckMdlExecSizeAttr, &SetMdlExeTimeOutSize}}
};

aclError aclmdlSetExecConfigOpt(aclmdlExecConfigHandle *handle, aclmdlExecConfigAttr attr,
    const void *attrValue, size_t valueSize) {
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attrValue);
  SetMdlExecConfigParamFunc paramFunc;
  uint32_t attrCount = sizeof(g_setMdlExecConfigMap) / sizeof(SetMdlExecConfigParamFuncMap);
  if (attr >= attrCount) {
    ACL_LOG_INNER_ERROR("attr set invalid.");
    return ACL_ERROR_INVALID_PARAM;
  }
  paramFunc = g_setMdlExecConfigMap[attr].configParamFunc;

  if (paramFunc.checkExecFunc != NULL) {
    aclError ret = paramFunc.checkExecFunc(attrValue, valueSize);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  } else {
    ACL_LOG_INNER_ERROR("not support set this attr.");
    return ACL_ERROR_INVALID_PARAM;
  }

  if (paramFunc.setExecFunc != NULL) {
    aclError ret = paramFunc.setExecFunc(handle, attrValue);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  return ACL_SUCCESS;
}

aclmdlExecConfigHandle *aclmdlCreateExecConfigHandle() {
  aclmdlExecConfigHandle *configExecHandle = (aclmdlExecConfigHandle *)mmMalloc(sizeof(aclmdlExecConfigHandle));
  if (configExecHandle == NULL) {
    return NULL;
  }
  configExecHandle->workPtr = NULL;
  configExecHandle->mpamId = 0UL;
  configExecHandle->aicQos = 0UL;
  configExecHandle->aicOst = 0UL;
  configExecHandle->workSize = 0UL;
  configExecHandle->mecTimeThreshHold = 0UL;
  return configExecHandle;
}

aclError aclmdlDestroyExecConfigHandle(const aclmdlExecConfigHandle *handle) {
  aclmdlExecConfigHandle *aliasHandle = (aclmdlExecConfigHandle *)handle;
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aliasHandle);
  ACL_DELETE_AND_SET_NULL(aliasHandle);
  return ACL_SUCCESS;
}