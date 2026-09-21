/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <stdlib.h>
#include <string.h>
#include "securec.h"
#include "acl/acl_base.h"
#include "acl/acl_mdl.h"
#include "framework/executor_c/ge_executor.h"
#include "log_inner.h"
#include "model_desc_internal.h"
#include "vector.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define IDX_0 0
#define IDX_1 1
#define IDX_2 2
#define IDX_3 3
#define IDX_4 4
#define IDX_5 5
#define TEN 10
#define MIDMAX 11
#define IDXMAX 21

const char *TENSOR_INPUT_STR = "_input_";
const char *TENSOR_OUTPUT_STR = "_output_";
const char *ACL_MODEL_PRE = "acl_modelId_";
const size_t TENSOR_INPUT_STR_LEN = 7U;
const size_t TENSOR_OUTPUT_STR_LEN = 8U;
const size_t ACL_MODEL_PRE_LEN = 12U;

static void ModelTensorDescDestroy(void *base) {
  aclmdlTensorDesc *config = (aclmdlTensorDesc *)base;
  if (config->name != NULL) {
    mmFree(config->name);
    config->name = NULL;
  }
  DeInitVector(&config->dims);
}

static void ModelDestroyDesc(aclmdlDesc *modelDesc) {
  DeInitVector(&modelDesc->inputDesc);
  DeInitVector(&modelDesc->outputDesc);
}

static aclError ConvertInOutTensorDesc(ModelInOutTensorDesc *inOutInfo, aclmdlTensorDesc *tensorDesc) {
  size_t len = strlen(inOutInfo->name) + 1;
  tensorDesc->name = (char *)mmMalloc(sizeof(char) * len);
  if (tensorDesc->name == NULL) {
    return ACL_ERROR_FAILURE;
  }
  errno_t ret = strcpy_s(tensorDesc->name, len, inOutInfo->name);
  if (ret != EOK) {
    mmFree(tensorDesc->name);
    tensorDesc->name = NULL;
    return ACL_ERROR_FAILURE;
  }
  tensorDesc->size = inOutInfo->size;
  tensorDesc->format = (aclFormat)inOutInfo->format;
  tensorDesc->dataType = (aclDataType)inOutInfo->dataType;

  InitVector(&tensorDesc->dims, sizeof(int64_t));
  size_t dimsSize = VectorSize(&inOutInfo->dims);
  for (size_t i = 0UL; i < dimsSize; ++i) {
    EmplaceBackVector(&tensorDesc->dims, VectorAt(&inOutInfo->dims, i));
  }
  return ACL_SUCCESS;
}

static aclError TransOmInfo2ModelDesc(ModelInOutInfo *info, aclmdlDesc *modelDesc) {
  size_t inputNum = VectorSize(&info->input_desc);
  for (size_t i = 0UL; i < inputNum; ++i) {
    aclmdlTensorDesc tensorDesc;
    ConvertInOutTensorDesc((ModelInOutTensorDesc *)VectorAt(&info->input_desc, i), &tensorDesc);
    EmplaceBackVector(&modelDesc->inputDesc, &tensorDesc);
  }

  size_t outputNum = VectorSize(&info->output_desc);
  for (size_t i = 0UL; i < outputNum; ++i) {
    aclmdlTensorDesc tensorDesc;
    ConvertInOutTensorDesc((ModelInOutTensorDesc *)VectorAt(&info->output_desc, i), &tensorDesc);
    EmplaceBackVector(&modelDesc->outputDesc, &tensorDesc);
  }
  return ACL_SUCCESS;
}

static aclError GetModelDescInfoInner(aclmdlDesc *modelDesc, ModelData *modelData) {
  ModelInOutInfo info;
  Status ret = GetModelDescInfoFromMem(modelData, &info);
  if (ret != SUCCESS) {
    ACL_LOG_CALL_ERROR("Ge failed, ret[%u]", ret);
    return ACL_ERROR_GE_FAILURE;
  }

  aclError aclRet = TransOmInfo2ModelDesc(&info, modelDesc);
  if (aclRet != ACL_SUCCESS) {
    return aclRet;
  }
  DestroyModelInOutInfo(&info);
  return ACL_SUCCESS;
}

static const char *GetRealTensorName(const aclmdlDesc *modelDesc, const char *tensorName) {
  size_t inputNum = VectorSize(&modelDesc->inputDesc);
  size_t outputNum = VectorSize(&modelDesc->outputDesc);
  for (size_t idx = 0U; idx < inputNum; ++idx) {
    const aclmdlTensorDesc *tensorDesc = (const aclmdlTensorDesc *)ConstVectorAt(&modelDesc->inputDesc, idx);
    if (strcmp(tensorDesc->name, tensorName) == 0) {
      return tensorDesc->name;
    }
  }

  for (size_t idx = 0U; idx < outputNum; ++idx) {
    const aclmdlTensorDesc *tensorDesc = (const aclmdlTensorDesc *)ConstVectorAt(&modelDesc->outputDesc, idx);
    if (strcmp(tensorDesc->name, tensorName) == 0) {
      return tensorDesc->name;
    }
  }
  return NULL;
}

static bool IsConvertTensorNameLegal(const aclmdlDesc *modelDesc, char *tensorName) {
  return (GetRealTensorName(modelDesc, tensorName) == NULL);
}

/**
 * @brief current conversion tensor name illegal needs to be transformed
 *
 * @param  desc [IN]          pointer to modelDesc
 * @param  tensorName [IN]    tensor name
 * @param  outTensor [OUT]    acl_modelId_${id}_input/output_${index}_${random string}
 *
 * @retval Success return ACL_SUCCESS
 * @retval Failure return ACL_ERROR_FAILURE
 */
static aclError TransConvertTensorNameToLegal(const aclmdlDesc *modelDesc, const char *tensorName, char *outTensor) {
  const size_t num = 26U;
  const size_t times = num * num * num;
  size_t nameLen = strlen(tensorName);
  errno_t ret = strcpy_s(outTensor, ACL_MAX_TENSOR_NAME_LEN, tensorName);
  if (ret != EOK) {
    return ACL_ERROR_FAILURE;
  }
  for (size_t i = 0; i < times; ++i) {
    static size_t n = 0;
    char str[IDX_5];
    str[IDX_0] = '_';
    str[IDX_1] = 'a' + (n / num / num) % num;
    str[IDX_2] = 'a' + (n / num) % num;
    str[IDX_3] = 'a' + n % num;
    str[IDX_4] = '\0';
    n++;
    ret = strcpy_s(outTensor + nameLen, ACL_MAX_TENSOR_NAME_LEN - nameLen, str);
    if (ret != EOK) {
      return ACL_ERROR_FAILURE;
    }
    if (IsConvertTensorNameLegal(modelDesc, outTensor)) {
      return ACL_SUCCESS;
    }
  }
  return ACL_ERROR_FAILURE;
}

static size_t NumToStr(size_t n, size_t max, char *s) {
  size_t i = 0;
  size_t j = 0;
  char tmp[IDXMAX];
  do {
    tmp[i++] = n % TEN + '0';
    n /= TEN;
  } while (n);

  if (i < max) {
    while (i > 0) {
      s[j++] = tmp[--i];
    }
    s[j] = '\0';
  }
  return j;
}

static int GenPreName(char *midStr, size_t midLen, bool isIn, char *name) {
  errno_t ret = strcpy_s(name, ACL_MAX_TENSOR_NAME_LEN, ACL_MODEL_PRE);
  if (ret != EOK) {
    return ret;
  }
  size_t tmpLen = ACL_MAX_TENSOR_NAME_LEN - ACL_MODEL_PRE_LEN;
  ret = strcpy_s(name + ACL_MODEL_PRE_LEN, tmpLen, midStr);
  if (ret != EOK) {
    return ret;
  }
  ret = strcpy_s(name + ACL_MODEL_PRE_LEN + midLen, tmpLen - midLen,
                 isIn ? TENSOR_INPUT_STR : TENSOR_OUTPUT_STR);
  if (ret != EOK) {
    return ret;
  }
  return ret;
}

/**
 * @brief Get tensor name to dims with or without realName.
 * If the length of tensor name is larger than 127 characters,
 * need convert tensor name to acl_modelId_${id}_input_${index}_${random string}
 * format, otherwise tensor name will be set to realName.
 *
 * @param  modelDesc [IN]     pointer to modelDesc
 * @param  realName [IN]      tensor name from model parser of aclmdlTensorDesc
 * @param  tensorType [IN]    tensor type, input or output
 * @param  index [IN]         tensor index from user
 * @param  dims [OUT]         tensor dims of aclmdlIODims, open to user
 *
 * @retval Success return ACL_SUCCESS
 * @retval Failure return ACL_ERROR_FAILURE
 */
static aclError GetTensorDescNameToDims(const aclmdlDesc *modelDesc, char *realName,
                                        bool isIn, const size_t index, aclmdlIODims *const dims) {
  if ((strlen(realName) + 1U) > ACL_MAX_TENSOR_NAME_LEN) {
    // Use convertName because realName is too long.
    char midStr[MIDMAX];
    size_t midLen = NumToStr(modelDesc->modelId, MIDMAX, midStr);
    errno_t ret = GenPreName(midStr, midLen, isIn, dims->name);
    if (ret != EOK) {
      return ACL_ERROR_FAILURE;
    }
    char idxStr[IDXMAX];
    (void)NumToStr(index, IDXMAX, idxStr);
    size_t tLen = (isIn ? TENSOR_INPUT_STR_LEN : TENSOR_OUTPUT_STR_LEN);
    ret = strcpy_s(dims->name + ACL_MODEL_PRE_LEN + midLen + tLen,
                   ACL_MAX_TENSOR_NAME_LEN - ACL_MODEL_PRE_LEN - midLen - tLen, idxStr);
    if (ret != EOK) {
      return ACL_ERROR_FAILURE;
    }
    ACL_LOG_INFO("RealName is over than %d, use convertName=%s", ACL_MAX_TENSOR_NAME_LEN, dims->name);
    if (IsConvertTensorNameLegal(modelDesc, dims->name)) {
      return ACL_SUCCESS;  // dims->name is convertName
    }
    char convertName[ACL_MAX_TENSOR_NAME_LEN];
    aclError res = TransConvertTensorNameToLegal(modelDesc, dims->name, convertName);
    if (res != ACL_SUCCESS) {
      ACL_LOG_WARN("Use name %s may has conflict risk", convertName);
    } else {
      // dims->name is convertName with random string. If the name of the
      // converted tensor conflicts with the name of the existing tensor in the
      // model, "_${random string}" will be added to the tail of the converted
      // name, otherwise the random string will not be added.
      errno_t ert = strcpy_s(dims->name, ACL_MAX_TENSOR_NAME_LEN, convertName);
      if (ert != EOK) {
        return ACL_ERROR_FAILURE;
      }
    }
  } else {
    errno_t ret = strcpy_s(dims->name, ACL_MAX_TENSOR_NAME_LEN, realName);  // dims->name is realName
    if (ret != EOK) {
      return ACL_ERROR_FAILURE;
    }
  }
  return ACL_SUCCESS;
}

static aclError GetDims(const aclmdlDesc *modelDesc, const size_t index, aclmdlIODims *const dims, bool isIn) {
  if (modelDesc == NULL || dims == NULL) {
    ACL_LOG_ERROR("modelDesc/dims must not be NULL");
    return ACL_ERROR_INVALID_PARAM;
  }

  Vector desc = (isIn == true) ? modelDesc->inputDesc : modelDesc->outputDesc;
  size_t descSize = VectorSize(&desc);
  if (index >= descSize) {
    ACL_LOG_ERROR("index[%zu] is over than %zu", index, descSize);
    return ACL_ERROR_INVALID_PARAM;
  }

  aclmdlTensorDesc *tensorDesc = (aclmdlTensorDesc *)VectorAt(&desc, index);
  aclError ret = GetTensorDescNameToDims(modelDesc, tensorDesc->name, isIn, index, dims);
  if (ret != ACL_SUCCESS) {
    ACL_LOG_ERROR("Get name failed, ret=%d", ret);
    return ret;
  }

  size_t dimSize = VectorSize(&tensorDesc->dims);
  if (dimSize > ACL_MAX_DIM_CNT) {
    ACL_LOG_INNER_ERROR("DimsSize[%zu] is over than %d", dimSize, ACL_MAX_DIM_CNT);
    return ACL_ERROR_STORAGE_OVER_LIMIT;
  }
  dims->dimCount = dimSize;
  for (size_t i = 0U; i < dimSize; ++i) {
    dims->dims[i] = *((int64_t *)VectorAt(&tensorDesc->dims, i));
  }
  return ACL_SUCCESS;
}

/**
 * @brief In order to simplify the process of name check, first concatenate string
 * like acl_modelId_${id from modelDesc}_${input/output from tensorName of user}_,
 * e.g., acl_modelId_0_input_, and then compare it with the same length of
 * tensorName to check if it is valid, it will return NULL if concatenate string
 * isn't satisfy condition. If the name is valid, get the corresponding tensor
 * name according to the idx.
 *
 * @param  modelDesc [IN]     pointer to modelDesc
 * @param  tensorName [IN]    tensor name from user
 *
 * @retval Pointer to real tensor name
 * @retval Failure return NULL
 */
static const char *TransTensorNameToReal(const aclmdlDesc *modelDesc, const char *tensorName) {
  char *input = strstr(tensorName, "_input_");
  char *output = strstr(tensorName, "_output_");
  if (input == NULL && output == NULL) {
    return NULL;
  }
  bool isIn = (input != NULL);
  char usrName[ACL_MAX_TENSOR_NAME_LEN];
  char midStr[MIDMAX];
  size_t midLen = NumToStr(modelDesc->modelId, MIDMAX, midStr);
  errno_t ret = GenPreName(midStr, midLen, isIn, usrName);
  if (ret != EOK) {
    return NULL;
  }
  size_t nameLen = strlen(usrName);
  uint64_t idx = 0;
  bool valid = false;
  size_t strLen = strlen(tensorName);
  uint8_t base = 10U;
  for (size_t i = nameLen; i < strLen; ++i) {
    if (tensorName[i] >= '0' && tensorName[i] <= '9') {
      idx = idx * base + (tensorName[i] - '0');
      valid = true;
    } else {
      break;
    }
  }

  if (!valid || strncmp(usrName, tensorName, nameLen) != 0) {
    return NULL;
  }

  // idx valid check is in VectorAt
  Vector desc = isIn ? modelDesc->inputDesc : modelDesc->outputDesc;
  aclmdlTensorDesc *tensorDesc = (aclmdlTensorDesc *)VectorAt(&desc, idx);
  if (tensorDesc != NULL) {
    return tensorDesc->name;
  }

  ACL_LOG_INNER_ERROR("There is no input_%lu/output_%lu in name[%s]", idx, idx, tensorName);
  return NULL;
}

static aclError GetIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *idx, bool isIn) {
  if (modelDesc == NULL || name == NULL || idx == NULL) {
    ACL_LOG_ERROR("modelDesc/name/index must not be NULL");
    return ACL_ERROR_INVALID_PARAM;
  }
  Vector desc = (isIn == true) ? modelDesc->inputDesc : modelDesc->outputDesc;
  size_t size = VectorSize(&desc);
  for (size_t i = 0U; i < size; ++i) {
    aclmdlTensorDesc *tensorDesc = (aclmdlTensorDesc *)VectorAt(&desc, i);
    if (strcmp(tensorDesc->name, name) == 0) {
      *idx = i;
      ACL_LOG_DEBUG("Get name[%s] by index[%zu] success", name, *idx);
      return ACL_SUCCESS;
    }
  }

  ACL_LOG_INNER_ERROR("There is no name[%s]", name);
  return ACL_ERROR_INVALID_PARAM;
}

aclmdlDesc *aclmdlCreateDesc() {
  aclmdlDesc *desc = (aclmdlDesc *)mmMalloc(sizeof(aclmdlDesc));
  if (desc == NULL) {
    return NULL;
  }
  InitVector(&desc->inputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->inputDesc, ModelTensorDescDestroy);
  InitVector(&desc->outputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->outputDesc, ModelTensorDescDestroy);
  return desc;
}

aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc) {
  if (modelDesc == NULL) {
    ACL_LOG_ERROR("modelDesc must not be NULL");
    return ACL_ERROR_INVALID_PARAM;
  }
  ModelDestroyDesc(modelDesc);
  mmFree(modelDesc);
  modelDesc = NULL;
  return ACL_SUCCESS;
}

aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId) {
  if (modelDesc == NULL) {
    ACL_LOG_ERROR("ModelDesc must not be NULL");
    return ACL_ERROR_INVALID_PARAM;
  }

  ModelInOutInfo info;
  Status ret = GetModelDescInfo(modelId, &info);
  if (ret != SUCCESS) {
    ACL_LOG_CALL_ERROR("Ge failed, modelId[%u], ret[%u]", modelId, ret);
    return ACL_ERROR_GE_FAILURE;
  }

  aclError aclRet = TransOmInfo2ModelDesc(&info, modelDesc);
  if (aclRet != ACL_SUCCESS) {
    return aclRet;
  }
  modelDesc->modelId = modelId;
  DestroyModelInOutInfo(&info);
  return ACL_SUCCESS;
}

aclError aclmdlGetDescFromFile(aclmdlDesc *modelDesc, const char *modelPath) {
  if (modelDesc == NULL || modelPath == NULL) {
    ACL_LOG_ERROR("ModelDesc/modelPath must not be NULL");
    return ACL_ERROR_INVALID_PARAM;
  }
  ModelData modelData = {0};
  Status ret = LoadDataFromFile(modelPath, &modelData);
  if (ret != SUCCESS) {
    ACL_LOG_CALL_ERROR("Load file[%s] failed, ge ret[%u]", modelPath, ret);
    return ACL_ERROR_GE_FAILURE;
  }
  aclError result = GetModelDescInfoInner(modelDesc, &modelData);
  FreeModelData(&modelData);
  return result;
}

aclError aclmdlGetDescFromMem(aclmdlDesc *modelDesc, const void *model,
                              size_t modelSize) {
  if (modelDesc == NULL || model == NULL) {
    ACL_LOG_ERROR("modelDesc/model must not be NULL");
    return ACL_ERROR_INVALID_PARAM;
  }

  ModelData modelData = {0};
  modelData.modelData = (void *)model;
  modelData.modelLen = modelSize;
  aclError result = GetModelDescInfoInner(modelDesc, &modelData);
  if (result != ACL_SUCCESS) {
    return result;
  }
  return ACL_SUCCESS;
}

size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc) {
  if (modelDesc == NULL) {
    ACL_LOG_INNER_ERROR("modelDesc is NULL");
    return 0U;
  }
  return VectorSize(&modelDesc->inputDesc);
}

size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc) {
  if (modelDesc == NULL) {
    ACL_LOG_INNER_ERROR("modelDesc is NULL");
    return 0U;
  }
  return VectorSize(&modelDesc->outputDesc);
}

__attribute__ ((noinline)) static aclmdlTensorDesc *mdlGetTensorDesc(const aclmdlDesc *mdlDesc, size_t idx, bool isIn) {
  Vector desc = isIn ? mdlDesc->inputDesc : mdlDesc->outputDesc;
  aclmdlTensorDesc *tensorDesc = (aclmdlTensorDesc *)VectorAt(&desc, idx);
  if (tensorDesc == NULL) {
    ACL_LOG_ERROR("%s is invalid, idx[%zu]", isIn ? "Input" : "Output", idx);
  }
  return tensorDesc;
}

size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index) {
  if (modelDesc == NULL) {
    return 0U;
  }
  aclmdlTensorDesc *tensorDesc = mdlGetTensorDesc(modelDesc, index, true);
  return (tensorDesc == NULL) ? 0U : tensorDesc->size;
}

size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index) {
  if (modelDesc == NULL) {
    return 0U;
  }
  aclmdlTensorDesc *tensorDesc = mdlGetTensorDesc(modelDesc, index, false);
  return (tensorDesc == NULL) ? 0U : tensorDesc->size;
}

aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
  return GetDims(modelDesc, index, dims, true);
}

aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
  return GetDims(modelDesc, index, dims, false);
}

// User can call aclmdlGetTensorRealName, pass in the converted name from
// aclmdlIODims using aclmdlGetInputDims/aclmdlGetOutputDims and then get the
// original name of tensor. If the original name of tensor is passed to this
// interface, the original name of tensor is still obtained.
const char *aclmdlGetTensorRealName(const aclmdlDesc *modelDesc, const char *name) {
  if (modelDesc == NULL || name == NULL) {
    ACL_LOG_ERROR("modelDesc/name must not be NULL");
    return NULL;
  }

  // when the name from user can be found in the modelDesc tensor name, direct return this name.
  const char *realTensorName = GetRealTensorName(modelDesc, name);
  if (realTensorName != NULL) {
      return realTensorName;
  }

  // check if name like acl_modelId_${id}_input_${index}_${random string} format.
  realTensorName = TransTensorNameToReal(modelDesc, name);
  if (realTensorName != NULL) {
      return realTensorName;
  }

  ACL_LOG_INNER_ERROR("Name[%s] is invalid", name);
  return NULL;
}

const char *aclmdlGetInputNameByIndex(const aclmdlDesc *modelDesc, size_t index) {
  if (modelDesc == NULL) {
    return "";
  }
  aclmdlTensorDesc *tensorDesc = mdlGetTensorDesc(modelDesc, index, true);
  return (tensorDesc == NULL) ? "" : tensorDesc->name;
}

const char *aclmdlGetOutputNameByIndex(const aclmdlDesc *modelDesc, size_t index) {
  if (modelDesc == NULL) {
    return "";
  }
  aclmdlTensorDesc *tensorDesc = mdlGetTensorDesc(modelDesc, index, false);
  return (tensorDesc == NULL) ? "" : tensorDesc->name;
}

aclFormat aclmdlGetInputFormat(const aclmdlDesc *modelDesc, size_t index) {
  if (modelDesc == NULL) {
    return ACL_FORMAT_UNDEFINED;
  }
  aclmdlTensorDesc *tensorDesc = mdlGetTensorDesc(modelDesc, index, true);
  return (tensorDesc == NULL) ? ACL_FORMAT_UNDEFINED : tensorDesc->format;
}

aclFormat aclmdlGetOutputFormat(const aclmdlDesc *modelDesc, size_t index) {
  if (modelDesc == NULL) {
    return ACL_FORMAT_UNDEFINED;
  }
  aclmdlTensorDesc *tensorDesc = mdlGetTensorDesc(modelDesc, index, false);
  return (tensorDesc == NULL) ? ACL_FORMAT_UNDEFINED : tensorDesc->format;
}

aclDataType aclmdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index) {
  if (modelDesc == NULL) {
    return ACL_DT_UNDEFINED;
  }
  aclmdlTensorDesc *tensorDesc = mdlGetTensorDesc(modelDesc, index, true);
  return (tensorDesc == NULL) ? ACL_DT_UNDEFINED : tensorDesc->dataType;
}

aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index) {
  if (modelDesc == NULL) {
    return ACL_DT_UNDEFINED;
  }
  aclmdlTensorDesc *tensorDesc = mdlGetTensorDesc(modelDesc, index, false);
  return (tensorDesc == NULL) ? ACL_DT_UNDEFINED : tensorDesc->dataType;
}

aclError aclmdlGetInputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index) {
  return GetIndexByName(modelDesc, name, index, true);
}

aclError aclmdlGetOutputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index) {
  return GetIndexByName(modelDesc, name, index, false);
}

#if defined(__cplusplus)
}
#endif
