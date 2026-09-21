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
#include "model_executor.h"
#include "model_loader.h"
#include "model_desc.h"
#include "model_manager.h"
#include "model_parse.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "framework/executor_c/ge_executor.h"

enum GeInitStatus {
  INIT_STATUS_UNINITIALIZED,
  INIT_STATUS_INTERMEDIATE,
  INIT_STATUS_INITIALIZED
};
static mmAtomicType g_isInited = INIT_STATUS_UNINITIALIZED;

Status GeInitialize(void) {
  uint32_t expected = INIT_STATUS_UNINITIALIZED;
  bool flag = mmCompareAndSwap(&g_isInited, expected, INIT_STATUS_INTERMEDIATE);
  if (!flag) {
    return SUCCESS;
  }
  InitGeModelDescManager();
  mmValueStore(&g_isInited, INIT_STATUS_INITIALIZED);
  GELOGI("init ge executor over");
  return SUCCESS;
}

Status GeFinalize(void) {
  uint32_t expected = INIT_STATUS_INITIALIZED;
  bool flag = mmCompareAndSwap(&g_isInited, expected, INIT_STATUS_INTERMEDIATE);
  if (!flag) {
    return SUCCESS;
  }
  DeInitGeModelDescManager();
  mmValueStore(&g_isInited, INIT_STATUS_UNINITIALIZED);
  GELOGI("deinit ge executor over");
  return SUCCESS;
}

Status LoadDataFromFile(const char *modelPath, ModelData *data) {
  if (g_isInited != INIT_STATUS_INITIALIZED) {
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  const Status ret = LoadOfflineModelFromFile(modelPath, data);
  if (ret != SUCCESS) {
    FreeModelData(data);
    return ret;
  }
  return SUCCESS;
}

Status GeLoadModelFromData(uint32_t *modelId, const ModelData *data) {
  if (g_isInited != INIT_STATUS_INITIALIZED) {
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  if (data == NULL) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "para is invalid");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  const Status ret = LoadOfflineModelFromData(modelId, data);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[load][model] from data failed, modelId[%d]", *modelId);
    return ret;
  }
  return SUCCESS;
}

Status GetMemAndWeightSize(const char *fileName, size_t *workSize, size_t *weightSize) {
  if (g_isInited != INIT_STATUS_INITIALIZED) {
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  ModelData model = {0};
  Status ret = LoadDataFromFile(fileName, &model);
  if ((ret != SUCCESS) || (model.modelData == NULL)) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[load][data] from file failed ret[%d]", ret);
    return ret;
  }
  ret = GetModelMemAndWeightSize(&model, workSize, weightSize);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "get workspace and weight size failed ret[%d]", ret);
  }
  FreeModelData(&model);
  return ret;
}

Status GetPartitionSize(const char *fileName, GePartitionSize *mdlPartitionSize) {
  if (g_isInited != INIT_STATUS_INITIALIZED) {
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  ModelData modelData = {0};
  Status ret = LoadDataFromFile(fileName, &modelData);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "Load file[%s] failed ret[%u]", fileName, ret);
    return ret;
  }
  ret = GetModelPartitionSize(&modelData, mdlPartitionSize);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "get workSize and partition sizes failed ret[%u]", ret);
  }
  FreeModelData(&modelData);
  return ret;
}

Status UnloadModel(uint32_t modelId) {
  if (g_isInited != INIT_STATUS_INITIALIZED) {
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  return UnloadOfflineModel(modelId);
}

Status ExecModel(uint32_t modelId, ExecHandleDesc *execDesc, bool sync,
                 const InputData *inputData, OutputData *outputData) {
  if (g_isInited != INIT_STATUS_INITIALIZED) {
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  Status ret = ModelExecuteInner(modelId, execDesc, sync, inputData, outputData);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[model] exec failed, modelId[%u].", modelId);
    return ret;
  }

  GELOGI("[model] exec success, sync[%d], modelId[%u]", sync, modelId);
  return SUCCESS;
}

Status GetModelDescInfoFromMem(const ModelData *modelData, ModelInOutInfo *info) {
  ModelPartition partition;
  partition.type = MODEL_INOUT_INFO;
  Status ret = GetPartInfoFromModel(modelData, &partition);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "get part info from model failed.");
    return ret;
  }
  InitModelInOutInfo(info);
  ret = ParseModelIoDescInfo(modelData, partition.data, partition.size, info);
  if (modelData->flag == NEED_READ_FROM_FD) {
    mmFree(partition.data);
    partition.data = NULL;
  }
  if (ret != SUCCESS) {
    DeInitModelInOutInfo(info);
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "parse model io partition failed.");
    return ret;
  }
  return SUCCESS;
}

static Status ModelDescInfoDepthCopy(GeModelDesc *mdlDesc, bool isInput, ModelInOutInfo *info) {
  Vector *descVector = isInput ? &info->input_desc : &info->output_desc;
  Vector *mdlDescVector = isInput ? &mdlDesc->ioInfo.input_desc : &mdlDesc->ioInfo.output_desc;
  size_t size;
  if (isInput) {
    size = VectorSize(&mdlDesc->ioInfo.input_desc);
  } else {
    size = VectorSize(&mdlDesc->ioInfo.output_desc);
  }
  ResizeModelInOutTensorDesc(size, descVector);
  if (VectorSize(descVector) != size) {
    return ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED;
  }
  Status ret = SUCCESS;
  for (size_t i = 0UL; i < size; ++i) {
    ModelInOutTensorDesc *desc = VectorAt(descVector, i);
    ModelInOutTensorDesc *innerDesc = VectorAt(mdlDescVector, i);
    desc->size = innerDesc->size;
    desc->dataType = innerDesc->dataType;
    desc->format = innerDesc->format;
    GELOGI("desc size[%zu] type[%d] format[%d]", desc->size, desc->dataType, desc->format);
    if (innerDesc->name != NULL) {
      uint32_t len = (uint32_t)strlen(innerDesc->name) + 1;
      desc->name = (char *)mmMalloc(len);
      if (desc->name == NULL) {
        return ACL_ERROR_GE_MEMORY_ALLOCATION;
      }
      // dst_len == src_len
      (void)strcpy_s(desc->name, len, innerDesc->name);
    }
    size_t dims_size = VectorSize(&innerDesc->dims);
    if (ReSizeVector(&desc->dims, dims_size) != dims_size) {
      ret = ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      break;
    }
    for (size_t index = 0UL; index < dims_size; ++index) {
      int64_t dim = *((int64_t *)(VectorAt(&innerDesc->dims, index)));
      *((int64_t *)VectorAt(&desc->dims, index)) = dim;
    }
  }
  if (ret != SUCCESS) {
    return ret;
  }
  GELOGI("copy type[%d] info success", isInput);
  return SUCCESS;
}

Status GetModelDescInfo(uint32_t modelId, ModelInOutInfo *info) {
  if (g_isInited != INIT_STATUS_INITIALIZED) {
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  GeModelDesc *mdlDesc = GetModelDescRef(modelId);
  if (mdlDesc == NULL) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "get modelDesc failed, modelId[%u]", modelId);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  InitModelInOutInfo(info);
  // process inOutInfo deep copy prevents unload model.
  Status ret = ModelDescInfoDepthCopy(mdlDesc, true, info);
  if (ret != SUCCESS) {
    DeInitModelInOutInfo(info);
    ReleaseModelDescRef(mdlDesc);
    return ret;
  }

  ret = ModelDescInfoDepthCopy(mdlDesc, false, info);
  if (ret != SUCCESS) {
    DeInitModelInOutInfo(info);
    ReleaseModelDescRef(mdlDesc);
    return ret;
  }
  ReleaseModelDescRef(mdlDesc);
  GELOGI("get model info success, modelId[%u]", modelId);
  return SUCCESS;
}

void DestroyModelInOutInfo(ModelInOutInfo *info) {
  DeInitVector(&info->input_desc);
  DeInitVector(&info->output_desc);
}