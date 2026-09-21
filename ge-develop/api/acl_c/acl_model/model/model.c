/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_mdl.h"
#include "acl/acl_base.h"
#include "framework/executor_c/ge_executor.h"
#include "model_desc_internal.h"
#include "log_inner.h"
#include "runtime/rt.h"
#include "model_config.h"

static void SetPartFromHandle(const aclmdlConfigHandle *handle, ModelData *data) {
  data->part.weightPtr = handle->exeOMInfo.weightPtr;
  data->part.weightSize = handle->exeOMDesc.weightSize;
  data->part.modelDescPtr = handle->exeOMInfo.modelDescPtr;
  data->part.modelDescSize = handle->exeOMDesc.modelDescSize;
  data->part.kernelPtr = handle->exeOMInfo.kernelPtr;
  data->part.kernelSize = handle->exeOMDesc.kernelSize;
  data->part.paramPtr = handle->exeOMInfo.kernelArgsPtr;
  data->part.paramSize = handle->exeOMDesc.kernelArgsSize;
  data->part.taskPtr = handle->exeOMInfo.staticTaskPtr;
  data->part.taskSize = handle->exeOMDesc.staticTaskSize;
  data->part.dynTaskPtr = handle->exeOMInfo.dynamicTaskPtr;
  data->part.dynTaskSize = handle->exeOMDesc.dynamicTaskSize;
  data->part.fifoPtr = handle->exeOMInfo.fifoTaskPtr;
  data->part.fifoSize = handle->exeOMDesc.fifoTaskSize;
}

static aclError ModelLoadFromFileWithMem(const aclmdlConfigHandle *handle, uint32_t *const modelId) {
  ModelData data = {0};
  data.priority = handle->priority;
  data.memType = handle->memType;
  char *modelPath = handle->loadPath;
  SetPartFromHandle(handle, &data);
  Status ret = LoadDataFromFile(modelPath, &data);
  if (ret != SUCCESS) {
    ACL_LOG_CALL_ERROR("load data from file[%s] failed, ret[%u]", modelPath, ret);
    return ret;
  }
  uint32_t id = 0U;
  ret = GeLoadModelFromData(&id, &data);
  if (ret != SUCCESS) {
      FreeModelData(&data);
      ACL_LOG_CALL_ERROR("load model from data failed, ret[%u]", ret);
      return ret;
  }

  *modelId = id;
  FreeModelData(&data);
  return ACL_SUCCESS;
}

static aclError ModelLoadFromMemWithMem(const aclmdlConfigHandle *handle, uint32_t *const modelId) {
  size_t modelSize = handle->mdlSize;
  void *model = handle->mdlAddr;
  if ((modelSize == 0U) || (model == NULL)) {
    ACL_LOG_INNER_ERROR("param[%zu] is invalid.", modelSize);
    return ACL_ERROR_INVALID_PARAM;
  }

  ModelData data;
  data.modelData = model;
  data.modelLen = (uint64_t)modelSize;
  data.priority = handle->priority;
  data.memType = handle->memType;
  data.fd = NULL;
  data.flag = NO_NEED_READ_FROM_FD;
  SetPartFromHandle(handle, &data);
  uint32_t id = 0U;
  Status ret = GeLoadModelFromData(&id, &data);
  if (ret != SUCCESS) {
      ACL_LOG_CALL_ERROR("load model from data failed, ret[%u].", ret);
      return ret;
  }

  *modelId = id;
  return ACL_SUCCESS;
}

static aclError SetExecHandle(const aclmdlExecConfigHandle *handle, aclrtStream stream, ExecHandleDesc *execDesc) {
  void *workPtr = NULL;
  size_t workSize = 0U;
  if (handle != NULL) {
    if (handle->workPtr != NULL) {
      workPtr = handle->workPtr;
      workSize = handle->workSize;
    }
  }
  if ((workPtr == NULL) && (stream != NULL)) {
    (void)rtStreamGetWorkspace(stream, &workPtr, &workSize);
  }
  if (workPtr == NULL) {
    ACL_LOG_ERROR("not set workSpacePtr.");
    return ACL_ERROR_INVALID_PARAM;
  }
  execDesc->stream = stream;
  execDesc->workPtr = workPtr;
  execDesc->workSize = workSize;
  if (handle != NULL) {
    execDesc->mpamId = handle->mpamId;
    execDesc->aicQos = handle->aicQos;
    execDesc->aicOst = handle->aicOst;
    execDesc->mecTimeThreshHold = handle->mecTimeThreshHold;
  }
  return ACL_SUCCESS;
}

static aclError ModelExecute(const uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output,
                             const bool sync, aclrtStream stream, const aclmdlExecConfigHandle *handle) {
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(input);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(output);
  ExecHandleDesc execDesc = {0};
  aclError err = SetExecHandle(handle, stream, &execDesc);
  if (err != ACL_SUCCESS) {
    return err;
  }

  const char *syncMode = (sync == true) ? "sync" : "async";
  const Status ret = ExecModel(modelId, &execDesc, sync, (const InputData *)input, (OutputData *)output);
  if (ret != SUCCESS) {
    ACL_LOG_CALL_ERROR("%s execute model failed, ret[%u], modelId[%u]", syncMode, ret, modelId);
    return ret;
  }
  ACL_LOG_INFO("success execute, modelId[%u] mode[%d]", modelId, sync);
  return ACL_SUCCESS;
}

aclError aclmdlLoadWithConfig(const aclmdlConfigHandle *handle, uint32_t *modelId) {
  if ((handle == NULL) || (modelId == NULL)) {
    ACL_LOG_ERROR("param must not be NULL.");
    return ACL_ERROR_INVALID_PARAM;
  }
  if (!CheckMdlConfigHandle(handle)) {
    return ACL_ERROR_INVALID_PARAM;
  }
  aclError ret;
  switch (handle->mdlLoadType) {
    case ACL_MDL_LOAD_FROM_FILE:
    case ACL_MDL_LOAD_FROM_FILE_WITH_MEM: {
      ret = ModelLoadFromFileWithMem(handle, modelId);
      break;
    }
    case ACL_MDL_LOAD_FROM_MEM:
    case ACL_MDL_LOAD_FROM_MEM_WITH_MEM: {
      ret = ModelLoadFromMemWithMem(handle, modelId);
      break;
    }
    default:
      ACL_LOG_INNER_ERROR("load type[%zu] is invalid, it should be in [%d, %d]",
          handle->mdlLoadType, ACL_MDL_LOAD_FROM_FILE, ACL_MDL_LOAD_FROM_MEM_WITH_MEM);
      return ACL_ERROR_INVALID_PARAM;
  }
  if (ret != ACL_SUCCESS) {
    ACL_LOG_INNER_ERROR("model load failed, type[%zu].", handle->mdlLoadType);
    return ret;
  }
  return ACL_SUCCESS;
}

aclError aclmdlExecuteV2(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream,
                         const aclmdlExecConfigHandle *handle) {
  return ModelExecute(modelId, input, output, true, stream, handle);
}

aclError aclmdlExecuteAsyncV2(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream,
                              const aclmdlExecConfigHandle *handle) {
  return ModelExecute(modelId, input, output, false, stream, handle);
}

aclError aclmdlUnload(uint32_t modelId) {
  const Status ret = UnloadModel(modelId);
  if (ret != SUCCESS) {
    ACL_LOG_CALL_ERROR("unload model failed, ret[%u], modelId[%u]", ret, modelId);
    return ret;
  }
  return ACL_SUCCESS;
}

aclError aclmdlQuerySize(const char *fileName, size_t *workSize, size_t *weightSize) {
  if ((fileName == NULL) || (workSize == NULL) || (weightSize == NULL)) {
    ACL_LOG_ERROR("param must not be NULL.");
    return ACL_ERROR_INVALID_PARAM;
  }
  const Status ret = GetMemAndWeightSize(fileName, workSize, weightSize);
  if (ret != SUCCESS) {
    ACL_LOG_CALL_ERROR("query size failed, ge ret[%u]", ret);
    return ret;
  }
  return ACL_SUCCESS;
}

aclError aclmdlQueryExeOMDesc(const char *fileName, aclmdlExeOMDesc *mdlPartitionSize) {
  if ((fileName == NULL) || (mdlPartitionSize == NULL)) {
    ACL_LOG_ERROR("param must not be NULL.");
    return ACL_ERROR_INVALID_PARAM;
  }
  const Status ret = GetPartitionSize(fileName, (GePartitionSize *)mdlPartitionSize);
  if (ret != SUCCESS) {
    ACL_LOG_CALL_ERROR("query partition size failed, ge ret[%u]", ret);
    return ret;
  }
  return ACL_SUCCESS;
}

aclmdlDataset *aclmdlCreateDataset() {
  DataSet *dataset = (DataSet *)mmMalloc(sizeof(DataSet));
  if (dataset == NULL) {
    return NULL;
  }
  InitVector(&dataset->blobs, sizeof(DataBlob));
  dataset->io_addr = NULL;
  dataset->io_addr_host = NULL;
  dataset->ioa_size = 0U;
  return (aclmdlDataset *)dataset;
}

aclError aclmdlDestroyDataset(const aclmdlDataset *dataset) {
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dataset);
  DataSet *geDataSet = (DataSet *)dataset;
  DeInitVector(&geDataSet->blobs);
  if (geDataSet->io_addr != NULL) {
    (void)rtFree(geDataSet->io_addr);
  }
  if (geDataSet->io_addr_host != NULL) {
    (void)mmFree(geDataSet->io_addr_host);
  }
  geDataSet->ioa_size = 0U;
  geDataSet->io_addr = NULL;
  geDataSet->io_addr_host = NULL;
  mmFree(geDataSet);
  return ACL_SUCCESS;
}

aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer) {
  if ((dataset == NULL) || (dataBuffer == NULL)) {
    ACL_LOG_ERROR("invalid input args.");
    return ACL_ERROR_INVALID_PARAM;
  }
  DataBuffer *geDatabuffer = (DataBuffer *)dataBuffer;
  DataSet *geDataSet = (DataSet *)dataset;
  DataBlob blob;
  blob.dataBuffer = geDatabuffer;
  if (EmplaceBackVector(&geDataSet->blobs, &blob) == NULL) {
    return ACL_ERROR_INTERNAL_ERROR;
  }
  return ACL_SUCCESS;
}

size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataset) {
  const DataSet *geDataSet = (const DataSet *)dataset;
  if (geDataSet == NULL) {
    ACL_LOG_ERROR("invalid input args.");
    REPORT_INPUT_ERROR(INVALID_NULL_POINTER_MSG, ARRAY("param"), ARRAY("dataset"));
    return 0U;
  }
  return VectorSize(&geDataSet->blobs);
}

aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index) {
  DataSet *geDataSet = (DataSet *)dataset;
  if (geDataSet == NULL) {
    ACL_LOG_ERROR("invalid input args.");
    return NULL;
  }
  DataBlob *blob = VectorAt(&geDataSet->blobs, index);
  if (blob == NULL) {
    ACL_LOG_ERROR("dataset has no data.");
    return NULL;
  }
  return (aclDataBuffer *)blob->dataBuffer;
}