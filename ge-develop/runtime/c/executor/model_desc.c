/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_desc.h"
#include "model_parse.h"
#include "runtime/mem.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
Status CheckOmHeadWithMem(const ModelData *model_data) {
  if (model_data->modelData == NULL) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "model is invalid.");
    return ACL_ERROR_GE_LOAD_MODEL;
  }

  ModelFileHeader *header = (ModelFileHeader *)model_data->modelData;
  if (header->magic != MODEL_FILE_MAGIC_NUM) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "om data is invalid.");
    return ACL_ERROR_GE_LOAD_MODEL;
  }
  return SUCCESS;
}

Status GetPartInfoFromModel(const ModelData *modelData, ModelPartition *partition) {
  uint8_t *base_addr = (uint8_t *)modelData->modelData;
  uint32_t partitionNum = 0U;
  Status ret = ParsePartitionPreProcess(modelData, &partitionNum);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "model parse failed. ret[%u]", ret);
    return ret;
  }

  size_t offset = sizeof(ModelFileHeader) + sizeof(ModelPartitionTable);
  size_t base_offset = offset + sizeof(ModelPartitionMemInfo) * partitionNum;
  uint64_t validLen = modelData->modelLen - base_offset;

  for (size_t i = 0UL; i < partitionNum; ++i) {
    ModelPartitionMemInfo *partitionMemInfo = (ModelPartitionMemInfo *)(base_addr + offset);
    if ((partitionMemInfo->mem_size == 0) || ((partitionMemInfo->mem_size + partitionMemInfo->mem_offset) > validLen)) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "partition type %d, mem_size is %zu offset is %zu, model_len is %zu, invalid",
             partitionMemInfo->type, (size_t)partitionMemInfo->mem_size, (size_t)partitionMemInfo->mem_offset,
             (size_t)modelData->modelLen);
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    if (partitionMemInfo->type == partition->type) {
      partition->size = (uint32_t)partitionMemInfo->mem_size;
      if (modelData->flag == NEED_READ_FROM_FD) {
        partition->data = (uint8_t *)mmMalloc(sizeof(uint8_t) * partition->size);
        if (partition->data == NULL) {
          return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
        }
        ret = ReadDataFromFd(modelData->fd, base_offset + partitionMemInfo->mem_offset, partitionMemInfo->mem_size,
                             partition->data);
        if (ret != SUCCESS) {
          mmFree(partition->data);
          partition->data = NULL;
          return ret;
        }
      } else {
        partition->data = base_addr + base_offset + partitionMemInfo->mem_offset;
      }
      return SUCCESS;
    }
    offset += sizeof(ModelPartitionMemInfo);
  }
  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "cant find partition[%d]", partition->type);
  return ACL_ERROR_GE_INTERNAL_ERROR;
}

Status GetModelMemAndWeightSize(const ModelData *modelData, size_t *workSize, size_t *weightSize) {
  ModelPartition partition;
  partition.type = PRE_MODEL_DESC;
  Status ret = GetPartInfoFromModel(modelData, &partition);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "get part info from model failed.");
    return ret;
  }
  ModelDesc *model_desc = (ModelDesc *)((uint8_t *)partition.data);
  *workSize = model_desc->workspace_size;
  *weightSize = model_desc->weight_size;
  if (modelData->flag == NEED_READ_FROM_FD) {
    mmFree(partition.data);
    partition.data = NULL;
  }
  return SUCCESS;
}

static Status GetModelFifoSize(const ModelData *modelData, size_t *fifoSize) {
  ModelPartition partition;
  partition.type = PRE_MODEL_DESC_EXTEND;
  Status ret = GetPartInfoFromModel(modelData, &partition);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "get part info from model failed.");
    return ret;
  }

  GeModelDesc mdlDesc = {0};
  ret = ParseModelDescExtend(modelData, partition.data, (size_t)partition.size, &mdlDesc);
  if (ret == SUCCESS) {
    *fifoSize = mdlDesc.fifoInfo.totalSize;
  }

  if (mdlDesc.fifoInfo.fifoBaseAddr != NULL) {
    (void)rtFree(mdlDesc.fifoInfo.fifoBaseAddr);
    mdlDesc.fifoInfo.fifoBaseAddr = NULL;
  }
  DeInitModelFifoInfo(&(mdlDesc.fifoInfo));

  if (modelData->flag == NEED_READ_FROM_FD) {
    mmFree(partition.data);
    partition.data = NULL;
  }
  return ret;
}

static Status GetModelAllPartitionSize(const ModelData *modelData,
                                      uint32_t partitionNum, GePartitionSize *mdlPartitionSize) {
  size_t offset = sizeof(ModelFileHeader) + sizeof(ModelPartitionTable);
  size_t base_offset = offset + partitionNum * sizeof(ModelPartitionMemInfo);
  uint64_t validLen = modelData->modelLen - base_offset;

  for (uint32_t i = 0; i < partitionNum; i++) {
    ModelPartitionMemInfo *partitionMemInfo = (ModelPartitionMemInfo *)((uint8_t *)modelData->modelData + offset);
    if ((partitionMemInfo->mem_size == 0) || (partitionMemInfo->mem_offset + partitionMemInfo->mem_size > validLen)) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR,
             "partition type %d, mem_size is %zu offset is %zu, model_len is %zu, invalid",
             partitionMemInfo->type, (size_t)partitionMemInfo->mem_size, (size_t)partitionMemInfo->mem_offset,
             (size_t)modelData->modelLen);
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    switch (partitionMemInfo->type) {
      case TBE_KERNELS:
        mdlPartitionSize->kernelSize = (size_t)partitionMemInfo->mem_size;
        break;
      case TASK_PARAM:
        mdlPartitionSize->kernelArgsSize = (size_t)partitionMemInfo->mem_size;
        break;
      case STATIC_TASK_DESC:
        mdlPartitionSize->staticTaskSize = (size_t)partitionMemInfo->mem_size;
        break;
      case DYNAMIC_TASK_DESC:
        mdlPartitionSize->dynamicTaskSize = (size_t)partitionMemInfo->mem_size;
        break;
      case PRE_MODEL_DESC: {
        Status ret = GetModelMemAndWeightSize(modelData, &mdlPartitionSize->workSize, &mdlPartitionSize->weightSize);
        if (ret != SUCCESS) {
          return ret;
        }
        mdlPartitionSize->modelDescSize = sizeof(ModelDesc);
        break;
      }
      case PRE_MODEL_DESC_EXTEND: {
        Status ret = GetModelFifoSize(modelData, &(mdlPartitionSize->fifoSize));
        if (ret != SUCCESS) {
          return ret;
        }
        break;
      }
      default:
        break;
    }
    offset += sizeof(ModelPartitionMemInfo);
  }
  return SUCCESS;
}

Status GetModelPartitionSize(const ModelData *modelData, GePartitionSize *mdlPartitionSize) {
  uint32_t partitionNum;
  Status ret = ParsePartitionPreProcess(modelData, &partitionNum);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = GetModelAllPartitionSize(modelData, partitionNum, mdlPartitionSize);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "get partitionSize failed. ret = %u", ret);
  }
  return ret;
}

