/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include "runtime/mem.h"
#include "model_desc.h"
#include "model_parse.h"

typedef Status (*ParseFromFd)(const ModelData *modelData, uint8_t *data, size_t size, void *appInfo);

Status ReadDataFromFd(mmFileHandle *fd, size_t offset, size_t len, void *dst) {
  if (fd == NULL) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "fd is NULL");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  (void)mmSeekFile(fd, offset, MM_SEEK_FILE_BEGIN);
  size_t readLen = mmReadFile(dst, sizeof(uint8_t), len, fd);
  if ((len == 0) || (readLen != len)) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "Read data from fd failed");
    return ACL_ERROR_GE_LOAD_MODEL;
  }
  return SUCCESS;
}

static Status ParseModelDesc(const ModelData *modelData, size_t offset,
                             uint8_t *data, size_t size, GeModelDesc *mdlDesc) {
  void *dstAddr = modelData->part.modelDescPtr;
  if ((modelData->part.modelDescPtr == NULL) || (modelData->part.modelDescSize < size)) {
    rtError_t rtRet = rtMalloc((void **)&dstAddr, size, mdlDesc->memType, 0);
    if (rtRet != RT_ERROR_NONE) {
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    mdlDesc->innerPtrState = mdlDesc->innerPtrState | INNER_PRE_MODEL_DESC_PTR;
    GELOGD("use innerModelDescPtr.");
  }
  mdlDesc->part.modelDescPtr = dstAddr;

  if (modelData->flag == NEED_READ_FROM_FD) {
    return ReadDataFromFd(modelData->fd, offset, size, dstAddr);
  }

  Status ret = (Status)rtMemcpy(dstAddr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "copy partition to device failed");
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  return SUCCESS;
}

static Status ParseWeightData(const ModelData *modelData, size_t offset,
                              uint8_t *data, size_t size, GeModelDesc *mdlDesc) {
  void *dstAddr = modelData->part.weightPtr;
  if ((modelData->part.weightPtr == NULL) || (modelData->part.weightSize < size)) {
    rtError_t rtRet = rtMalloc((void **)&dstAddr, size, mdlDesc->memType, 0);
    if (rtRet != RT_ERROR_NONE) {
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    mdlDesc->innerPtrState = mdlDesc->innerPtrState | INNER_WEIGHTS_DATA_PTR;
    GELOGD("use innerWeightPtr.");
  }
  mdlDesc->part.weightPtr = dstAddr;

  if (modelData->flag == NEED_READ_FROM_FD) {
    return ReadDataFromFd(modelData->fd, offset, size, dstAddr);
  }

  Status ret = (Status)rtMemcpy(dstAddr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "copy partition to device failed");
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  return SUCCESS;
}

static Status ParseTbeKernels(const ModelData *modelData, size_t offset,
                              uint8_t *data, size_t size, GeModelDesc *mdlDesc) {
  void *dstAddr = modelData->part.kernelPtr;
  if ((modelData->part.kernelPtr == NULL) || (modelData->part.kernelSize < size)) {
    rtError_t rtRet = rtMalloc((void **)&dstAddr, size, mdlDesc->memType, 0);
    if (rtRet != RT_ERROR_NONE) {
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    mdlDesc->innerPtrState = mdlDesc->innerPtrState | INNER_TBE_KERNELS_PTR;
    GELOGD("use innerKernelPtr.");
  }
  mdlDesc->part.kernelPtr = dstAddr;

  if (modelData->flag == NEED_READ_FROM_FD) {
    return ReadDataFromFd(modelData->fd, offset, size, dstAddr);
  }

  Status ret = (Status)rtMemcpy(dstAddr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "copy partition to device failed");
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  return SUCCESS;
}

static Status ParseStaticTaskDesc(const ModelData *modelData, size_t offset,
                                  uint8_t *data, size_t size, GeModelDesc *mdlDesc) {
  void *dstAddr = modelData->part.taskPtr;
  if ((modelData->part.taskPtr == NULL) || (modelData->part.taskSize < size)) {
    rtError_t rtRet = rtMalloc((void **)&dstAddr, size, mdlDesc->memType, 0);
    if (rtRet != RT_ERROR_NONE) {
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    mdlDesc->innerPtrState = mdlDesc->innerPtrState | INNER_STATIC_TASK_DESC_PTR;
    GELOGD("use innerTaskPtr.");
  }
  mdlDesc->part.taskPtr = dstAddr;
  mdlDesc->part.taskSize = size;  // dump needs to use this taskSize from model file

  if (modelData->flag == NEED_READ_FROM_FD) {
    return ReadDataFromFd(modelData->fd, offset, size, dstAddr);
  }

  Status ret = (Status)rtMemcpy(dstAddr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "copy partition to device failed");
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  return SUCCESS;
}

static Status ParseDynamicTaskDesc(const ModelData *modelData, size_t offset,
                                   uint8_t *data, size_t size, GeModelDesc *mdlDesc) {
  void *dstAddr = modelData->part.dynTaskPtr;
  if ((modelData->part.dynTaskPtr == NULL) || (modelData->part.dynTaskSize < size)) {
    rtError_t rtRet = rtMalloc((void **)&dstAddr, size, mdlDesc->memType, 0);
    if (rtRet != RT_ERROR_NONE) {
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    mdlDesc->innerPtrState = mdlDesc->innerPtrState | INNER_DYNAMIC_TASK_DESC_PTR;
    GELOGD("use innerDynTaskPtr.");
  }
  mdlDesc->part.dynTaskPtr = dstAddr;

  if (modelData->flag == NEED_READ_FROM_FD) {
    return ReadDataFromFd(modelData->fd, offset, size, dstAddr);
  }

  Status ret = (Status)rtMemcpy(dstAddr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "copy partition to device failed");
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  return SUCCESS;
}

static Status ParseTaskParam(const ModelData *modelData, size_t offset,
                             uint8_t *data, size_t size, GeModelDesc *mdlDesc) {
  void *dstAddr = modelData->part.paramPtr;
  if ((modelData->part.paramPtr == NULL) || (modelData->part.paramSize < size)) {
    rtError_t rtRet = rtMalloc((void **)&dstAddr, size, mdlDesc->memType, 0);
    if (rtRet != RT_ERROR_NONE) {
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    mdlDesc->innerPtrState = mdlDesc->innerPtrState | INNER_TASK_PARAM_PTR;
    GELOGD("use innerParamPtr.");
  }
  mdlDesc->part.paramPtr = dstAddr;

  if (modelData->flag == NEED_READ_FROM_FD) {
    return ReadDataFromFd(modelData->fd, offset, size, dstAddr);
  }

  Status ret = (Status)rtMemcpy(dstAddr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "copy partition to device failed");
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  return SUCCESS;
}

void ModelInfoDestroy(void *base) {
  ModelInOutTensorDesc *config = (ModelInOutTensorDesc *)base;
  if (config->name != NULL) {
    mmFree(config->name);
    config->name = NULL;
  }
  DeInitVector(&config->dims);
}

static void InitModelInOutTensorDesc(Vector *descVector) {
  InitVector(descVector, sizeof(ModelInOutTensorDesc));
  SetVectorDestroyItem(descVector, ModelInfoDestroy);
}

void InitModelInOutInfo(ModelInOutInfo *info) {
  InitModelInOutTensorDesc(&info->input_desc);
  InitModelInOutTensorDesc(&info->output_desc);
}

void DeInitModelInOutInfo(ModelInOutInfo *info) {
  DeInitVector(&info->input_desc);
  DeInitVector(&info->output_desc);
}

void InitModelFifoInfo(ModelFifoInfo *fifoInfo) {
  fifoInfo->fifoNum = 0U;
  fifoInfo->fifoBaseAddr = NULL;
  fifoInfo->fifoAllAddr = NULL;
  fifoInfo->totalSize = 0U;
}

void DeInitModelFifoInfo(ModelFifoInfo *fifoInfo) {
  if (fifoInfo->fifoAllAddr != NULL) {
    mmFree(fifoInfo->fifoAllAddr);
    fifoInfo->fifoAllAddr = NULL;
  }
}

void ResizeModelInOutTensorDesc(size_t size, Vector *descVector) {
  if (ReSizeVector(descVector, size) != size) {
    return;
  }
  for (size_t i = 0UL; i < size; ++i) {
    ModelInOutTensorDesc *desc = VectorAt(descVector, i);
    desc->size = 0UL;
    desc->dataType = DT_MAX;
    desc->format = FORMAT_MAX;
    desc->name = NULL;
    InitVector(&desc->dims, sizeof(int64_t));
  }
}

static Status GetModelInOutDesc(uint8_t *data, size_t size, bool is_input, ModelInOutInfo *info) {
  size_t offset = 0UL;
  if (!CheckLenValid(size, offset, sizeof(uint32_t))) {
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  uint32_t desc_num = 0; // save desc_num in first 4 bytes as uint32_t, same to .om
  (void)memcpy_s((char *)(&desc_num), sizeof(uint32_t), (char *)data, sizeof(uint32_t));
  GELOGD("desc num is %u, is_input is %d.", desc_num, is_input);
  offset += sizeof(uint32_t);
  ModelTensorDescBaseInfo base_info;
  Vector *descVector = is_input ? &info->input_desc : &info->output_desc;
  ResizeModelInOutTensorDesc(desc_num, descVector);
  if (VectorSize(descVector) != desc_num) {
    return ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED;
  }
  Status ret = SUCCESS;
  for (uint32_t i = 0UL; i < desc_num; ++i) {
    if (!CheckLenValid(size, offset, sizeof(ModelTensorDescBaseInfo))) {
      ret = ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      break;
    }
    (void)memcpy_s((char *)(&base_info), sizeof(ModelTensorDescBaseInfo),
                   (char *)(data + offset), sizeof(ModelTensorDescBaseInfo));
    GELOGD("current index %u, format %d, dataType %d, name len %u,"
           "dims len %u, dimsV2 %u, shapeRange %u.",
           i, base_info.format, base_info.dt, base_info.name_len,
           base_info.dims_len, base_info.dimsV2_len, base_info.shape_range_len);
    offset += sizeof(ModelTensorDescBaseInfo);
    ModelInOutTensorDesc *desc = VectorAt(descVector, i);
    desc->size = base_info.size;
    desc->dataType = base_info.dt;
    desc->format = base_info.format;
    size_t tlvLen = base_info.name_len + base_info.dims_len + base_info.dimsV2_len + base_info.shape_range_len;
    if (!CheckLenValid(size, offset, tlvLen)) {
      ret = ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      break;
    }
    unsigned long long len = base_info.name_len + 1;
    desc->name = (char *)mmMalloc(len);
    if (desc->name == NULL) {
      ret = ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      break;
    }
    ret = (Status)memcpy_s(desc->name, len, (uint8_t *)(data + offset), base_info.name_len);
    if (ret != 0) {
      break;
    }
    desc->name[base_info.name_len] = '\0';
    offset += base_info.name_len;
    size_t dims_size = base_info.dims_len / sizeof(int64_t);
    if (ReSizeVector(&desc->dims, dims_size) != dims_size) {
      ret = ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      break;
    }

    for (uint32_t index = 0UL; index < dims_size; ++index) {
      (void)memcpy_s((char *)(VectorAt(&desc->dims, index)), sizeof(int64_t),
                     (char *)(data + offset), sizeof(int64_t));
      offset += sizeof(int64_t);
    }
    offset += base_info.dimsV2_len + base_info.shape_range_len;
  }
  if (ret != SUCCESS) {
    return ret;
  }
  GELOGD("get model inOut info %d success.", is_input);
  return SUCCESS;
}

static Status GetModelInputDesc(uint8_t *data, size_t size, ModelInOutInfo *info) {
  return GetModelInOutDesc(data, size, true, info);
}

static Status GetModelOutputDesc(uint8_t *data, size_t size, ModelInOutInfo *info) {
  return GetModelInOutDesc(data, size, false, info);
}

Status ParseModelIoDescInfo(const ModelData *modelData, uint8_t *data, size_t size, ModelInOutInfo *info) {
  (void)modelData;
  Status ret = ACL_ERROR_GE_INTERNAL_ERROR;
  size_t offset = 0UL;
  while (offset < size) {
    ModelDescTlvConfig config;
    if (!CheckLenValid(size, offset, sizeof(uint32_t))) {
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    uint32_t type = 0;
    (void)memcpy_s((char *)(&type), sizeof(uint32_t), (char *)(data + offset), sizeof(uint32_t));
    config.type = (int32_t)type;
    offset += sizeof(uint32_t);
    if (!CheckLenValid(size, offset, sizeof(uint32_t))) {
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    uint32_t len = 0;
    (void)memcpy_s((char *)(&len), sizeof(uint32_t), (char *)(data + offset), sizeof(uint32_t));
    config.length = len;
    offset += sizeof(uint32_t);
    GELOGD("type %u, length is %u, total size is %zu", type, len, size);
    config.value = (data + offset);
    if (!CheckLenValid(size, offset, len)) {
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    switch (config.type) {
      case MODEL_INPUT_DESC: {
        ret = GetModelInputDesc(config.value, config.length, info);
        break;
      }
      case MODEL_OUTPUT_DESC: {
        ret = GetModelOutputDesc(config.value, config.length, info);
        break;
      }
      default:
        break;
    }
    if (ret != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "get model inOut info type[%d] failed.", config.type);
      return ret;
    }
    offset += len;
  }
  return SUCCESS;
}

static Status ParserPartionFromFd(const ModelData *modelData, size_t offset,
                                  size_t size, void *appInfo, ParseFromFd parserFunc) {
  uint8_t *data = (uint8_t *)mmMalloc(size * sizeof(uint8_t));
  if (data == NULL) {
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  Status ret = ReadDataFromFd(modelData->fd, offset, size, data);
  if (ret != SUCCESS) {
    mmFree(data);
    data = NULL;
    return ACL_ERROR_GE_LOAD_MODEL;
  }
  ret = parserFunc(modelData, data, size, appInfo);
  if (ret != SUCCESS) {
    mmFree(data);
    data = NULL;
    return ret;
  }
  mmFree(data);
  data = NULL;
  return SUCCESS;
}

static Status ParseModelIoDesc(const ModelData *modelData, size_t offset, uint8_t *data,
                               size_t size, GeModelDesc *mdlDesc) {
  if (modelData->flag == NEED_READ_FROM_FD) {
    return ParserPartionFromFd(modelData, offset, size,
                               (void *)&mdlDesc->ioInfo,
                               (ParseFromFd)ParseModelIoDescInfo);
  }
  return ParseModelIoDescInfo(modelData, data, size, &mdlDesc->ioInfo);
}

Status ParsePartitionPreProcess(const ModelData *modelData, uint32_t *partitionNum) {
  Status ret = CheckOmHeadWithMem(modelData);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "check om head failed.");
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  size_t offset = sizeof(ModelFileHeader);
  size_t tableLen = sizeof(ModelPartitionTable);
  if (!CheckLenValid(modelData->modelLen, offset, tableLen)) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "parse model partition table failed. model_len[%zu],"
           " offset[%zu], sizeof(table)[%zu]",
           (size_t)modelData->modelLen, offset, tableLen);
    return ACL_ERROR_GE_LOAD_MODEL;
  }
  uint8_t *baseAddr = (uint8_t *)modelData->modelData;
  ModelPartitionTable *table = (ModelPartitionTable *)((uint32_t *)(baseAddr + offset));
  *partitionNum = table->num;

  offset += tableLen;
  if (((*partitionNum) > MAX_PARTITION_NUM) ||
      (!CheckLenValid(modelData->modelLen, offset, sizeof(ModelPartitionMemInfo) * (*partitionNum)))) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "parse partition table failed. model_len[%zu],"
           " offset[%zu], sizeof(table)[%zu], num[%u]",
           (size_t)modelData->modelLen, offset, tableLen, *partitionNum);
    return ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED;
  }
  return SUCCESS;
}

static Status ProcFifoInfo(const ModelData *modelData, uint8_t *tlvValue, uint32_t len, GeModelDesc *mdlDesc) {
  uint64_t offset = 0UL;
  ModelFifoInfo *geFifoInfo = &(mdlDesc->fifoInfo);
  if (!CheckLenValid(len, offset, sizeof(struct ModelGlobalDataInfo))) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "proc fifo failed.");
    return ACL_ERROR_GE_LOAD_MODEL;
  }
  struct ModelGlobalDataInfo *tlvFifoInfo = (struct ModelGlobalDataInfo *)(tlvValue);
  offset += sizeof(struct ModelGlobalDataInfo);
  geFifoInfo->totalSize = tlvFifoInfo->total_size;
  geFifoInfo->fifoNum = tlvFifoInfo->num;
  uint64_t memTlvSize = sizeof(uint64_t);
  GELOGI("fifoNum[%u] totalSize[%lu] offset[%lu].", tlvFifoInfo->num,
          tlvFifoInfo->total_size, offset);
  if ((geFifoInfo->fifoNum > 0) && (CheckLenValid(len, offset, geFifoInfo->fifoNum * memTlvSize))) {
    geFifoInfo->fifoAllAddr = (uint64_t *)mmMalloc(geFifoInfo->fifoNum * sizeof(uint64_t));
    if (geFifoInfo->fifoAllAddr == NULL) {
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "mmMalloc failed.");
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }
    geFifoInfo->fifoBaseAddr = modelData->part.fifoPtr;
    if ((modelData->part.fifoPtr == NULL) || (modelData->part.fifoSize < geFifoInfo->totalSize)) {
      rtError_t rtRet = rtMalloc(&geFifoInfo->fifoBaseAddr, geFifoInfo->totalSize, mdlDesc->memType, 0);
      if (rtRet != RT_ERROR_NONE) {
        GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "rtMalloc failed.");
        return ACL_ERROR_GE_LOAD_MODEL;
      }
      mdlDesc->innerPtrState = mdlDesc->innerPtrState | INNER_FIFO_PTR;
      GELOGD("use innerFifoPtr");
    }
    mdlDesc->part.fifoPtr = geFifoInfo->fifoBaseAddr;
    geFifoInfo->fifoAllAddr[0] = (uintptr_t)geFifoInfo->fifoBaseAddr;
    for (uint8_t i = 1; i < geFifoInfo->fifoNum; i++) {
      GELOGI("index[%u], mem_size[%lu].", i, tlvFifoInfo->mem_size[i - 1]);
      geFifoInfo->fifoAllAddr[i] = geFifoInfo->fifoAllAddr[i - 1] + tlvFifoInfo->mem_size[i - 1];
      offset += memTlvSize;
    }
    if ((geFifoInfo->fifoAllAddr[geFifoInfo->fifoNum - 1] - geFifoInfo->fifoAllAddr[0] +
         tlvFifoInfo->mem_size[geFifoInfo->fifoNum - 1]) != tlvFifoInfo->total_size) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "fifo total_size invalid.");
      return ACL_ERROR_GE_LOAD_MODEL;
    }
    return SUCCESS;
  }
  GELOGE(ACL_ERROR_GE_PARAM_INVALID, "proc fifo failed.");
  return ACL_ERROR_GE_LOAD_MODEL;
}

static Status ParseTlvList(const ModelData *modelData, uint8_t *tlvList, uint64_t tlvListLen, GeModelDesc *mdlDesc) {
  uint64_t offset = 0UL;
  while (CheckLenValid(tlvListLen, offset, sizeof(struct TlvHead))) {
    struct TlvHead *tlv = (struct TlvHead *)(tlvList + offset);
    offset += sizeof(struct TlvHead);
    GELOGI("total len[%lu] tlv len[%u] type[%u] offset[%lu].", tlvListLen,
           tlv->len, tlv->type, offset);
    if (!CheckLenValid(tlvListLen, offset, tlv->len)) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "tlv len invalid.");
      return ACL_ERROR_GE_LOAD_MODEL;
    }
    if (ProcFifoInfo(modelData, tlv->data, tlv->len, mdlDesc) != SUCCESS) {
      return ACL_ERROR_GE_LOAD_MODEL;
    }
    offset += tlv->len;
  }
  if (offset != tlvListLen) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "ParseTlvList invalid.");
    return ACL_ERROR_GE_LOAD_MODEL;
  }
  return SUCCESS;
}

Status ParseModelDescExtend(const ModelData *modelData, uint8_t *data, size_t size, GeModelDesc *mdlDesc) {
  if (!CheckLenValid(size, 0, sizeof(struct ModelExtendHead))) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "part size[%zu] < sizeof(ModelExtendHead)[%zu].", size,
           sizeof(struct ModelExtendHead));
    return ACL_ERROR_GE_LOAD_MODEL;
  }
  struct ModelExtendHead *extendDesc = (struct ModelExtendHead *)data;
  if (extendDesc->magic != MODEL_EXTEND_DESC_MAGIC_NUM) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "extend part magic invalid.");
    return ACL_ERROR_GE_LOAD_MODEL;
  }
  return ParseTlvList(modelData, data + sizeof(struct ModelExtendHead), extendDesc->len, mdlDesc);
}

static Status ParseModelDescExtendInfo(const ModelData *modelData, size_t offset,
                                       uint8_t *data, size_t size, GeModelDesc *mdlDesc) {
  if (modelData->flag == NEED_READ_FROM_FD) {
    return ParserPartionFromFd(modelData, offset, size, (void *)mdlDesc,
                               (ParseFromFd)ParseModelDescExtend);
  }
  return ParseModelDescExtend(modelData, data, size, mdlDesc);
}

Status MdlPartitionParse(const ModelData *modelData, GeModelDesc *mdlDesc) {
  uint32_t partitionNum = 0U;
  Status ret = ParsePartitionPreProcess(modelData, &partitionNum);
  if (ret != SUCCESS) {
    return ret;
  }

  size_t offset = sizeof(ModelFileHeader) + sizeof(ModelPartitionTable);
  size_t memLen = sizeof(ModelPartitionMemInfo);
  size_t baseOffset = offset + memLen * partitionNum;
  GELOGD("model partition num is %u baseOffset is %zu.", partitionNum, baseOffset);

  uint8_t *baseAddr = (uint8_t *)modelData->modelData;
  uint8_t *partBaseAddr = baseAddr + baseOffset;
  uint64_t validLen = modelData->modelLen - baseOffset;
  for (size_t i = 0UL; i < partitionNum; ++i) {
    ModelPartitionMemInfo *partitionInfo = (ModelPartitionMemInfo *)(baseAddr + offset);
    size_t size = partitionInfo->mem_size;
    uint8_t *data = partBaseAddr + partitionInfo->mem_offset;
    if ((partitionInfo->mem_offset > validLen) || (size == 0) ||
        (size > validLen - partitionInfo->mem_offset)) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "partition type %d, mem_size is %zu offset is %zu,"
             " model_len is %zu, invalid",
          partitionInfo->type, size, (size_t)partitionInfo->mem_offset,
          (size_t)modelData->modelLen);
      return ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED;
    }
    GELOGD("partition type %d, mem_size is %zu mem_offset is %zu",
           partitionInfo->type, size, (size_t)partitionInfo->mem_offset);
    size_t dataOffset = baseOffset + partitionInfo->mem_offset;
    switch (partitionInfo->type) {
      case PRE_MODEL_DESC: {
        ret = ParseModelDesc(modelData, dataOffset, data, size, mdlDesc);
        break;
      }
      case PRE_MODEL_DESC_EXTEND: {
        ret = ParseModelDescExtendInfo(modelData, dataOffset, data, size, mdlDesc);
        break;
      }
      case WEIGHTS_DATA: {
        ret = ParseWeightData(modelData, dataOffset, data, size, mdlDesc);
        break;
      }
      case TBE_KERNELS: {
        ret = ParseTbeKernels(modelData, dataOffset, data, size, mdlDesc);
        break;
      }
      case STATIC_TASK_DESC: {
        ret = ParseStaticTaskDesc(modelData, dataOffset, data, size, mdlDesc);
        break;
      }
      case DYNAMIC_TASK_DESC: {
        ret = ParseDynamicTaskDesc(modelData, dataOffset, data, size, mdlDesc);
        break;
      }
      case TASK_PARAM: {
        ret = ParseTaskParam(modelData, dataOffset, data, size, mdlDesc);
        break;
      }
      case MODEL_INOUT_INFO: {
        ret = ParseModelIoDesc(modelData, dataOffset, data, size, mdlDesc);
        break;
      }
      default:
        break;
    }
    if (ret != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "parse model type[%d] failed.", partitionInfo->type);
      return ret;
    }
    offset += memLen;
  }
  return SUCCESS;
}