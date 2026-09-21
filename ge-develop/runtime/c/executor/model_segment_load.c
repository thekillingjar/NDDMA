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
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "framework/executor_c/ge_executor.h"

#define MAX_PATH_LEN_OF_LITEOS 128

Status LoadOfflineModelFromFile(const char *modelPath, ModelData *modelData) {
  if (strlen(modelPath) > MAX_PATH_LEN_OF_LITEOS) {
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  mmFileHandle *fd = mmOpenFile(modelPath, FILE_READ_BIN);
  if (fd == NULL) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[open][file] failed, file %s", modelPath);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }
  (void)mmSeekFile(fd, 0, MM_SEEK_FILE_END);
  int32_t len = mmTellFile(fd);
  if (len == -1) {
    mmCloseFile(fd);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  size_t headTableLen = sizeof(ModelFileHeader) + sizeof(ModelPartitionTable);
  if (headTableLen >= (size_t)len) {
    mmCloseFile(fd);
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "the size[%zu] of file head and table is larger than file len %zu",
           headTableLen, (size_t)len);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  ModelPartitionTable table;
  Status ret = ReadDataFromFd(fd, sizeof(ModelFileHeader), sizeof(ModelPartitionTable), (void *)(&table));
  if (ret != SUCCESS) {
    mmCloseFile(fd);
    return ret;
  }

  size_t ctrlLen = headTableLen + table.num * sizeof(ModelPartitionMemInfo);
  if ((ctrlLen >= (size_t)len) || (table.num > MAX_PARTITION_NUM)) {
    mmCloseFile(fd);
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "the size[%zu] of file ctrl info is incorrect[%zu]", ctrlLen, (size_t)len);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  char *data = (char *)mmMalloc(ctrlLen * sizeof(uint8_t));
  if (data == NULL) {
    mmCloseFile(fd);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  ret = ReadDataFromFd(fd, 0, ctrlLen, (void *)(data));
  if (ret != SUCCESS) {
    mmCloseFile(fd);
    mmFree(data);
    return ret;
  }

  modelData->modelData = data;
  modelData->modelLen = len;
  modelData->flag = NEED_READ_FROM_FD;
  modelData->fd = fd;
  return SUCCESS;
}

void FreeModelData(ModelData *data) {
  if (data->modelData != NULL) {
    mmFree(data->modelData);
    data->modelData = NULL;
    data->modelLen = 0;
  }
  if (data->fd != NULL) {
    mmCloseFile(data->fd);
    data->fd = NULL;
  }
}