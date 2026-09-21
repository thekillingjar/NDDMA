/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/limits.h>
#include "model_loader.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "framework/executor_c/ge_executor.h"
Status LoadOfflineModelFromFile(const char *modelPath, ModelData *modelData) {
  char trustPath[PATH_MAX] = {};
  if (mmRealPath(modelPath, trustPath, MMPA_MAX_PATH) != EN_OK) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "realpath is invalid");
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }
  if (mmAccess(trustPath, R_OK) != 0) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "can not access file '%s'.", modelPath);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }
  int fd = open(trustPath, O_RDONLY);
  if (fd < 0) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[open][file] failed, file %s", modelPath);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  size_t len = (size_t)lseek(fd, 0, SEEK_END);
  (void)lseek(fd, 0, SEEK_SET);

  if (sizeof(ModelFileHeader) >= len) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "file head size[%zu] larger than file len[%zu]",
           sizeof(ModelFileHeader), len);
    close(fd);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  char *data = (char *)(mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0));
  if (data == MAP_FAILED) {
    close(fd);
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "load from file failed, len %zu, file %s", len, modelPath);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  close(fd);
  modelData->modelData = data;
  modelData->modelLen = len;
  modelData->flag = NO_NEED_READ_FROM_FD;
  modelData->fd = NULL;
  return SUCCESS;
}

void FreeModelData(ModelData *data) {
  if (data->modelData != NULL) {
    munmap(data->modelData, data->modelLen);
    data->modelData = NULL;
  }
}