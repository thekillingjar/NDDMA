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
#include "acl/acl_base.h"
#include "framework/executor_c/ge_executor_types.h"
#include "mmpa_api.h"

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size) {
  DataBuffer *dataBuffer = (DataBuffer *)mmMalloc(sizeof(DataBuffer));
  if (dataBuffer == NULL) {
    return NULL;
  }
  dataBuffer->data = data;
  dataBuffer->length = size;
  return (aclDataBuffer *)dataBuffer;
}

aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) {
  if (dataBuffer == NULL) {
    return ACL_ERROR_INVALID_PARAM;
  }
  mmFree((void *)dataBuffer);
  return ACL_SUCCESS;
}

void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer) {
  if (dataBuffer == NULL) {
    return NULL;
  }
  return ((const DataBuffer *)dataBuffer)->data;
}

size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer) {
  if (dataBuffer == NULL) {
    return 0UL;
  }
  return (size_t)((const DataBuffer *)dataBuffer)->length;
}