/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_C_TYPES_H_
#define INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_C_TYPES_H_
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "mmpa_api.h"
#include "vector.h"
#if defined(__cplusplus)
extern "C" {
#endif
typedef uint32_t Status;
#define SUCCESS  0

typedef struct {
  void *weightPtr;
  size_t weightSize;
  void *modelDescPtr;
  size_t modelDescSize;
  void *kernelPtr;
  size_t kernelSize;
  void *paramPtr;
  size_t paramSize;
  void *taskPtr;
  size_t taskSize;
  void *dynTaskPtr;
  size_t dynTaskSize;
  void *fifoPtr;
  size_t fifoSize;
} Partition;

typedef enum {
  NO_NEED_READ_FROM_FD,
  NEED_READ_FROM_FD
} ReadFileFlag;

typedef struct {
  void *modelData;
  uint64_t modelLen;
  int32_t priority;
  mmFileHandle *fd;
  ReadFileFlag flag;
  Partition part;
  size_t memType;
  int32_t resv[3];
} ModelData;

typedef struct {
  void *data;
  uint64_t length;
} DataBuffer;

typedef struct {
  DataBuffer *dataBuffer;
} DataBlob;

typedef struct {
  Vector blobs; // type : DataBlob
  uint64_t* io_addr;
  uint64_t* io_addr_host;
  uint32_t ioa_size;
} DataSet;

typedef DataSet InputData;
typedef DataSet OutputData;

typedef struct {
  void *stream;
  void *workPtr;
  size_t workSize;
  size_t mpamId;
  size_t aicQos;
  size_t aicOst;
  size_t mecTimeThreshHold;
} ExecHandleDesc;

typedef struct {
  size_t workSize;
  size_t weightSize;
  size_t modelDescSize;
  size_t kernelSize;
  size_t kernelArgsSize;
  size_t staticTaskSize;
  size_t dynamicTaskSize;
  size_t fifoSize;
} GePartitionSize;

typedef struct {
  uint32_t tlvType;
  uint32_t (*pfnTlvProc)(uint32_t len, uint8_t *tlvValue, void *appInfo);
} TlvProcPair;

#if defined(__cplusplus)
}
#endif

#endif  // INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_C_TYPES_H_
