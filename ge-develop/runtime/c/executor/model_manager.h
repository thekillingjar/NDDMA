/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_EXECUTOR_C_MODEL_MANAGER_H_
#define GE_EXECUTOR_C_MODEL_MANAGER_H_
#include "framework/executor_c/ge_executor_types.h"
#include "framework/executor_c/types.h"
#include "ref_obj.h"
#ifdef __cplusplus
extern "C" {
#endif
#define INVALID_PHY_MODLE_ID 0xFFFFFFFF

typedef struct {
  uint32_t fifoNum;
  void *fifoBaseAddr;
  uint64_t *fifoAllAddr;
  uint64_t totalSize;
} ModelFifoInfo;

typedef struct {
  uint32_t modelId;
  uint32_t phyModelId;
  uint64_t innerPtrState;
  Partition part;
  ModelInOutInfo ioInfo;
  uint64_t stepId;
  void *modelDbgHandle;
  ModelFifoInfo fifoInfo;
  size_t memType;
} GeModelDesc;

typedef struct {
  RefObj refObj;
  GeModelDesc modelDesc;
  void *appInfo;
} ModelDescRefObj;

uint32_t GenerateModelId(void);
void InitGeModelDescManager(void);
void DeInitGeModelDescManager(void);
Status AddModelDescRef(ModelDescRefObj *mdlDescRef);
void DestroyModelDesc(GeModelDesc *modelDesc);
GeModelDesc *GetModelDescRef(uint32_t modelId);
Status DelModelDescRef(uint32_t modelId);
void ReleaseModelDescRef(GeModelDesc *modelDesc);
void* FuncNull(RefObj *refObj);
#ifdef __cplusplus
}
#endif
#endif  // GE_EXECUTOR_C_MODEL_MANAGER_H_
