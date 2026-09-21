/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_MODEL_DESC_INTERNAL_H
#define ACL_MODEL_DESC_INTERNAL_H

#include "acl/acl_base.h"
#include "acl/acl_mdl.h"
#include "vector.h"

#define ACL_MAX_TENSOR_NAME_LEN 128
#define ACL_MAX_DIM_CNT 128

typedef struct {
  void *weightPtr;
  void *modelDescPtr;
  void *kernelPtr;
  void *kernelArgsPtr;
  void *staticTaskPtr;
  void *dynamicTaskPtr;
  void *fifoTaskPtr;
} aclmdlExeOMInfo;

struct aclmdlConfigHandle {
  int32_t priority;
  size_t mdlLoadType;
  char *loadPath;
  void *mdlAddr;
  size_t mdlSize;
  uint64_t attrState;
  size_t memType;
  aclmdlExeOMInfo exeOMInfo;
  aclmdlExeOMDesc exeOMDesc;
};

struct aclmdlExecConfigHandle {
  void *workPtr;
  size_t workSize;
  size_t mpamId;
  size_t aicQos;
  size_t aicOst;
  size_t mecTimeThreshHold;
};

typedef struct {
  char *name;
  size_t size;
  aclFormat format;
  aclDataType dataType;
  Vector dims;
} aclmdlTensorDesc;

struct aclmdlDesc {
  uint32_t modelId;
  Vector inputDesc;
  Vector outputDesc;
};

#endif  // ACL_MODEL_DESC_INTERNAL_H
