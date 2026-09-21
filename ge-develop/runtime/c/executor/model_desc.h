/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_EXECUTOR_C_MODEL_DESC_H_
#define GE_EXECUTOR_C_MODEL_DESC_H_
#include "framework/executor_c/ge_executor_types.h"
#include "framework/executor_c/types.h"
#ifdef __cplusplus
extern "C" {
#endif
static inline bool CheckLenValid(const size_t total, const size_t offset, const size_t nextLen) {
  return !(nextLen > total - offset);
}
Status GetPartInfoFromModel(const ModelData *modelData, ModelPartition *partition);
Status GetModelMemAndWeightSize(const ModelData *modelData, size_t *workSize, size_t *weightSize);
Status GetModelPartitionSize(const ModelData *modelData, GePartitionSize *mdlPartitionSize);
Status CheckOmHeadWithMem(const ModelData *model_data);
#ifdef __cplusplus
}
#endif
#endif  // GE_EXECUTOR_C_MODEL_DESC_H_
