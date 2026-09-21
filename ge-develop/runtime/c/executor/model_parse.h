/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_EXECUTOR_C_MODEL_PARSE_H_
#define GE_EXECUTOR_C_MODEL_PARSE_H_
#include "framework/executor_c/ge_executor_types.h"
#include "framework/executor_c/types.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "framework/common/tlv/nano_common_desc.h"
#include "framework/common/tlv/tlv.h"
#include "model_manager.h"
#ifdef __cplusplus
extern "C" {
#endif

#define MAX_PARTITION_NUM 16
#define INNER_PRE_MODEL_DESC_PTR      (0x1 << PRE_MODEL_DESC)
#define INNER_WEIGHTS_DATA_PTR        (0x1 << WEIGHTS_DATA)
#define INNER_TBE_KERNELS_PTR         (0x1 << TBE_KERNELS)
#define INNER_STATIC_TASK_DESC_PTR    (0x1 << STATIC_TASK_DESC)
#define INNER_DYNAMIC_TASK_DESC_PTR   (0x1 << DYNAMIC_TASK_DESC)
#define INNER_TASK_PARAM_PTR          (0x1 << TASK_PARAM)
#define INNER_FIFO_PTR                (0x1 << PRE_MODEL_DESC_EXTEND)

Status MdlPartitionParse(const ModelData *modelData, GeModelDesc *mdlDesc);
Status ParseModelIoDescInfo(const ModelData *modelData, uint8_t *data, size_t size, ModelInOutInfo *info);
Status ParseModelDescExtend(const ModelData *modelData, uint8_t *data, size_t size, GeModelDesc *mdlDesc);
void ModelInfoDestroy(void *base);
Status ParsePartitionPreProcess(const ModelData *modelData, uint32_t *partitionNum);
void InitModelInOutInfo(ModelInOutInfo *info);
void DeInitModelInOutInfo(ModelInOutInfo *info);
void ResizeModelInOutTensorDesc(size_t size, Vector *descVector);
Status ReadDataFromFd(mmFileHandle *fd, size_t offset, size_t len, void *dst);
void InitModelFifoInfo(ModelFifoInfo *fifoInfo);
void DeInitModelFifoInfo(ModelFifoInfo *fifoInfo);
#ifdef __cplusplus
}
#endif
#endif  // GE_EXECUTOR_C_MODEL_PARSE_H_
