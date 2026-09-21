/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_EXECUTOR_C_DBG_PROFILING_H_
#define GE_EXECUTOR_C_DBG_PROFILING_H_
#include "framework/executor_c/ge_executor.h"
#include "ge/ge_error_codes.h"
#ifdef __cplusplus
extern "C" {
#endif
void SetMsprofEnable(bool flag);
int32_t MsprofCtrlHandleFunc(uint32_t dataType, void *data, uint32_t dataLen);
GE_FUNC_VISIBILITY  bool DbgGetprofEnable(void);
GE_FUNC_VISIBILITY Status DbgProfInit(const char *cfg);
GE_FUNC_VISIBILITY Status DbgProfReportDataProcess(uint32_t modelId, char *om);
GE_FUNC_VISIBILITY Status DbgProfDeInit(void);
GE_FUNC_VISIBILITY Status DbgNotifySetDevice(uint32_t chipId, uint32_t deviceId);
#ifdef __cplusplus
}
#endif
#endif  // GE_EXECUTOR_C_DBG_PROFILING_H_
