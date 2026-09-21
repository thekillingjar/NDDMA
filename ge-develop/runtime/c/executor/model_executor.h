/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_EXECUTOR_C_MODEL_EXECUTOR_H_
#define GE_EXECUTOR_C_MODEL_EXECUTOR_H_
#include "framework/executor_c/ge_executor_types.h"
#include "framework/executor_c/types.h"
#ifdef __cplusplus
extern "C" {
#endif
Status ModelExecuteInner(uint32_t modelId, ExecHandleDesc *execDesc, bool sync, const InputData *input_data,
                         OutputData *output_data);
#ifdef __cplusplus
}
#endif
#endif  // GE_EXECUTOR_C_MODEL_EXECUTOR_H_
