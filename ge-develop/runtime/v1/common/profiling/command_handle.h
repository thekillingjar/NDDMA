/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_PROFILING_COMMAND_HANDLE_H_
#define GE_COMMON_PROFILING_COMMAND_HANDLE_H_

#include "runtime/base.h"

namespace ge {
// RTS Callback(runtime/base.h): typedef rtError_t (*rtProfCtrlHandle)(uint32_t type, void *data, uint32_t len)
rtError_t ProfCtrlHandle(const uint32_t ctrl_type, void *const ctrl_data, const uint32_t data_len);
}
#endif  // GE_COMMON_PROFILING_COMMAND_HANDLE_H_
