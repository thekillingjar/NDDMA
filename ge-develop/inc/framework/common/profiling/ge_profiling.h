/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_GE_PROFILING_H_
#define INC_FRAMEWORK_COMMON_GE_PROFILING_H_

#include "ge/ge_api_error_codes.h"
#include "runtime/base.h"

GE_FUNC_VISIBILITY ge::Status ProfGetDeviceFormGraphId(const uint32_t graph_id, uint32_t &device_id);

#endif  // INC_FRAMEWORK_COMMON_GE_PROFILING_H_
