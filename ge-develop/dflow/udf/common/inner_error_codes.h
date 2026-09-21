/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_COMMON_INNER_ERROR_CODES_H
#define UDF_COMMON_INNER_ERROR_CODES_H
#include <cstdint>

namespace FlowFunc {
constexpr int32_t FLOW_FUNC_ERR_INIT_AGAIN = 0x0090;    // need init again, use for flow func init
constexpr int32_t FLOW_FUNC_ERR_PROC_PENDING = 0x0091;  // proc pending, need other event wakeup
constexpr int32_t FLOW_FUNC_ERR_QUEUE_FULL = 0x0092;  // queue full, some times not error
constexpr int32_t FLOW_FUNC_ERR_QUEUE_EMPTY = 0x0093;  // queue empty, some times not error
constexpr int32_t FLOW_FUNC_SUSPEND_FAILED = 0x0094;  // suspend udf executor failed
constexpr int32_t FLOW_FUNC_RECOVER_FAILED = 0x0095;  // clear and recover udf executor failed
constexpr int32_t FLOW_FUNC_DISCARD_INPUT_ERROR = 0x0096; // discard all input failed in suspend status
constexpr int32_t FLOW_FUNC_PROCESSOR_PARAM_ERROR = 0x0097; // evnet msg from processor is invalid

constexpr int32_t HICAID_SUCCESS = 0x0000; // Success code
constexpr int32_t HICAID_FAILED = 0x0001; // Failed code
constexpr int32_t HICAID_ERR_PARAM_INVALID = 0x0002; // param invalid

// 0x01xx used for queue error
constexpr int32_t HICAID_ERR_QUEUE_FAILED = 0x0101; // queue op failed
constexpr int32_t HICAID_ERR_QUEUE_EMPTY = 0x0102; // queue empty
constexpr int32_t HICAID_ERR_QUEUE_FULL = 0x0103; // queue full
}

#endif  // UDF_COMMON_INNER_ERROR_CODES_H
