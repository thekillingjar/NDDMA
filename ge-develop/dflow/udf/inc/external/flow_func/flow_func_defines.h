/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_DEFINES_H
#define FLOW_FUNC_DEFINES_H
#include <cstdint>

#define FLOW_FUNC_VISIBILITY __attribute__((visibility("default")))

namespace FlowFunc {
constexpr int32_t FLOW_FUNC_SUCCESS = 0; // Success code

constexpr int32_t FLOW_FUNC_ERR_PARAM_INVALID = 164000; // param invalid
constexpr int32_t FLOW_FUNC_ERR_ATTR_NOT_EXITS = 164001; // attr is not exits
constexpr int32_t FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH = 164002; // attr type is mismatch

// error code start from 3 means system is abnormal, but still available.
constexpr int32_t FLOW_FUNC_ERR_DATA_ALIGN_FAILED = 364000; // data align failed

constexpr int32_t FLOW_FUNC_FAILED = 564000; // Failed code, udf internal error
constexpr int32_t FLOW_FUNC_ERR_TIME_OUT_ERROR = 564001;
constexpr int32_t FLOW_FUNC_ERR_NOT_SUPPORT = 564002; // not support
constexpr int32_t FLOW_FUNC_STATUS_REDEPLOYING = 564003; // proc redeploying
constexpr int32_t FLOW_FUNC_STATUS_EXIT = 564004; // proc exiting 

constexpr int32_t FLOW_FUNC_ERR_DRV_ERROR = 564100; // drv common error
constexpr int32_t FLOW_FUNC_ERR_MEM_BUF_ERROR = 564101; // mem buf error
constexpr int32_t FLOW_FUNC_ERR_QUEUE_ERROR = 564102; // queue error
constexpr int32_t FLOW_FUNC_ERR_EVENT_ERROR = 564103; // event error

// user define error can start from 9900000
constexpr int32_t FLOW_FUNC_ERR_USER_DEFINE_START = 9900000; // user define error code start
constexpr int32_t FLOW_FUNC_ERR_USER_DEFINE_END = 9999999; // user define error code end
}

#endif // FLOW_FUNC_DEFINES_H
