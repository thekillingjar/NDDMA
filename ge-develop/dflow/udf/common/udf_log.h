/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UDF_LOG_H
#define COMMON_UDF_LOG_H

#include <cstdint>
#include <unistd.h>
#include <sys/syscall.h>
#include "dlog_pub.h"

namespace FlowFunc {
inline int64_t GetTid() {
    thread_local static const int64_t tid = static_cast<int64_t>(syscall(__NR_gettid));
    return tid;
}

constexpr int32_t kModuleIdUdf = static_cast<int32_t>(UDF);
constexpr int32_t kModuleIdUdfRun = RUN_LOG_MASK | static_cast<int32_t>(UDF);

#define UDF_LOG_DEBUG(fmt, ...)                                            \
    dlog_debug(FlowFunc::kModuleIdUdf, "[%s][tid:%ld]: " fmt, __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
#define UDF_LOG_INFO(fmt, ...)                                            \
    dlog_info(FlowFunc::kModuleIdUdf, "[%s][tid:%ld]: " fmt, __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
#define UDF_LOG_WARN(fmt, ...)                                            \
    dlog_warn(FlowFunc::kModuleIdUdf, "[%s][tid:%ld]: " fmt, __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
#define UDF_LOG_ERROR(fmt, ...)                                            \
    dlog_error(FlowFunc::kModuleIdUdf, "[%s][tid:%ld]: " fmt, __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)

#define UDF_RUN_LOG_ERROR(fmt, ...)                                        \
    dlog_error(FlowFunc::kModuleIdUdfRun, "[%s][tid:%ld][RUN]: " fmt, __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
#define UDF_RUN_LOG_WARN(fmt, ...)                                        \
    dlog_warn(FlowFunc::kModuleIdUdfRun, "[%s][tid:%ld][RUN]: " fmt, __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
#define UDF_RUN_LOG_INFO(fmt, ...)                                         \
    dlog_info(FlowFunc::kModuleIdUdfRun, "[%s][tid:%ld][RUN]: " fmt, __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)

#define HICAID_LOG_DEBUG UDF_LOG_DEBUG
#define HICAID_LOG_INFO UDF_LOG_INFO
#define HICAID_LOG_WARN UDF_LOG_WARN
#define HICAID_LOG_ERROR UDF_LOG_ERROR
#define HICAID_RUN_LOG_INFO UDF_RUN_LOG_INFO
}

#endif // COMMON_UDF_LOG_H
