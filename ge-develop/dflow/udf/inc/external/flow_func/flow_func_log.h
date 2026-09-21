/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_LOG_H
#define FLOW_FUNC_LOG_H

#include "flow_func_defines.h"

namespace FlowFunc {
enum class FLOW_FUNC_VISIBILITY FlowFuncLogType {
    DEBUG_LOG = 0,
    RUN_LOG = 1
};
enum class FLOW_FUNC_VISIBILITY FlowFuncLogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

class FLOW_FUNC_VISIBILITY FlowFuncLogger {
public:
    static FlowFuncLogger &GetLogger(FlowFuncLogType type);

    static const char *GetLogExtHeader();

    FlowFuncLogger() = default;

    virtual ~FlowFuncLogger() = default;

    virtual bool IsLogEnable(FlowFuncLogLevel level) = 0;

    virtual void Error(const char *fmt, ...) __attribute__((format(printf, 2, 3))) = 0;

    virtual void Warn(const char *fmt, ...) __attribute__((format(printf, 2, 3))) = 0;

    virtual void Info(const char *fmt, ...) __attribute__((format(printf, 2, 3))) = 0;

    virtual void Debug(const char *fmt, ...) __attribute__((format(printf, 2, 3))) = 0;
};
}

#define FLOW_FUNC_LOG_DEBUG(fmt, ...)                                               \
do {                                                                                \
    FlowFunc::FlowFuncLogger &debugLogger = FlowFunc::FlowFuncLogger::GetLogger(    \
        FlowFunc::FlowFuncLogType::DEBUG_LOG);                                      \
    if (debugLogger.IsLogEnable(FlowFunc::FlowFuncLogLevel::DEBUG)) {               \
        debugLogger.Debug("[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,  \
            FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);            \
    }                                                                               \
} while (0)

#define FLOW_FUNC_LOG_INFO(fmt, ...)                                                \
do {                                                                                \
    FlowFunc::FlowFuncLogger &debugLogger = FlowFunc::FlowFuncLogger::GetLogger(    \
        FlowFunc::FlowFuncLogType::DEBUG_LOG);                                      \
    if (debugLogger.IsLogEnable(FlowFunc::FlowFuncLogLevel::INFO)) {                \
        debugLogger.Info("[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,   \
            FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);            \
    }                                                                               \
} while (0)

#define FLOW_FUNC_LOG_WARN(fmt, ...)                                                \
do {                                                                                \
    FlowFunc::FlowFuncLogger &debugLogger = FlowFunc::FlowFuncLogger::GetLogger(    \
        FlowFunc::FlowFuncLogType::DEBUG_LOG);                                      \
    if (debugLogger.IsLogEnable(FlowFunc::FlowFuncLogLevel::WARN)) {                \
        debugLogger.Warn("[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,   \
            FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);            \
    }                                                                               \
} while (0)

#define FLOW_FUNC_LOG_ERROR(fmt, ...)                                                   \
do {                                                                                    \
    FlowFunc::FlowFuncLogger::GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG).Error(    \
        "[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,                        \
        FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);                    \
} while (0)

#define FLOW_FUNC_RUN_LOG_INFO(fmt, ...)                                                \
do {                                                                                    \
    FlowFunc::FlowFuncLogger &runLogger = FlowFunc::FlowFuncLogger::GetLogger(          \
        FlowFunc::FlowFuncLogType::RUN_LOG);                                            \
    if (runLogger.IsLogEnable(FlowFunc::FlowFuncLogLevel::INFO)) {                      \
        runLogger.Info("[%s:%d][%s]%s[RUN]: " fmt, __FILE__, __LINE__, __FUNCTION__,    \
            FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);                \
    }                                                                                   \
} while (0)

#define FLOW_FUNC_RUN_LOG_ERROR(fmt, ...)                                           \
do {                                                                                \
    FlowFunc::FlowFuncLogger::GetLogger(FlowFunc::FlowFuncLogType::RUN_LOG).Error(  \
        "[%s:%d][%s]%s[RUN]: " fmt, __FILE__, __LINE__, __FUNCTION__,               \
        FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);                \
} while (0)
#endif // FLOW_FUNC_LOG_H
