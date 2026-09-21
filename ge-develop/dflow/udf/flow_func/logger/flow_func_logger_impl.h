/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_LOGGER_FLOW_FUNC_LOGGER_IMPL_H
#define FLOW_FUNC_LOGGER_FLOW_FUNC_LOGGER_IMPL_H

#include <cstdint>
#include <atomic>
#include <string>
#include "dlog_pub.h"
#include "flow_func/flow_func_log.h"
#include "flow_func_logger_flow_control.h"

namespace FlowFunc {

class FlowFuncLoggerImpl : public FlowFuncLogger {
public:
    explicit FlowFuncLoggerImpl(FlowFuncLogType type) : type_(type), flow_control_(type) {}

    ~FlowFuncLoggerImpl() override = default;

    bool IsLogEnable(FlowFuncLogLevel level) override;

    void Error(const char *fmt, ...) override;

    void Warn(const char *fmt, ...) override;

    void Info(const char *fmt, ...) override;

    void Debug(const char *fmt, ...) override;

    std::string StatisticInfo() const;

    void RefreshLimitNum();

private:

    constexpr int32_t GetLogModuleId() const {
        if (type_ == FlowFuncLogType::RUN_LOG) {
            return RUN_LOG_MASK | static_cast<int32_t>(APP);
        } else {
            return DEBUG_LOG_MASK | static_cast<int32_t>(APP);
        }
    }

    constexpr int32_t ConvertLogLevel(FlowFuncLogLevel level) const {
        switch (level) {
            case FlowFuncLogLevel::DEBUG:
                return DLOG_DEBUG;
            case FlowFuncLogLevel::INFO:
                return DLOG_INFO;
            case FlowFuncLogLevel::WARN:
                return DLOG_WARN;
            case FlowFuncLogLevel::ERROR:
                return DLOG_ERROR;
            default:
                return DLOG_NULL;
        }
    }

private:
    FlowFuncLogType type_;
    FlowFuncLoggerFlowControl flow_control_;
    std::atomic<uint64_t> debug_log_num_{0};
    std::atomic<uint64_t> info_log_num_{0};
    std::atomic<uint64_t> warn_log_num_{0};
    std::atomic<uint64_t> error_log_num_{0};

    std::atomic<uint64_t> flow_control_debug_num_{0};
    std::atomic<uint64_t> flow_control_info_log_num_{0};
    std::atomic<uint64_t> flow_control_warn_log_num_{0};
    std::atomic<uint64_t> flow_control_error_log_num_{0};
};
}

#endif // FLOW_FUNC_LOGGER_FLOW_FUNC_LOGGER_IMPL_H
