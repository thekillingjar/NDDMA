/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_logger_impl.h"
#include <sstream>
#include "common/udf_log.h"
#include "flow_func_logger_manager.h"

namespace FlowFunc {
namespace {
std::string GenLogExtHeader() {
    int64_t tid = GetTid();
    return "[tid:" + std::to_string(tid) + "]";
}
}

FlowFuncLogger &FlowFuncLogger::GetLogger(FlowFuncLogType type) {
    return FlowFuncLoggerManager::Instance().GetLogger(type);
}

const char *FlowFuncLogger::GetLogExtHeader() {
    thread_local static std::string log_ext_header = GenLogExtHeader();
    return log_ext_header.c_str();
}

bool FlowFuncLoggerImpl::IsLogEnable(FlowFuncLogLevel level) {
    int32_t log_module_id = GetLogModuleId();
    int32_t log_level = ConvertLogLevel(level);
    return CheckLogLevel(log_module_id, log_level) == 1;
}

void FlowFuncLoggerImpl::Error(const char *fmt, ...) {
    int32_t log_module_id = GetLogModuleId();
    int32_t log_level = ConvertLogLevel(FlowFuncLogLevel::ERROR);
    if (CheckLogLevel(log_module_id, log_level) != 1) {
        return;
    }

    if (!flow_control_.TryAcquire()) {
        ++flow_control_error_log_num_;
        return;
    }

    va_list arg_list;
    va_start(arg_list, fmt);
    DlogVaList(log_module_id, log_level, fmt, arg_list);
    va_end(arg_list);
    ++error_log_num_;
}

void FlowFuncLoggerImpl::Warn(const char *fmt, ...) {
    int32_t log_module_id = GetLogModuleId();
    int32_t log_level = ConvertLogLevel(FlowFuncLogLevel::WARN);
    if (CheckLogLevel(log_module_id, log_level) != 1) {
        return;
    }

    if (!flow_control_.TryAcquire()) {
        ++flow_control_warn_log_num_;
        return;
    }

    va_list arg_list;
    va_start(arg_list, fmt);
    DlogVaList(log_module_id, log_level, fmt, arg_list);
    va_end(arg_list);
    ++warn_log_num_;
}

void FlowFuncLoggerImpl::Info(const char *fmt, ...) {
    int32_t log_module_id = GetLogModuleId();
    int32_t log_level = ConvertLogLevel(FlowFuncLogLevel::INFO);
    if (CheckLogLevel(log_module_id, log_level) != 1) {
        return;
    }
    if (!flow_control_.TryAcquire()) {
        ++flow_control_info_log_num_;
        return;
    }

    va_list arg_list;
    va_start(arg_list, fmt);
    DlogVaList(log_module_id, log_level, fmt, arg_list);
    va_end(arg_list);
    ++info_log_num_;
}

void FlowFuncLoggerImpl::Debug(const char *fmt, ...) {
    int32_t log_module_id = GetLogModuleId();
    int32_t log_level = ConvertLogLevel(FlowFuncLogLevel::DEBUG);
    if (CheckLogLevel(log_module_id, log_level) != 1) {
        return;
    }
    if (!flow_control_.TryAcquire()) {
        ++flow_control_debug_num_;
        return;
    }
    va_list arg_list;
    va_start(arg_list, fmt);
    DlogVaList(log_module_id, log_level, fmt, arg_list);
    va_end(arg_list);
    ++debug_log_num_;
}

std::string FlowFuncLoggerImpl::StatisticInfo() const {
    std::ostringstream str;
    str << "write log num[log_e:" << error_log_num_.load() << ", log_w:" << warn_log_num_.load() <<
        ", log_i:" << info_log_num_.load() << ", log_d:" << debug_log_num_.load() << "], controlled num[log_e:"
        << flow_control_error_log_num_.load() << ", log_w:" << flow_control_warn_log_num_.load() <<
        ", log_i:" << flow_control_info_log_num_.load() << ", log_d:" << flow_control_debug_num_.load() << "]";
    return str.str();
}

void FlowFuncLoggerImpl::RefreshLimitNum() {
    flow_control_.RefreshLimitNum();
}
}