/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_logger_manager.h"
#include "flow_func_logger_impl.h"
#include "flow_func/flow_func_timer.h"
#include "common/udf_log.h"

namespace FlowFunc {
namespace {
FlowFuncLoggerImpl g_run_logger(FlowFuncLogType::RUN_LOG);
FlowFuncLoggerImpl g_debug_logger(FlowFuncLogType::DEBUG_LOG);
}

FlowFuncLoggerManager &FlowFuncLoggerManager::Instance() {
    static FlowFuncLoggerManager instance;
    return instance;
}

FlowFuncLoggerManager::~FlowFuncLoggerManager() {
    Finalize();
}

FlowFuncLogger &FlowFuncLoggerManager::GetLogger(FlowFuncLogType type) const {
    if (type == FlowFuncLogType::RUN_LOG) {
        return g_run_logger;
    } else {
        return g_debug_logger;
    }
}

void FlowFuncLoggerManager::RefreshFlowControl() const {
    g_run_logger.RefreshLimitNum();
    g_debug_logger.RefreshLimitNum();
}

void FlowFuncLoggerManager::DumpStatistic() const {
    UDF_RUN_LOG_INFO("flow func log statistic, debugLog={%s}, runLog={%s}", g_debug_logger.StatisticInfo().c_str(),
        g_run_logger.StatisticInfo().c_str());
}

int32_t FlowFuncLoggerManager::Init() {
    logger_timer_handle_ = FlowFuncTimer::Instance().CreateTimer([this]() { RefreshFlowControl(); });
    if (logger_timer_handle_ == nullptr) {
        UDF_LOG_ERROR("create timer for logger manager failed.");
        return FLOW_FUNC_FAILED;
    }
    // refresh once
    RefreshFlowControl();
    auto ret = FlowFuncTimer::Instance().StartTimer(logger_timer_handle_, 1000U, false);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("failed to start log timer, ret=%d.", ret);
        return ret;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncLoggerManager::Finalize() noexcept {
    if (logger_timer_handle_ != nullptr) {
        (void)FlowFuncTimer::Instance().StopTimer(logger_timer_handle_);
        FlowFuncTimer::Instance().DeleteTimer(logger_timer_handle_);
        logger_timer_handle_ = nullptr;
        DumpStatistic();
    }
}

}