/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_logger_flow_control.h"
#include "dlog_pub.h"
#include "common/udf_log.h"

namespace FlowFunc {
void FlowFuncLoggerFlowControl::RefreshLimitNum() {
    uint16_t replenish_num = 0U;
    uint16_t max_limit_num = 0U;
    if (type_ == FlowFuncLogType::DEBUG_LOG) {
        int enable_event = 0;
        // default is debug log type
        int32_t dlog_level = dlog_getlevel(static_cast<int32_t>(APP), &enable_event);
        switch (dlog_level) {
            case DLOG_INFO:
                // info log level: limit 100/s, transient limit 1000.
                replenish_num = 100U;
                max_limit_num = 1000U;
                break;
            case DLOG_WARN:
            case DLOG_ERROR:
                // error and warn log limit 50/s, transient limit 400.
                replenish_num = 50U;
                max_limit_num = 400U;
                break;
            default:
                is_need_flow_control_ = false;
                return;
        }
    } else {
        // run log limit 50/s, transient limit 400.
        replenish_num = 50U;
        max_limit_num = 400U;
    }

    bool control_resume = false;
    {
        while (flow_control_lock.test_and_set(std::memory_order_acquire)) {
            // spin lock
        }
        // if first need flow control, set it to max limit num.
        if (!is_need_flow_control_) {
            limit_num_ = max_limit_num;
            is_need_flow_control_ = true;
        } else {
            limit_num_ += replenish_num;
            if (limit_num_ > max_limit_num) {
                limit_num_ = max_limit_num;
            }
        }
        if (is_exceed_limit_) {
            control_resume = true;
            is_exceed_limit_ = false;
        }
        flow_control_lock.clear(std::memory_order_release);
    }

    if (control_resume) {
        UDF_LOG_INFO("flow func [%s] log flow control resume.", type_ == FlowFuncLogType::DEBUG_LOG ? "DEBUG" : "RUN");
    }
}

bool FlowFuncLoggerFlowControl::TryAcquire() {
    if (!is_need_flow_control_) {
        return true;
    }

    bool acquire_ok = false;
    bool first_control = false;
    {
        while (flow_control_lock.test_and_set(std::memory_order_acquire)) {
            // spin lock
        }

        if (!is_exceed_limit_) {
            if (limit_num_ > 0) {
                --limit_num_;
                acquire_ok = true;
            } else {
                is_exceed_limit_ = true;
                first_control = true;
            }
        }
        flow_control_lock.clear(std::memory_order_release);
    }
    if (first_control) {
        UDF_RUN_LOG_WARN("flow func [%s] log is flow controlled.",
            type_ == FlowFuncLogType::DEBUG_LOG ? "DEBUG" : "RUN");
    }
    return acquire_ok;
}
}