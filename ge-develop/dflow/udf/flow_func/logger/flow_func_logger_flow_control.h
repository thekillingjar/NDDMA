/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_LOGGER_FLOW_CONTROL_H
#define FLOW_FUNC_LOGGER_FLOW_CONTROL_H

#include <atomic>
#include "flow_func/flow_func_log.h"

namespace FlowFunc {

class FlowFuncLoggerFlowControl {
public:
    explicit FlowFuncLoggerFlowControl(FlowFuncLogType type) : type_(type) {
    }

    ~FlowFuncLoggerFlowControl() = default;

    void RefreshLimitNum();

    /**
     * @brief acquire log num
     * @return true: acquire success, false: acquire failed
     */
    bool TryAcquire();

private:
    FlowFuncLogType type_;

    std::atomic_flag flow_control_lock{ATOMIC_FLAG_INIT};
    volatile bool is_need_flow_control_ = false;
    volatile bool is_exceed_limit_ = false;
    volatile uint16_t limit_num_ = 0;
};
}

#endif // FLOW_FUNC_LOGGER_FLOW_CONTROL_H
