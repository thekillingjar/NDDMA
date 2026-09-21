/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_LOGGER_FLOW_FUNC_LOGGER_MANAGER_H
#define FLOW_FUNC_LOGGER_FLOW_FUNC_LOGGER_MANAGER_H

#include "flow_func/flow_func_log.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowFuncLoggerManager {
public:
    static FlowFuncLoggerManager &Instance();

    FlowFuncLogger &GetLogger(FlowFuncLogType type) const;

    int32_t Init();

    void Finalize() noexcept;

    void DumpStatistic() const;

private:
    FlowFuncLoggerManager() = default;

    ~FlowFuncLoggerManager();

    void RefreshFlowControl() const;

    void *logger_timer_handle_ = nullptr;
};
}
#endif // FLOW_FUNC_LOGGER_FLOW_FUNC_LOGGER_MANAGER_H
