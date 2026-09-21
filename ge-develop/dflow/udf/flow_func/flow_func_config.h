/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_FLOW_FUNC_CONFIG_H
#define UDF_FLOW_FUNC_CONFIG_H

#include <cstdint>
#include <functional>
#include "flow_func/flow_func_defines.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowFuncConfig {
public:
    virtual bool IsOnDevice() const = 0;

    virtual bool IsRunOnAiCpu() const = 0;

    virtual uint32_t GetDeviceId() const = 0;

    virtual int32_t GetPhyDeviceId() const = 0;

    virtual uint32_t GetMainSchedGroupId() const = 0;

    virtual int32_t GetMemGroupId() const = 0;

    virtual std::function<int32_t()> GetLimitThreadInitFunc() const = 0;

    virtual uint32_t GetWorkerSchedGroupId() const = 0;

    virtual bool GetAbnormalStatus() const = 0;

    virtual uint32_t GetCurrentSchedThreadIdx() const = 0;

    virtual void SetCurrentSchedGroupId(uint32_t current_sched_group_id) = 0;

    virtual uint32_t GetCurrentSchedGroupId() const = 0;

    virtual uint32_t GetFlowMsgQueueSchedGroupId() const = 0;

    virtual uint32_t GetWorkerNum() const = 0;

    virtual bool GetExitFlag() const = 0;
protected:
    FlowFuncConfig() = default;

    virtual ~FlowFuncConfig() = default;
};
}

#endif // UDF_FLOW_FUNC_CONFIG_H
