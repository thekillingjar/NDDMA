/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_config_manager.h"
#include "common/udf_log.h"

namespace FlowFunc {
std::shared_ptr<FlowFuncConfig> FlowFuncConfigManager::config_ = nullptr;

class FlowFuncConfigDefault : public FlowFuncConfig {
public:
    FlowFuncConfigDefault() = default;

    ~FlowFuncConfigDefault() override = default;

    bool IsOnDevice() const override {
        return false;
    }

    bool IsRunOnAiCpu() const override {
        return false;
    }

    uint32_t GetDeviceId() const override {
        return 0U;
    }

    int32_t GetPhyDeviceId() const override {
        return -1;
    }

    uint32_t GetMainSchedGroupId() const override {
        return 0U;
    }

    int32_t GetMemGroupId() const override {
        return 0U;
    }

    std::function<int32_t()> GetLimitThreadInitFunc() const override {
        return nullptr;
    }

    uint32_t GetWorkerSchedGroupId() const override {
        return 0U;
    }

    bool GetAbnormalStatus() const override {
        return false;
    }

    uint32_t GetCurrentSchedThreadIdx() const override {
        return 0U;
    }

    void SetCurrentSchedGroupId(uint32_t current_sched_group_id) override {
        (void)current_sched_group_id; 
    }

    uint32_t GetCurrentSchedGroupId() const override {
        return 0U;
    }

    uint32_t GetFlowMsgQueueSchedGroupId() const override {
        return 0U;
    }

    uint32_t GetWorkerNum() const override {
        return 0U;
    }

    bool GetExitFlag() const override {
        return false;
    }
};

const std::shared_ptr<FlowFuncConfig> &FlowFuncConfigManager::GetConfig() {
    if (config_ != nullptr) {
        return config_;
    }
    static FlowFuncConfigDefault default_config;
    static std::shared_ptr<FlowFuncConfig> default_config_ptr(&default_config, [](FlowFuncConfig *) {});
    return default_config_ptr;
}

void FlowFuncConfigManager::SetConfig(const std::shared_ptr<FlowFuncConfig> &config) {
    if (config_ == nullptr) {
        config_ = config;
    } else {
        UDF_LOG_WARN("repeat init");
    }
}
}
