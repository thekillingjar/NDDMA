/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_THREAD_POOL_H
#define FLOW_FUNC_THREAD_POOL_H

#include <functional>
#include <thread>
#include <vector>
#include <string>
#include "ascend_hal.h"
#include "common/count_down_latch.h"

namespace FlowFunc {
class DeviceCpuInfo {
public:
    int32_t Init(uint32_t device_id);

    uint32_t GetAicpuPhysIndex(uint32_t logic_idx) const {
        if (logic_idx < aicpu_indexes_.size()) {
            return aicpu_indexes_[logic_idx];
        } else if (aicpu_indexes_.empty()) {
            return 0;
        } else {
            return aicpu_indexes_[0];
        }
    }

    uint32_t GetAicpuNum() const {
        return aicpu_num_;
    }

private:
    static int32_t GetCpuNum(uint32_t device_id, DEV_MODULE_TYPE model_type, uint32_t &num);

    // aicpu core num
    uint32_t aicpu_num_;
    std::vector<uint32_t> aicpu_indexes_;
};

class FlowFuncThreadPool {
public:
    explicit FlowFuncThreadPool(std::string thread_name_prefix) : thread_name_prefix_(std::move(thread_name_prefix)) {}

    ~FlowFuncThreadPool();

    int32_t Init(const std::function<void(uint32_t)> &thread_func, uint32_t default_thread_num);

    void WaitForStop() noexcept;

    void WaitAllThreadReady();

    void ThreadReady(uint32_t thread_index);

    void ThreadAbnormal(uint32_t thread_index);

    uint32_t GetThreadNum() const;

    static int32_t ThreadSecureCompute();

private:
    void SetThreadName(uint32_t thread_idx) const;

    static int32_t WriteTidForAffinity();

    void BindAicpu(uint32_t thread_idx) const;

    std::vector<std::thread> workers_;

    std::string thread_name_prefix_;

    DeviceCpuInfo device_cpu_info_;
    CountDownLatch wait_thread_ready_latch_;
};
}
#endif // FLOW_FUNC_THREAD_POOL_H
