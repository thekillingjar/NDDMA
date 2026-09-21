/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_TIMER_H
#define FLOW_FUNC_TIMER_H

#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <functional>
#include "flow_func/flow_func_defines.h"
#include "mmpa/mmpa_api.h"
namespace FlowFunc {
using TimerCallback = std::function<void(void)>;
constexpr uint32_t kMsToUsCast = 1000U;

struct TimerInfo {
    uint64_t next_timestamp;
    uint32_t period;
    uint32_t timer_id;
    bool oneshot_flag;
    bool is_start;
    bool invoke_by_worker;
    uint8_t rsv;
    TimerCallback timer_callback;
};

class FLOW_FUNC_VISIBILITY FlowFuncTimer {
public:
    static FlowFuncTimer &Instance();
    static uint64_t GetCurrentTimestamp();
    void Init(uint32_t device_id);
    void Finalize();
    /**
     * @brief create timer.
     * @param callback timer callback
     * @param invoke_by_worker
     *  true: send event to worker thread to invoke,
     *  false: invoke by timer
     * @return
     */
    void *CreateTimer(const TimerCallback &callback, bool invoke_by_worker = true);
    int32_t DeleteTimer(void *handle);
    int32_t StartTimer(void *handle, uint32_t period, bool oneshot);
    int32_t StopTimer(void *handle);
    void ExecCallBack(uint32_t timer_id);

private:
    FlowFuncTimer() = default;
    ~FlowFuncTimer() = default;
    void ProcessEachTimerInContext();
    void TimerThreadLoop();
    void TriggerTimerEvent(uint32_t timer_id) const;
    mmRWLock_t timer_guard_{};
    std::thread timer_thread_;
    volatile bool running_ = false;
    std::map<uint32_t, std::shared_ptr<TimerInfo>> timer_info_map_;
    uint32_t device_id_ = 0U;
};
}
#endif // FLOW_FUNC_TIMER_H
