/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_timer.h"
#include "ascend_hal.h"
#include "common/common_define.h"
#include "common/udf_log.h"
#include "common/rw_lock.h"
#include "common/util.h"
#include "flow_func_config_manager.h"

namespace FlowFunc {
constexpr uint32_t kFlowFuncTimerWaitInterval = 2U;
FlowFuncTimer &FlowFuncTimer::Instance() {
    static FlowFuncTimer inst;
    return inst;
}

void FlowFuncTimer::Init(uint32_t device_id) {
    (void)mmRWLockInit(&timer_guard_);
    running_ = true;
    device_id_ = device_id;
    UDF_LOG_DEBUG("FlowFuncTimer Init enter. running_=%u.", running_);
    timer_thread_ = std::thread(&FlowFuncTimer::TimerThreadLoop, this);
}

void FlowFuncTimer::Finalize() {
    running_ = false;
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }
    (void)mmRWLockDestroy(&timer_guard_);
}

uint64_t FlowFuncTimer::GetCurrentTimestamp() {
    struct timeval tv{};
    int32_t ret = gettimeofday(&tv, nullptr);
    if (ret != 0) {
        UDF_LOG_ERROR("Func gettimeofday may failed, ret:%d", ret);
    }
    const auto totalUsedTime = tv.tv_usec + (tv.tv_sec * 1000000);  // 1000000: seconds to microseconds
    return static_cast<uint64_t>(totalUsedTime);
}

void *FlowFuncTimer::CreateTimer(const TimerCallback &callback, bool invoke_by_worker) {
    static uint32_t timer_cnt = 0U;
    if (callback == nullptr) {
        UDF_LOG_ERROR("Timer callback is null.");
        return nullptr;
    }
    const WriteProtect wp(&timer_guard_);
    if (timer_cnt == UINT32_MAX) {
        UDF_LOG_ERROR("Create timer reaches maximum limit[%u].", UINT32_MAX);
        return nullptr;
    }
    std::shared_ptr<TimerInfo> timer = MakeShared<TimerInfo>();
    if (timer == nullptr) {
        UDF_LOG_ERROR("Failed to create timer.");
        return nullptr;
    }
    timer->timer_id = timer_cnt;
    timer->timer_callback = callback;
    timer->invoke_by_worker = invoke_by_worker;
    timer_info_map_[timer_cnt] = timer;
    ++timer_cnt;

    UDF_LOG_DEBUG("CreateTimer success. timer_id=%u", timer->timer_id);
    return static_cast<void*>(timer.get());
}

int32_t FlowFuncTimer::DeleteTimer(void *handle) {
    auto timer = static_cast<TimerInfo*>(handle);
    if (timer == nullptr) {
        UDF_LOG_ERROR("Timer handle is null, return failed.");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    auto del_timer_id = timer->timer_id;
    const WriteProtect wp(&timer_guard_);
    std::map<uint32_t, std::shared_ptr<TimerInfo>>::const_iterator iter = timer_info_map_.find(del_timer_id);
    if (iter == timer_info_map_.cend()) {
        UDF_LOG_ERROR("cannot find timer info, delete timer[%u] failed.", del_timer_id);
        return FLOW_FUNC_FAILED;
    }
    UDF_LOG_DEBUG("begin to delete timer, timer_id=%u.", del_timer_id);
    (void)timer_info_map_.erase(iter);
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncTimer::StartTimer(void *handle, uint32_t period, bool oneshot) {
    UDF_LOG_DEBUG("StartTimer enter. period=%u, oneshot=%u", period, oneshot);
    auto timer = static_cast<TimerInfo*>(handle);
    if (timer == nullptr) {
        UDF_LOG_ERROR("Timer handle is null. ");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    const ReadProtect rp(&timer_guard_);
    if (timer_info_map_.find(timer->timer_id) == timer_info_map_.end()) {
        UDF_LOG_ERROR("cannot find timer info, start timer[%u] failed.", timer->timer_id);
        return FLOW_FUNC_FAILED;
    }
    timer->period = period;
    timer->oneshot_flag = oneshot;
    timer->is_start = true;
    timer->next_timestamp = GetCurrentTimestamp() + static_cast<uint64_t>(period) * kMsToUsCast;
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncTimer::StopTimer(void *handle) {
    auto timer = static_cast<TimerInfo*>(handle);
    if (timer == nullptr) {
        UDF_LOG_ERROR("Timer handle is null. ");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    const ReadProtect rp(&timer_guard_);
    if (timer_info_map_.find(timer->timer_id) == timer_info_map_.end()) {
        UDF_LOG_ERROR("cannot find timer info, stop timer[%u] failed.", timer->timer_id);
        return FLOW_FUNC_FAILED;
    }
    timer->is_start = false;
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncTimer::TriggerTimerEvent(uint32_t timer_id) const {
    UDF_LOG_DEBUG("Trigger timer event, timer_id=%u.", timer_id);
    event_summary event_info_summary = {};
    event_info_summary.pid = getpid();
    event_info_summary.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdTimer);
    event_info_summary.subevent_id = timer_id;
    event_info_summary.msg = nullptr;
    event_info_summary.msg_len = 0U;
    event_info_summary.dst_engine = FlowFuncConfigManager::GetConfig()->IsRunOnAiCpu() ? ACPU_LOCAL : CCPU_LOCAL;
    event_info_summary.grp_id = FlowFuncConfigManager::GetConfig()->GetMainSchedGroupId();

    drvError_t ret = halEschedSubmitEvent(device_id_, &event_info_summary);
    if (ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("Failed to submit timer event. drvRet=%d.", static_cast<int32_t>(ret));
    }
}

void FlowFuncTimer::ExecCallBack(uint32_t timer_id) {
    const ReadProtect rp(&timer_guard_);
    if (timer_info_map_.empty()) {
        UDF_LOG_ERROR("timer_info_map_ is empty.");
        return;
    }
    auto iter = timer_info_map_.find(timer_id);
    if (iter == timer_info_map_.end()) {
        UDF_LOG_ERROR("timer_id is invalid.");
        return;
    }
    (*iter).second->timer_callback();
}
void FlowFuncTimer::ProcessEachTimerInContext() {
    const auto current_time_stamp = GetCurrentTimestamp();
    std::vector<TimerCallback> inner_call_back_list;
    {
        const ReadProtect rp(&timer_guard_);
        for (const auto &iter : timer_info_map_) {
            auto timerInfo = iter.second;
            if (!(timerInfo->is_start) || (current_time_stamp < timerInfo->next_timestamp)) {
                continue;
            }
            if (timerInfo->invoke_by_worker) {
                TriggerTimerEvent(iter.first);
            } else {
                inner_call_back_list.emplace_back(iter.second->timer_callback);
            }
            if (timerInfo->oneshot_flag) {
                timerInfo->is_start = false;
            } else {
                timerInfo->next_timestamp = current_time_stamp + static_cast<uint64_t>(timerInfo->period) * kMsToUsCast;
            }
        }
    }

    for (const auto &callback:inner_call_back_list) {
        callback();
    }
}

void FlowFuncTimer::TimerThreadLoop() {
    constexpr const char *kThreadName = "udf_timer";
    int32_t ret = pthread_setname_np(pthread_self(), kThreadName);
    if (ret != 0) {
        UDF_LOG_WARN("pthread_setname_np failed, kThreadName=%s, ret=%d", kThreadName, ret);
    }
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kFlowFuncTimerWaitInterval));
        ProcessEachTimerInContext();
    }
    UDF_RUN_LOG_INFO("Timer loop thread exit.");
}
}