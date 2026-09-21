/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mbuf_flow_msg_queue.h"
#include "ascend_hal.h"
#include "common/scope_guard.h"
#include "common/udf_log.h"
#include "common/inner_error_codes.h"
#include "flow_func/mbuf_flow_msg.h"
#include "reader_writer/proxy_queue_wrapper.h"
#include "flow_func/flow_func_config_manager.h"

namespace FlowFunc {
namespace {
constexpr int32_t kDequeueWaitPerLoop = 1000;
}
thread_local bool MbufFlowMsgQueue::handle_event_ = false;

MbufFlowMsgQueue::MbufFlowMsgQueue(
    const std::shared_ptr<QueueWrapper> &queue_wrapper, const QueueDevInfo &queue_info)
    : queue_wrapper_(queue_wrapper), queue_info_(queue_info) {
    flow_msg_queue_sched_group_id_ = FlowFuncConfigManager::GetConfig()->GetFlowMsgQueueSchedGroupId();
}

int32_t MbufFlowMsgQueue::Dequeue(std::shared_ptr<FlowMsg> &flow_msg, int32_t timeout) {
    UDF_LOG_DEBUG("Dequeue flow msg, %s, timeout=%d(ms).", queue_wrapper_->GetQueueInfo().c_str(), timeout);
    Mbuf *mbuf = nullptr;
    int32_t ret = DequeueMbuf(mbuf, timeout);
    if ((ret == FLOW_FUNC_ERR_TIME_OUT_ERROR) || (ret == FLOW_FUNC_STATUS_EXIT)) {
        return ret;
    }
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR(
            "dequeue mbuf failed, %s, timeout=%d(ms), ret=%d", queue_wrapper_->GetQueueInfo().c_str(), timeout, ret);
        return ret;
    }

    if (mbuf == nullptr) {
        UDF_LOG_ERROR("dequeue mbuf null, %s, timeout=%d", queue_wrapper_->GetQueueInfo().c_str(), timeout);
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }

    static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
    std::shared_ptr<Mbuf> mbuf_ptr(mbuf, mbuf_deleter);

    auto mbuf_flow_msg = new (std::nothrow) MbufFlowMsg(mbuf_ptr);
    if (mbuf_flow_msg == nullptr) {
        UDF_LOG_ERROR("new MbufFlowMsg failed, %s.", queue_wrapper_->GetQueueInfo().c_str());
        return FLOW_FUNC_FAILED;
    }
    flow_msg.reset(mbuf_flow_msg);
    auto init_ret = mbuf_flow_msg->Init();
    if (init_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("data init failed, ret=%d, %s.", init_ret, queue_wrapper_->GetQueueInfo().c_str());
        return init_ret;
    }
    const auto ret_code = mbuf_flow_msg->GetRetCode();
    if (ret_code != 0) {
        UDF_RUN_LOG_ERROR("Input flow msg is invalid, error code[%d], %s.", ret_code, queue_wrapper_->GetQueueInfo().c_str());
        return ret_code;
    }
    UDF_LOG_DEBUG("Dequeue flow msg success, %s, fetch msg=%s",
        queue_wrapper_->GetQueueInfo().c_str(),
        mbuf_flow_msg->DebugString().c_str());
    return FLOW_FUNC_SUCCESS;
}

int32_t MbufFlowMsgQueue::Depth() const {
    return queue_wrapper_->QueryQueueDepth();;
}

int32_t MbufFlowMsgQueue::Size() const {
    return queue_wrapper_->QueryQueueSize();
}

bool MbufFlowMsgQueue::StatusOk() const {
    return !queue_failed_;
}

void MbufFlowMsgQueue::DiscardAllInputData() {
    queue_failed_ = false;
    if (queue_wrapper_->DiscardMbuf() != HICAID_SUCCESS) {
        queue_failed_ = true;
        UDF_LOG_ERROR("Discard all input data failed, %s", queue_wrapper_->GetQueueInfo().c_str());
    }
}

int32_t MbufFlowMsgQueue::DequeueMbuf(Mbuf *&mbuf) const {
    int32_t hicaid_ret = queue_wrapper_->Dequeue(mbuf);
    if (hicaid_ret == HICAID_SUCCESS) {
        UDF_LOG_DEBUG("dequeue success, %s", queue_wrapper_->GetQueueInfo().c_str());
        return FLOW_FUNC_SUCCESS;
    } else if (hicaid_ret == HICAID_ERR_QUEUE_EMPTY) {
        UDF_LOG_DEBUG("queue is empty, %s", queue_wrapper_->GetQueueInfo().c_str());
        return FLOW_FUNC_ERR_QUEUE_EMPTY;
    } else {
        UDF_LOG_ERROR("Dequeue failed, hicaid_ret=%d, %s", hicaid_ret, queue_wrapper_->GetQueueInfo().c_str());
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }
}

int32_t MbufFlowMsgQueue::DequeueMbuf(Mbuf *&mbuf, int32_t timeout) {
    std::unique_lock<std::mutex> lock(dequeue_mutex_);
    int32_t ret = FLOW_FUNC_SUCCESS;
    if (timeout == 0) {
        ret = DequeueMbuf(mbuf);
        if (ret == FLOW_FUNC_ERR_QUEUE_EMPTY) {
            UDF_LOG_WARN("Queue is empty and timeout=0, %s", queue_wrapper_->GetQueueInfo().c_str());
            ret = FLOW_FUNC_ERR_TIME_OUT_ERROR;
        }
        return ret;
    }

    ret = SubQueueEnqueEvent();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("sub queue enqueue event failed, queue_id=%u", queue_info_.queue_id);
        return ret;
    }

    std::function<void()> unsub_func = [this]() {
        (void)UnsubQueueEnqueEvent();
        SwapOutFlowMsgQueueEventGroup();
    };
    // make sure unsub on exit.
    const ScopeGuard unsub_guard(unsub_func);

    ret = DequeueMbuf(mbuf);
    if (ret != FLOW_FUNC_ERR_QUEUE_EMPTY) {
        return ret;
    }

    SwapOutGlobalGroup();

    struct event_info event = {};
    int64_t wait_time = 0L;
    const uint32_t thread_idx = FlowFuncConfigManager::GetConfig()->GetCurrentSchedThreadIdx();
    do {
        if (queue_info_.is_proxy_queue) {
            wait_time += kProxyQueueDefaultDequeueTimeout;
            // proxy queue no event
        } else {
            handle_event_ = false;
            drvError_t drv_ret = halEschedWaitEvent(FlowFuncConfigManager::GetConfig()->GetDeviceId(), flow_msg_queue_sched_group_id_,
                thread_idx, kDequeueWaitPerLoop, &event);
            if (drv_ret == DRV_ERROR_NONE) {
                handle_event_ = true;
                FlowFuncConfigManager::GetConfig()->SetCurrentSchedGroupId(flow_msg_queue_sched_group_id_);
                UDF_LOG_DEBUG("receive event, event_id=%d, subevent_id=%u ",
                    static_cast<int32_t>(event.comm.event_id), event.comm.subevent_id);
            } else if (drv_ret == DRV_ERROR_SCHED_WAIT_TIMEOUT) {
                wait_time += kDequeueWaitPerLoop;
                continue;
            } else {
                // LOG ERROR
                UDF_LOG_ERROR("wait event failed, device_id=%u, threadIndex=%u, groupId=%u, drv_ret=%d.",
                    FlowFuncConfigManager::GetConfig()->GetDeviceId(), thread_idx, flow_msg_queue_sched_group_id_,
                    static_cast<int32_t>(drv_ret));
                return FLOW_FUNC_ERR_DRV_ERROR;
            }
        }
        ret = DequeueMbuf(mbuf);
        if (ret != FLOW_FUNC_ERR_QUEUE_EMPTY) {
            return ret;
        }
    } while (((wait_time < timeout) || (timeout == -1)) && (!FlowFuncConfigManager::GetConfig()->GetAbnormalStatus()) &&
             (!FlowFuncConfigManager::GetConfig()->GetExitFlag()));
    if (FlowFuncConfigManager::GetConfig()->GetAbnormalStatus()) {
        UDF_LOG_ERROR("Stop dequeue result of now system status is abnormal. Wait to redeploy.");
        return FLOW_FUNC_STATUS_REDEPLOYING;
    }
    if (FlowFuncConfigManager::GetConfig()->GetExitFlag()) {
        UDF_LOG_INFO("Receive term signal, stop dequeue.");
        return FLOW_FUNC_STATUS_EXIT;
    }
    UDF_LOG_WARN(
        "wait event timeout, wait_time=%ld, timeout=%d(ms), %s", wait_time, timeout, queue_wrapper_->GetQueueInfo().c_str());
    return FLOW_FUNC_ERR_TIME_OUT_ERROR;
}

int32_t MbufFlowMsgQueue::SubQueueEnqueEvent() const {
    // proxy queue no need sub queue event
    if (queue_info_.is_proxy_queue) {
        return FLOW_FUNC_SUCCESS;
    }
    struct QueueSubPara queue_sub_para{};
    queue_sub_para.devId = queue_info_.device_id;
    queue_sub_para.qid = queue_info_.queue_id;
    queue_sub_para.queType = QUEUE_TYPE_SINGLE;
    queue_sub_para.eventType = QUEUE_ENQUE_EVENT;
    queue_sub_para.groupId = flow_msg_queue_sched_group_id_;
    queue_sub_para.flag = QUEUE_SUB_FLAG_SPEC_THREAD;
    queue_sub_para.threadId = FlowFuncConfigManager::GetConfig()->GetCurrentSchedThreadIdx();
    drvError_t drv_ret = halQueueSubEvent(&queue_sub_para);
    if ((drv_ret != DRV_ERROR_NONE) && (drv_ret != DRV_ERROR_REPEATED_SUBSCRIBED)) {
        UDF_LOG_ERROR("queue sub event failed, drv_ret=%d, queue_id=%u, queueEventType=%d.",
            static_cast<int32_t>(drv_ret), queue_info_.queue_id, static_cast<int32_t>(QUEUE_ENQUE_EVENT));
        return FLOW_FUNC_ERR_EVENT_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t MbufFlowMsgQueue::UnsubQueueEnqueEvent() const {
    // proxy queue no need un sub queue event
    if (queue_info_.is_proxy_queue) {
        return FLOW_FUNC_SUCCESS;
    }
    struct QueueUnsubPara queue_unsub_para{};
    queue_unsub_para.devId = FlowFuncConfigManager::GetConfig()->GetDeviceId();
    queue_unsub_para.qid = queue_info_.queue_id;
    queue_unsub_para.eventType = QUEUE_ENQUE_EVENT;
    drvError_t drv_ret = halQueueUnsubEvent(&queue_unsub_para);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("queue unsub event failed, drv_ret=%d, queue_id=%u, queueEventType=%d.",
            static_cast<int32_t>(drv_ret), queue_info_.queue_id, static_cast<int32_t>(QUEUE_ENQUE_EVENT));
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

void MbufFlowMsgQueue::SwapOutGlobalGroup() const {
    const uint32_t thread_idx = FlowFuncConfigManager::GetConfig()->GetCurrentSchedThreadIdx();
    const uint32_t current_sched_group_id = FlowFuncConfigManager::GetConfig()->GetCurrentSchedGroupId();
    if (current_sched_group_id != flow_msg_queue_sched_group_id_) {
        auto drv_ret = halEschedThreadSwapout(FlowFuncConfigManager::GetConfig()->GetDeviceId(), current_sched_group_id, thread_idx);
        if (drv_ret != DRV_ERROR_NONE) {
            UDF_LOG_ERROR("halEschedThreadSwapout failed, groupId=%u, thread_idx=%u, drv_ret=%d",
                current_sched_group_id, thread_idx, static_cast<int32_t>(drv_ret));
        }
    }
}

void MbufFlowMsgQueue::SwapOutFlowMsgQueueEventGroup() const {
    // no event is handling, no need swap out.
    if (!handle_event_) {
        return;
    }
    handle_event_ = false;
    const uint32_t thread_idx = FlowFuncConfigManager::GetConfig()->GetCurrentSchedThreadIdx();
    auto drv_ret = halEschedThreadSwapout(FlowFuncConfigManager::GetConfig()->GetDeviceId(), flow_msg_queue_sched_group_id_, thread_idx);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("halEschedThreadSwapout failed, groupId=%u, thread_idx=%u, drv_ret=%d",
            flow_msg_queue_sched_group_id_, thread_idx, static_cast<int32_t>(drv_ret));
    }
}
}  // namespace FlowFunc