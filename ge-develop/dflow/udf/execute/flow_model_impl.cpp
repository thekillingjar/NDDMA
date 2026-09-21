/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_model_impl.h"
#include <functional>
#include "common/udf_log.h"
#include "common/scope_guard.h"
#include "common/util.h"
#include "flow_func/mbuf_flow_msg.h"
#include "config/global_config.h"
#include "reader_writer/queue_wrapper.h"
#include "reader_writer/proxy_queue_wrapper.h"
#include "common/inner_error_codes.h"

namespace FlowFunc {
namespace {
constexpr int32_t kDequeWaitPerLoop = 1000;
// attach time out 3s
constexpr int32_t kQueueAttachTimeout = 3 * 1000;

std::vector<int32_t> GetQueueSize(const std::vector<std::unique_ptr<QueueWrapper>> &queue_wrappers) {
    std::vector<int32_t> queue_size_list(queue_wrappers.size());
    for (size_t idx = 0; idx < queue_wrappers.size(); ++idx) {
        queue_size_list[idx] = queue_wrappers[idx]->QueryQueueSize();
    }
    return queue_size_list;
}
}
thread_local bool FlowModelImpl::handle_event_ = false;

FlowModelImpl::FlowModelImpl(std::vector<QueueDevInfo> input_queue_infos, std::vector<QueueDevInfo> output_queue_infos,
    std::unique_ptr<DataAligner> data_aligner)
    : input_queue_infos_(std::move(input_queue_infos)), output_queue_infos_(std::move(output_queue_infos)),
    data_aligner_(std::move(data_aligner)) {
}

int32_t FlowModelImpl::Init() {
    invoke_model_sched_group_id_ = GlobalConfig::Instance().GetInvokeModelSchedGroupId();
    input_queue_wrappers_.resize(input_queue_infos_.size());
    output_queue_wrappers_.resize(output_queue_infos_.size());
    for (size_t input_idx = 0; input_idx < input_queue_infos_.size(); ++input_idx) {
        const auto &input_queue_info = input_queue_infos_[input_idx];
        auto dev_ret = halQueueAttach(input_queue_info.device_id, input_queue_info.queue_id, kQueueAttachTimeout);
        if (dev_ret != DRV_ERROR_NONE) {
            UDF_LOG_ERROR("attached feed input queue[%u] failed, ret[%d], queueDeviceId[%u]", input_queue_info.queue_id,
                static_cast<int32_t>(dev_ret), input_queue_info.device_id);
            return FLOW_FUNC_ERR_QUEUE_ERROR;
        }
        if (input_queue_info.is_proxy_queue) {
            input_queue_wrappers_[input_idx].reset(
                new(std::nothrow) ProxyQueueWrapper(input_queue_info.device_id, input_queue_info.queue_id));
        } else {
            input_queue_wrappers_[input_idx].reset(
                new(std::nothrow) QueueWrapper(input_queue_info.device_id, input_queue_info.queue_id));
        }
        if (input_queue_wrappers_[input_idx] == nullptr) {
            UDF_LOG_ERROR("alloc queueWrapper failed, is_proxy_queue=%d, queue_id=%u.",
                static_cast<int32_t>(input_queue_info.is_proxy_queue), input_queue_info.queue_id);
            return FLOW_FUNC_FAILED;
        }
    }

    QueueSetInput input;
    QueueSetInputPara input_param;
    for (size_t output_idx = 0; output_idx < output_queue_infos_.size(); ++output_idx) {
        const auto &output_queue_info = output_queue_infos_[output_idx];
        auto output_queue_id = output_queue_info.queue_id;
        auto dev_ret = halQueueAttach(output_queue_info.device_id, output_queue_info.queue_id, kQueueAttachTimeout);
        if (dev_ret != DRV_ERROR_NONE) {
            UDF_LOG_ERROR("attached fetch output queue[%u] failed, ret[%d], queueDeviceId[%u]", output_queue_id,
                static_cast<int32_t>(dev_ret), output_queue_info.queue_id);
            return FLOW_FUNC_ERR_QUEUE_ERROR;
        }
        if (output_queue_info.is_proxy_queue) {
            output_queue_wrappers_[output_idx].reset(
                new(std::nothrow) ProxyQueueWrapper(output_queue_info.device_id, output_queue_id));
        } else {
            // flow model output is as input by udf and proxy queue no need set work mode
            input.queSetWorkMode.qid = output_queue_info.queue_id;
            input.queSetWorkMode.workMode = static_cast<uint32_t>(QUEUE_MODE_PULL);
            input_param.inBuff = static_cast<void *>(&input);
            input_param.inLen = static_cast<uint32_t>(sizeof(QueueSetInput));
            (void)halQueueSet(output_queue_info.device_id, QUEUE_SET_WORK_MODE, &input_param);

            output_queue_wrappers_[output_idx].reset(
                new(std::nothrow) QueueWrapper(output_queue_info.device_id, output_queue_id));
        }
        if (output_queue_wrappers_[output_idx] == nullptr) {
            UDF_LOG_ERROR("alloc output queueWrapper failed, is_proxy_queue=%d, queue_id=%u.",
                static_cast<int32_t>(output_queue_info.is_proxy_queue), output_queue_id);
            return FLOW_FUNC_FAILED;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowModelImpl::Run(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
    std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) {
    UDF_LOG_DEBUG("run flow model, input_msgs size=%zu, timeout=%d(ms)", input_msgs.size(), timeout);
    if (input_msgs.size() != input_queue_infos_.size()) {
        UDF_LOG_ERROR("input msgs num=%zu is not same as input queue num=%zu", input_msgs.size(),
            input_queue_infos_.size());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (data_aligner_ != nullptr) {
        data_aligner_->Reset();
    }
    std::unique_lock<std::mutex> guard(model_mutex_);
    for (size_t input_idx = 0UL; input_idx < input_queue_infos_.size(); ++input_idx) {
        int32_t feed_ret = Feed(input_idx, input_msgs[input_idx], timeout);
        if (feed_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("feed flow msg failed, queue id=%u, input_idx=%zu", input_queue_infos_[input_idx].queue_id,
                input_idx);
            return feed_ret;
        }
        UDF_LOG_DEBUG("feed flow msg success, queue id=%u, input_idx=%zu", input_queue_infos_[input_idx].queue_id,
            input_idx);
    }

    if (data_aligner_ != nullptr) {
        return AlignFetch(output_msgs, timeout);
    }
    for (size_t output_idx = 0UL; output_idx < output_queue_infos_.size(); ++output_idx) {
        std::shared_ptr<FlowMsg> flow_msg;
        int32_t fetch_ret = Fetch(output_idx, flow_msg, timeout);
        if (fetch_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("fetch flow msg failed, queue id=%u, output_idx=%zu", output_queue_infos_[output_idx].queue_id,
                output_idx);
            return fetch_ret;
        }
        UDF_LOG_DEBUG("fetch flow msg success, queue id=%u, output_idx=%zu", output_queue_infos_[output_idx].queue_id,
            output_idx);
        output_msgs.emplace_back(flow_msg);
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowModelImpl::Feed(size_t input_idx, const std::shared_ptr<FlowMsg> &flow_msg, int32_t timeout) {
    const auto &input_queue_info = input_queue_infos_[input_idx];
    UDF_LOG_DEBUG("feed flow msg, input_idx=%zu, qid=%u, timeout=%d(ms).", input_idx, input_queue_info.queue_id, timeout);
    auto mbuf_flow_msg = std::dynamic_pointer_cast<MbufFlowMsg>(flow_msg);
    if (mbuf_flow_msg == nullptr) {
        UDF_LOG_ERROR("not support custom define flow msg now, input_idx=%zu, qid=%u.", input_idx,
            input_queue_info.queue_id);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    current_trans_id_ = mbuf_flow_msg->GetTransactionId();

    UDF_LOG_DEBUG("input_idx=%zu, qid=%u, feed msg=%s", input_idx, input_queue_info.queue_id,
        mbuf_flow_msg->DebugString().c_str());
    Mbuf *feed_mbuf = nullptr;
    auto drv_int_ret = halMbufCopyRef(mbuf_flow_msg->GetMbuf(), &feed_mbuf);
    if ((drv_int_ret != static_cast<int32_t>(DRV_ERROR_NONE)) || (feed_mbuf == nullptr)) {
        UDF_LOG_ERROR("copy ref mbuf failed, input_idx=%zu, qid=%u.", input_idx, input_queue_info.queue_id);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
    std::unique_ptr<Mbuf, decltype(mbuf_deleter)> mbuf_guard(feed_mbuf, mbuf_deleter);

    int32_t hicard_ret = input_queue_wrappers_[input_idx]->Enqueue(feed_mbuf);
    if (hicard_ret == HICAID_SUCCESS) {
        (void)mbuf_guard.release();
        return FLOW_FUNC_SUCCESS;
    }

    if (hicard_ret == HICAID_ERR_QUEUE_FULL) {
        UDF_LOG_ERROR("Enqueue failed as queue full, input_idx=%zu, qid=%u", input_idx, input_queue_info.queue_id);
        return FLOW_FUNC_FAILED;
    }
    UDF_LOG_ERROR("Enqueue failed, hicard_ret=%d, input_idx=%zu, qid=%u", hicard_ret, input_idx, input_queue_info.queue_id);
    return FLOW_FUNC_ERR_QUEUE_ERROR;
}

int32_t FlowModelImpl::DequeueMbuf(size_t output_idx, Mbuf *&mbuf) const {
    uint32_t qid = output_queue_infos_[output_idx].queue_id;
    int32_t hicaid_ret = output_queue_wrappers_[output_idx]->Dequeue(mbuf);
    if (hicaid_ret == HICAID_SUCCESS) {
        UDF_LOG_DEBUG("dequeue for queue[%u] success", qid);
        return FLOW_FUNC_SUCCESS;
    } else if (hicaid_ret == HICAID_ERR_QUEUE_EMPTY) {
        UDF_LOG_DEBUG("queue is empty, queue_id=%u", qid);
        return FLOW_FUNC_ERR_QUEUE_EMPTY;
    } else {
        UDF_LOG_ERROR("Dequeue failed, hicaid_ret=%d, queue_id=%u", hicaid_ret, qid);
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }
}

int32_t FlowModelImpl::WaitAndHandleEvent(bool &is_continue) const {
    is_continue = false;
    struct event_info event = {};
    handle_event_ = false;
    drvError_t dev_ret = halEschedWaitEvent(GlobalConfig::Instance().GetDeviceId(), invoke_model_sched_group_id_,
        curr_sched_thread_id_, kDequeWaitPerLoop, &event);
    if (dev_ret == DRV_ERROR_NONE) {
        handle_event_ = true;
        GlobalConfig::Instance().SetCurrentSchedGroupId(invoke_model_sched_group_id_);
        UDF_LOG_DEBUG("receive event, event_id=%d, subevent_id=%u ",
            static_cast<int32_t>(event.comm.event_id), event.comm.subevent_id);
        // wake up and check exception
        if (event.comm.event_id == UdfEvent::kEventIdWakeUp) {
            is_continue = true;
            return FLOW_FUNC_SUCCESS;
        }
    } else if (dev_ret == DRV_ERROR_SCHED_WAIT_TIMEOUT) {
        is_continue = true;
        return FLOW_FUNC_ERR_DRV_ERROR;
    } else {
        // LOG ERROR
        UDF_LOG_ERROR("wait event failed, device_id=%u, threadIndex=%u, groupId=%u, dev_ret=%d.",
            GlobalConfig::Instance().GetDeviceId(), curr_sched_thread_id_, invoke_model_sched_group_id_,
            static_cast<int32_t>(dev_ret));
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowModelImpl::DequeueMbuf(size_t output_idx, Mbuf *&mbuf, int32_t timeout) {
    const auto &output_queue_info = output_queue_infos_[output_idx];
    uint32_t qid = output_queue_info.queue_id;
    int32_t ret = FLOW_FUNC_SUCCESS;
    if (timeout == 0) {
        ret = DequeueMbuf(output_idx, mbuf);
        if (ret == FLOW_FUNC_ERR_QUEUE_EMPTY) {
            UDF_LOG_ERROR("Dequeue failed as queue empty and timeout=0, queue_id=%u", qid);
            ret = FLOW_FUNC_FAILED;
        }
        return ret;
    }
    ret = SubQueueEvent(output_queue_info, QUEUE_ENQUE_EVENT);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("sub queue enqueue event failed, queue_id=%u", qid);
        return ret;
    }

    std::function<void()> unsub_func = [this, &output_queue_info]() {
        (void)UnsubQueueEvent(output_queue_info, QUEUE_ENQUE_EVENT);
        {
            std::unique_lock<std::mutex> guard(event_mt_);
            can_send_event_ = false;
        }
        SwapOutInvokeModelEventGroup();
    };
    // make sure unsub on exit.
    const ScopeGuard unsub_guard(unsub_func);

    ret = DequeueMbuf(output_idx, mbuf);
    if (ret != FLOW_FUNC_ERR_QUEUE_EMPTY) {
        return ret;
    }

    // swap aicpu on device.
    SwapOutGlobalGroup();
    if (!output_queue_info.is_proxy_queue) {
        std::unique_lock<std::mutex> guard(event_mt_);
        can_send_event_ = true;
    }

    bool is_continue = false;
    int64_t wait_time = 0L;
    do {
        if (CheckException() != FLOW_FUNC_SUCCESS) {
            return FLOW_FUNC_FAILED;
        }
        if (output_queue_info.is_proxy_queue) {
            wait_time += kProxyQueueDefaultDequeueTimeout;
            // proxy queue no event
        } else {
            ret = WaitAndHandleEvent(is_continue);
            if ((ret == FLOW_FUNC_ERR_DRV_ERROR) && is_continue) {
                wait_time += kDequeWaitPerLoop;
                continue;
            } else if (ret == FLOW_FUNC_ERR_DRV_ERROR) {
                return FLOW_FUNC_ERR_DRV_ERROR;
            } else if (is_continue) {
                // exception wake up, do CheckException then
                continue;
            }
        }
        ret = DequeueMbuf(output_idx, mbuf);
        if (ret != FLOW_FUNC_ERR_QUEUE_EMPTY) {
            return ret;
        }
    } while (((wait_time < timeout) || (timeout == -1)) && (!GlobalConfig::Instance().GetAbnormalStatus()) &&
             (!GlobalConfig::Instance().GetExitFlag()));
    if (GlobalConfig::Instance().GetAbnormalStatus()) {
        UDF_LOG_ERROR("Stop dequeue result of now system status is abnormal. Wait to redeploy.");
        return FLOW_FUNC_STATUS_REDEPLOYING;
    }
    UDF_LOG_ERROR("wait event timeout, wait_time=%ld, timeout=%d(ms), output_idx=%zu, inputQueueSize=%s, outputQueueSize=%s",
        wait_time, timeout, output_idx,
        ToString(GetQueueSize(input_queue_wrappers_)).c_str(), ToString(GetQueueSize(output_queue_wrappers_)).c_str());
    return FLOW_FUNC_ERR_TIME_OUT_ERROR;
}

int32_t FlowModelImpl::ParseMbuf(size_t output_idx, Mbuf *&mbuf, std::shared_ptr<FlowMsg> &flow_msg) const {
    uint32_t qid = output_queue_infos_[output_idx].queue_id;
    if (mbuf == nullptr) {
        UDF_LOG_ERROR("dequeue mbuf null, queue_id=%u", qid);
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }
    static auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
    std::shared_ptr<Mbuf> mbuf_ptr(mbuf, mbuf_deleter);

    auto mbuf_flow_msg = new(std::nothrow)MbufFlowMsg(mbuf_ptr);
    if (mbuf_flow_msg == nullptr) {
        UDF_LOG_ERROR("new MbufFlowMsg failed, queue_id=%u.", qid);
        return FLOW_FUNC_FAILED;
    }

    flow_msg.reset(mbuf_flow_msg);
    auto init_ret = mbuf_flow_msg->Init();
    if (init_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("data init failed, ret=%d, queue_id=%u.", init_ret, qid);
        return init_ret;
    }
    UDF_LOG_DEBUG("output_idx=%zu, qid=%u, fetch msg=%s", output_idx, qid, mbuf_flow_msg->DebugString().c_str());
    return FLOW_FUNC_SUCCESS;
}


int32_t FlowModelImpl::Fetch(size_t output_idx, std::shared_ptr<FlowMsg> &flow_msg, int32_t timeout) {
    uint32_t qid = output_queue_infos_[output_idx].queue_id;
    UDF_LOG_DEBUG("fetch flow msg, queue_id=%u, timeout=%d(ms).", qid, timeout);
    Mbuf *mbuf = nullptr;
    int32_t ret = DequeueMbuf(output_idx, mbuf, timeout);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("dequeue mbuf failed, queue_id=%u, timeout=%d(ms), ret=%d", qid, timeout, ret);
        return ret;
    }

    ret = ParseMbuf(output_idx, mbuf, flow_msg);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("parse mbuf failed, queue_id=%u, ret=%d", qid, ret);
        return ret;
    }
    return FLOW_FUNC_SUCCESS;
}


void FlowModelImpl::GetMsgs(std::vector<Mbuf *> &data, std::vector<std::shared_ptr<FlowMsg>> &flow_msg) const {
    UDF_LOG_DEBUG("Set output data, output num=%zu.", data.size());
    for (size_t idx = 0UL; idx < data.size(); ++idx) {
        std::shared_ptr<FlowMsg> outputMsg;
        int32_t ret = ParseMbuf(idx, data[idx], outputMsg);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("parse mbuf failed, ret=%d, index=%zu", ret, idx);
            return;
        }
        flow_msg.emplace_back(outputMsg);
    }
    return;
}

int32_t FlowModelImpl::AlignFetch(std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) {
    UDF_LOG_DEBUG("AlignFetch, timeout=%d, output size %u(ms).", timeout, output_queue_infos_.size());
    int32_t ret = FLOW_FUNC_SUCCESS;
    std::vector<Mbuf *> tmp_data;
    while (true) {
        // check whether have expire or over limit data.
        data_aligner_->TryTakeExpiredOrOverLimitData(tmp_data);
        if (!tmp_data.empty()) {
            UDF_LOG_ERROR("has not aligned data.");
            return FLOW_FUNC_FAILED;
        }
        size_t next_idx = data_aligner_->SelectNextIndex();
        Mbuf *tmp_buf = nullptr;
        ret = DequeueMbuf(next_idx, tmp_buf, timeout);
        if (ret != FLOW_FUNC_SUCCESS) {
            return ret;
        }

        uint64_t trans_id = 0;
        uint32_t data_label = 0;
        ret = data_aligner_->GetTransIdAndDataLabel(tmp_buf, trans_id, data_label);
        if (ret != HICAID_SUCCESS || trans_id != current_trans_id_) {
            halMbufFree(tmp_buf);
            UDF_LOG_ERROR("get trans_id error, idx=%zu, mbuf trans_id %llu, current trans_id %llu.",
                          next_idx, trans_id, current_trans_id_);
            continue;
        }

        ret = data_aligner_->PushAndAlignData(next_idx, tmp_buf, tmp_data);
        if (ret != HICAID_SUCCESS) {
            halMbufFree(tmp_buf);
            return FLOW_FUNC_FAILED;
        }
        if (!tmp_data.empty()) {
            GetMsgs(tmp_data, output_msgs);
            break;
        }
    }
    return ret;
}

int32_t FlowModelImpl::SubQueueEvent(const QueueDevInfo &queue_info, QUEUE_EVENT_TYPE queue_event_type) {
    // proxy queue no need sub queue event
    if (queue_info.is_proxy_queue) {
        return FLOW_FUNC_SUCCESS;
    }
    struct QueueSubPara queue_sub_para{};
    queue_sub_para.devId = queue_info.device_id;
    queue_sub_para.qid = queue_info.queue_id;
    queue_sub_para.queType = QUEUE_TYPE_SINGLE;
    queue_sub_para.eventType = queue_event_type;
    queue_sub_para.groupId = invoke_model_sched_group_id_;
    queue_sub_para.flag = QUEUE_SUB_FLAG_SPEC_THREAD;
    queue_sub_para.threadId = GlobalConfig::Instance().GetCurrentSchedThreadIdx();
    drvError_t dev_ret = halQueueSubEvent(&queue_sub_para);
    if ((dev_ret != DRV_ERROR_NONE) && (dev_ret != DRV_ERROR_REPEATED_SUBSCRIBED)) {
        UDF_LOG_ERROR("queue sub event failed, dev_ret=%d, queue_id=%u, queue_event_type=%d.",
            static_cast<int32_t>(dev_ret), queue_info.queue_id, static_cast<int32_t>(queue_event_type));
        return FLOW_FUNC_ERR_EVENT_ERROR;
    }
    curr_sched_thread_id_ = queue_sub_para.threadId;
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowModelImpl::UnsubQueueEvent(const QueueDevInfo &queue_info, QUEUE_EVENT_TYPE queue_event_type) const {
    // proxy queue no need un sub queue event
    if (queue_info.is_proxy_queue) {
        return FLOW_FUNC_SUCCESS;
    }
    struct QueueUnsubPara queue_unsub_para{};
    queue_unsub_para.devId = GlobalConfig::Instance().GetDeviceId();
    queue_unsub_para.qid = queue_info.queue_id;
    queue_unsub_para.eventType = queue_event_type;
    drvError_t dev_ret = halQueueUnsubEvent(&queue_unsub_para);
    if (dev_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("queue unsub event failed, dev_ret=%d, queue_id=%u, queue_event_type=%d.",
            static_cast<int32_t>(dev_ret), queue_info.queue_id, static_cast<int32_t>(queue_event_type));
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowModelImpl::SwapOutGlobalGroup() const {
    // only aicpu need swap out.
    if (!GlobalConfig::Instance().IsRunOnAiCpu()) {
        return;
    }
    const uint32_t thread_idx = GlobalConfig::Instance().GetCurrentSchedThreadIdx();
    const uint32_t current_sched_group_id = GlobalConfig::Instance().GetCurrentSchedGroupId();
    if (current_sched_group_id != invoke_model_sched_group_id_) {
        auto dev_ret = halEschedThreadSwapout(GlobalConfig::Instance().GetDeviceId(), current_sched_group_id, thread_idx);
        if (dev_ret != DRV_ERROR_NONE) {
            UDF_LOG_ERROR("halEschedThreadSwapout failed, groupId=%u, thread_idx=%u, dev_ret=%d",
                current_sched_group_id, thread_idx, static_cast<int32_t>(dev_ret));
        }
    }
}

int32_t FlowModelImpl::CheckException() {
    if (data_aligner_ == nullptr) {
        return FLOW_FUNC_SUCCESS;
    }
    uint64_t trans_id = 0;
    std::lock_guard<std::mutex> guard(exception_mt_);
    while (!exception_wait_report_.empty()) {
        trans_id = *(exception_wait_report_.cbegin());
        if (trans_id == current_trans_id_) {
            UDF_LOG_ERROR("Raise exception by user, trans_id=%lu.", trans_id);
            (void)exception_wait_report_.erase(trans_id);
            return FLOW_FUNC_FAILED;
        } else if (trans_id < current_trans_id_) {
            UDF_LOG_INFO("Raise exception that has been fetched, trans_id=%lu.", trans_id);
            (void)exception_wait_report_.erase(trans_id);
        } else {
            break;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowModelImpl::SwapOutInvokeModelEventGroup() const {
    // only aicpu need swap out.
    if (!GlobalConfig::Instance().IsRunOnAiCpu()) {
        return;
    }
    // no event is handling, no need swap out.
    if (!handle_event_) {
        return;
    }
    handle_event_ = false;
    const uint32_t thread_idx = GlobalConfig::Instance().GetCurrentSchedThreadIdx();
    auto dev_ret = halEschedThreadSwapout(GlobalConfig::Instance().GetDeviceId(), invoke_model_sched_group_id_, thread_idx);
    if (dev_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("halEschedThreadSwapout failed, groupId=%u, thread_idx=%u, dev_ret=%d",
            invoke_model_sched_group_id_, thread_idx, static_cast<int32_t>(dev_ret));
    }
}

void FlowModelImpl::DeleteExceptionTransId(uint64_t trans_id) {
    UDF_LOG_DEBUG("DeleteExceptionTransId, trans_id=%lu.", trans_id);
    if (data_aligner_ != nullptr) {
        data_aligner_->DeleteExceptionTransId(trans_id);
    }
}

void FlowModelImpl::AddExceptionTransId(uint64_t trans_id) {
    UDF_LOG_DEBUG("AddExceptionTransId, trans_id=%lu.", trans_id);
    if (trans_id != current_trans_id_) {
        return;
    }
    {
        std::lock_guard<std::mutex> guard(exception_mt_);
        exception_wait_report_.emplace(trans_id);
    }
    if (data_aligner_ != nullptr) {
        data_aligner_->AddExceptionTransId(trans_id);
    }
    std::unique_lock<std::mutex> guard(event_mt_);
    if (!can_send_event_) {
        return;
    }
    event_summary event_info_summary = {};
    event_info_summary.pid = getpid();
    event_info_summary.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdWakeUp);
    event_info_summary.subevent_id = 0U;
    event_info_summary.msg = nullptr;
    event_info_summary.msg_len = 0U;
    event_info_summary.dst_engine = GlobalConfig::Instance().IsRunOnAiCpu() ? ACPU_LOCAL : CCPU_LOCAL;
    event_info_summary.grp_id = invoke_model_sched_group_id_;
    event_info_summary.tid = curr_sched_thread_id_;

    drvError_t ret = halEschedSubmitEventToThread(GlobalConfig::Instance().GetDeviceId(), &event_info_summary);
    if (ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("Failed to submit event, dev_ret=%d, groupId=%u, threadId=%u.",
            static_cast<int32_t>(ret), invoke_model_sched_group_id_, curr_sched_thread_id_);
        return;
    }
}
}
