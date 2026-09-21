/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_MODEL_IMPL_H
#define FLOW_MODEL_IMPL_H

#include <mutex>
#include <atomic>
#include "ascend_hal.h"
#include "common/common_define.h"
#include "flow_func/flow_model.h"
#include "reader_writer/queue_wrapper.h"
#include "reader_writer/data_aligner.h"

namespace FlowFunc {

class FlowModelImpl : public FlowModel {
public:
    FlowModelImpl(std::vector<QueueDevInfo> input_queue_infos, std::vector<QueueDevInfo> output_queue_infos,
        std::unique_ptr<DataAligner> data_aligner = nullptr);

    ~FlowModelImpl() override = default;

    int32_t Init() override;

    int32_t Run(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
        std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) override;

    void AddExceptionTransId(uint64_t trans_id) override;

    void DeleteExceptionTransId(uint64_t trans_id) override;

private:
    int32_t Feed(size_t input_idx, const std::shared_ptr<FlowMsg> &flow_msg, int32_t timeout);

    int32_t Fetch(size_t output_idx, std::shared_ptr<FlowMsg> &flow_msg, int32_t timeout);

    int32_t SubQueueEvent(const QueueDevInfo &queue_info, QUEUE_EVENT_TYPE queue_event_type);

    int32_t UnsubQueueEvent(const QueueDevInfo &queue_info, QUEUE_EVENT_TYPE queue_event_type) const;

    int32_t DequeueMbuf(size_t output_idx, Mbuf *&mbuf, int32_t timeout);

    int32_t DequeueMbuf(size_t output_idx, Mbuf *&mbuf) const;

    void SwapOutGlobalGroup() const;

    void SwapOutInvokeModelEventGroup() const;

    int32_t WaitAndHandleEvent(bool &is_continue) const;

    int32_t CheckException();

    int32_t ParseMbuf(size_t output_idx, Mbuf *&mbuf, std::shared_ptr<FlowMsg> &flow_msg) const;

    void GetMsgs(std::vector<Mbuf *> &data, std::vector<std::shared_ptr<FlowMsg>> &flow_msg) const;

    int32_t AlignFetch(std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout);

    const std::vector<QueueDevInfo> input_queue_infos_;
    const std::vector<QueueDevInfo> output_queue_infos_;
    std::vector<std::unique_ptr<QueueWrapper>> input_queue_wrappers_;
    std::vector<std::unique_ptr<QueueWrapper>> output_queue_wrappers_;
    std::unique_ptr<DataAligner> data_aligner_;
    thread_local static bool handle_event_;
    std::mutex model_mutex_;
    uint32_t invoke_model_sched_group_id_ = 1;
    std::set<uint64_t> exception_wait_report_;
    std::mutex exception_mt_;
    uint64_t current_trans_id_ = 0;
    uint32_t curr_sched_thread_id_ = 0;
    bool can_send_event_{false};
    std::mutex event_mt_;
};
}
#endif // FLOW_MODEL_IMPL_H
