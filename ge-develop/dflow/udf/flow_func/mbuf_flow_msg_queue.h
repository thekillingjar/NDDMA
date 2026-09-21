/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_MBUF_FLOW_MSG_QUEUE_H
#define FLOW_FUNC_MBUF_FLOW_MSG_QUEUE_H

#include <mutex>
#include "common/common_define.h"
#include "flow_func/flow_msg_queue.h"
#include "reader_writer/queue_wrapper.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MbufFlowMsgQueue : public FlowMsgQueue {
public:
    MbufFlowMsgQueue(const std::shared_ptr<QueueWrapper> &queue_wrapper, const QueueDevInfo &queue_info);

    ~MbufFlowMsgQueue() override = default;

    int32_t Dequeue(std::shared_ptr<FlowMsg> &flow_msg, int32_t timeout = -1) override;

    int32_t Depth() const override;

    int32_t Size() const override;

    void DiscardAllInputData();

    bool StatusOk() const;

private:
    int32_t DequeueMbuf(Mbuf *&mbuf, int32_t timeout);

    int32_t DequeueMbuf(Mbuf *&mbuf) const;

    int32_t SubQueueEnqueEvent() const;

    int32_t UnsubQueueEnqueEvent() const;

    void SwapOutGlobalGroup() const;

    void SwapOutFlowMsgQueueEventGroup() const;

    thread_local static bool handle_event_;

    std::shared_ptr<QueueWrapper> queue_wrapper_;

    QueueDevInfo queue_info_;

    uint32_t flow_msg_queue_sched_group_id_ = 2;

    bool queue_failed_ = false;

    std::mutex dequeue_mutex_;
};
}

#endif // FLOW_FUNC_MBUF_FLOW_MSG_QUEUE_H
