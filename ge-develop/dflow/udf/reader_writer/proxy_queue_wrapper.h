/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HICAID_PROXY_QUEUE_WRAPPER_H
#define HICAID_PROXY_QUEUE_WRAPPER_H

#include "queue_wrapper.h"

namespace FlowFunc {
// default timeout is 1s.
constexpr int32_t kProxyQueueDefaultDequeueTimeout = 500;   // 500ms
constexpr int32_t kProxyQueueDefaultEnqueueTimeout = 2000;  // 2s

class FLOW_FUNC_VISIBILITY ProxyQueueWrapper : public QueueWrapper {
public:
    explicit ProxyQueueWrapper(uint32_t device_id, uint32_t queue_id,
                               int32_t dequeue_timeout = kProxyQueueDefaultDequeueTimeout,
                               int32_t enqueue_timeout = kProxyQueueDefaultEnqueueTimeout)
        : QueueWrapper(device_id, queue_id), dequeue_timeout_(dequeue_timeout), enqueue_timeout_(enqueue_timeout) {}

    ~ProxyQueueWrapper() override = default;

    int32_t Dequeue(Mbuf *&buf) override;

    int32_t Enqueue(Mbuf *buf) override;

    int32_t DequeueWithTimeout(Mbuf *&buf, int32_t timeout) override;

    bool NeedRetry() const override
    {
        // proxy queue need retry
        return true;
    }

protected:
    int32_t DiscardMbuffOnce() override;

private:
    int32_t dequeue_timeout_;
    int32_t enqueue_timeout_;
};
}
#endif // HICAID_PROXY_QUEUE_WRAPPER_H
