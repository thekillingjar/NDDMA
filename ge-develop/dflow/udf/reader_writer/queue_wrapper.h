/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HICAID_QUEUE_WRAPPER_H
#define HICAID_QUEUE_WRAPPER_H

#include <string>
#include "ascend_hal.h"
#include "flow_func/flow_func_defines.h"

namespace FlowFunc {

class FLOW_FUNC_VISIBILITY QueueWrapper {
public:
    QueueWrapper(uint32_t device_id, uint32_t queue_id) : device_id_(device_id), queue_id_(queue_id) {}

    virtual ~QueueWrapper() = default;

    virtual int32_t Enqueue(Mbuf *buf);

    virtual int32_t Dequeue(Mbuf *&buf);

    virtual int32_t DequeueWithTimeout(Mbuf *&buf, int32_t timeout);

    /**
     * @brief need retry again
     * @return false no need retry.
     */
    virtual bool NeedRetry() const {
        return false;
    }

    /**
     * @brief query left data num in queue.
     * @return queue size.
     */
    int32_t QueryQueueSize() const;

    int32_t QueryQueueDepth() const;

    int32_t DiscardMbuf();

    std::string GetQueueInfo() const {
        return "queue:" + std::to_string(queue_id_) + " device_id:" + std::to_string(device_id_);
    }

protected:
    virtual int32_t DiscardMbuffOnce();

    uint32_t device_id_;
    uint32_t queue_id_;
};
}
#endif // HICAID_QUEUE_WRAPPER_H
