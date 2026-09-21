/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "queue_wrapper.h"
#include <thread>
#include "common/inner_error_codes.h"
#include "common/udf_log.h"

namespace FlowFunc {
int32_t QueueWrapper::Enqueue(Mbuf *buf) {
    drvError_t drv_ret = halQueueEnQueue(device_id_, queue_id_, buf);
    if (drv_ret == DRV_ERROR_NONE) {
        return HICAID_SUCCESS;
    } else if (drv_ret == DRV_ERROR_QUEUE_FULL) {
        HICAID_LOG_INFO("halQueueEnQueue not complete as queue is full, queueId=%u", queue_id_);
        return HICAID_ERR_QUEUE_FULL;
    } else {
        HICAID_LOG_ERROR("halQueueEnQueue failed, drv_ret=%d, queueId=%u", static_cast<int32_t>(drv_ret), queue_id_);
        return HICAID_ERR_QUEUE_FAILED;
    }
}

int32_t QueueWrapper::Dequeue(Mbuf *&buf) {
    HICAID_LOG_DEBUG("ready dequeue for queue[%u], deviceId=%u", queue_id_, device_id_);
    void *tmp_buf = nullptr;
    drvError_t ret = halQueueDeQueue(device_id_, queue_id_, &tmp_buf);
    if (ret == DRV_ERROR_NONE) {
        buf = static_cast<Mbuf *>(tmp_buf);
        HICAID_LOG_DEBUG("dequeue for queue[%u] success", queue_id_);
        return HICAID_SUCCESS;
    } else if (ret == DRV_ERROR_QUEUE_EMPTY) {
        HICAID_LOG_INFO("halQueueDeQueue get none data as queue is empty, queueId=%u", queue_id_);
        return HICAID_ERR_QUEUE_EMPTY;
    } else {
        HICAID_LOG_ERROR("halQueueDeQueue failed, drv_ret=%d, queueId=%u", static_cast<int32_t>(ret), queue_id_);
        return HICAID_ERR_QUEUE_FAILED;
    }
}

int32_t QueueWrapper::QueryQueueSize() const {
    QueueInfo queue_info{};
    drvError_t drv_ret = halQueueQueryInfo(device_id_, queue_id_, &queue_info);
    if (drv_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halQueueQueryInfo failed, drv_ret=%d, deviceId=%u, queueId=%u", static_cast<int32_t>(drv_ret),
            device_id_, queue_id_);
        return 0;
    }
    return queue_info.size;
}

int32_t QueueWrapper::QueryQueueDepth() const {
    int32_t depth = 0;
    drvError_t drv_ret =
        halQueueGetStatus(device_id_, queue_id_, QUERY_QUEUE_DEPTH, sizeof(int32_t), static_cast<void *>(&depth));
    if (drv_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halQueueGetStatus failed, drv_ret=%d, deviceId=%u, queueId=%u", static_cast<int32_t>(drv_ret),
            device_id_, queue_id_);
        return 0;
    }
    return depth;
}

int32_t QueueWrapper::DequeueWithTimeout(Mbuf *&buf, int32_t timeout) {
    // max wait 3600
    constexpr uint32_t max_wait_time = 3600 * 1000;
    uint32_t wait_timeout = ((timeout < 0) || (static_cast<uint32_t>(timeout) > max_wait_time)) ?
                           max_wait_time : static_cast<uint32_t>(timeout);
    // retry interval is 10ms
    uint32_t retry_interval = 10;
    uint32_t retry_time = 0;
    QUEUE_QUERY_ITEM query_item = QUERY_QUEUE_STATUS;
    int32_t queue_status = static_cast<int32_t>(QUEUE_EMPTY);
    do {
        drvError_t drv_ret = halQueueGetStatus(device_id_, queue_id_, query_item, sizeof(queue_status), &queue_status);
        if (drv_ret != DRV_ERROR_NONE) {
            HICAID_LOG_ERROR("query queue status failed, drv_ret=%d, queue_id_=%u, query_item=%d, len=%zu",
                static_cast<int32_t>(drv_ret), queue_id_, static_cast<int32_t>(query_item), sizeof(queue_status));
            return HICAID_ERR_QUEUE_FAILED;
        }
        if (queue_status != static_cast<int32_t>(QUEUE_EMPTY)) {
            HICAID_LOG_INFO("wait queue not empty finish, queue_id_=%u, queue_status=%d, retry_time=%u ms.", queue_id_,
                queue_status, retry_time);
            break;
        }
        if (retry_time >= wait_timeout) {
            HICAID_LOG_ERROR("wait queue not empty timeout, queue_id_=%u, retry_time=%u ms", queue_id_, retry_time);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval));
        retry_time += retry_interval;
    } while (true);
    // queue is not empty
    return Dequeue(buf);
}

int32_t QueueWrapper::DiscardMbuffOnce() {
    Mbuf *tmp_buf = nullptr;
    const int32_t ret = Dequeue(tmp_buf);
    if (ret == HICAID_SUCCESS) {
        (void)halMbufFree(tmp_buf);
    }
    return ret;
}

int32_t QueueWrapper::DiscardMbuf() {
    int32_t ret = HICAID_SUCCESS;
    uint32_t count = 0U;
    while (ret == HICAID_SUCCESS) {
        ret = DiscardMbuffOnce();
        if (ret == HICAID_ERR_QUEUE_EMPTY) {
            break;
        } else if (ret == HICAID_SUCCESS) {
            ++count;
            continue;
        } else {
            HICAID_LOG_ERROR("Discard mbuff failed result of dequeue error.");
            return HICAID_ERR_QUEUE_FAILED;
        }
    }
    HICAID_LOG_INFO("Discard [%u] times data for queueId=%u.", count, queue_id_);
    return HICAID_SUCCESS;
}
}
