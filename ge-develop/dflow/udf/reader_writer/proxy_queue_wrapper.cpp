/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "proxy_queue_wrapper.h"
#include <memory>
#include "common/inner_error_codes.h"
#include "common/udf_log.h"

namespace FlowFunc {
int32_t ProxyQueueWrapper::Enqueue(Mbuf *buf) {
    if (buf == nullptr) {
        HICAID_LOG_ERROR("The input mbuf is nullptr. deviceId=%u, queueId=%u", device_id_, queue_id_);
        return HICAID_ERR_PARAM_INVALID;
    }
    uint32_t head_size = 0U;
    void *head_data = nullptr;
    auto drv_mbuf_ret = halMbufGetPrivInfo(buf, &head_data, &head_size);
    if (drv_mbuf_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halMbufGetPrivInfo failed, drvRet=%d.", drv_mbuf_ret);
        return HICAID_FAILED;
    }
    void *data = nullptr;
    drv_mbuf_ret = halMbufGetBuffAddr(buf, &data);
    if (drv_mbuf_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halMbufGetBuffAddr failed, drvRet=%d.", drv_mbuf_ret);
        return HICAID_FAILED;
    }
    uint64_t data_size = 0U;
    drv_mbuf_ret = halMbufGetDataLen(buf, &data_size);
    if (drv_mbuf_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halMbufGetDataLen failed, ret=%d.", drv_mbuf_ret);
        return HICAID_FAILED;
    }
    constexpr size_t total_len = sizeof(buff_iovec) + sizeof(iovec_info);
    const std::unique_ptr<uint8_t[]> vec_uptr(new(std::nothrow) uint8_t[total_len]);
    if (vec_uptr == nullptr) {
        HICAID_LOG_ERROR("Failed to alloc  buff_iovec, size=%zu.", total_len);
        return HICAID_FAILED;
    }
    auto *vec_ptr = reinterpret_cast<buff_iovec *>(vec_uptr.get());
    vec_ptr->context_base = head_data;
    vec_ptr->context_len = head_size;
    vec_ptr->count = 1U;
    vec_ptr->ptr[0].iovec_base = data;
    vec_ptr->ptr[0].len = data_size;
    const drvError_t drv_queue_ret = halQueueEnQueueBuff(device_id_, queue_id_, vec_ptr, enqueue_timeout_);
    if (drv_queue_ret == DRV_ERROR_QUEUE_FULL) {
        HICAID_LOG_INFO("halQueueEnQueueBuff not complete as queue is full, device_id=%u, queue_id=%u, timeout=%d(ms).",
            device_id_, queue_id_, enqueue_timeout_);
        return HICAID_ERR_QUEUE_FULL;
    }
    if (drv_queue_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR(
            "halQueueEnQueueBuff failed, drvRet=%d, device_id=%u, queue_id=%u, timeout=%d(ms).", drv_queue_ret, device_id_,
            queue_id_, enqueue_timeout_);
        return HICAID_FAILED;
    }
    (void)halMbufFree(buf);
    HICAID_LOG_DEBUG("Enqueue success, deviceId=%u, queueId=%u.", device_id_, queue_id_);
    return HICAID_SUCCESS;
}

int32_t ProxyQueueWrapper::Dequeue(Mbuf *&buf) {
    return DequeueWithTimeout(buf, dequeue_timeout_);
}

int32_t ProxyQueueWrapper::DequeueWithTimeout(Mbuf *&buf, int32_t timeout) {
    HICAID_LOG_DEBUG("ready dequeue for queue[%u], deviceId=%u", queue_id_, device_id_);
    uint64_t data_size = 0U;
    auto drv_queue_ret = halQueuePeek(device_id_, queue_id_, &data_size, timeout);
    if (drv_queue_ret == DRV_ERROR_QUEUE_EMPTY) {
        HICAID_LOG_INFO("halQueuePeek get queue empty. deviceId=%u, queueId=%u, timeout=%d(ms).",
                        device_id_, queue_id_, timeout);
        return HICAID_ERR_QUEUE_EMPTY;
    }
    if (drv_queue_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halQueuePeek failed, drvRet=%d, queueId=%u, timeout=%d(ms)",
                         drv_queue_ret, queue_id_, timeout);
        return HICAID_FAILED;
    }
    Mbuf *mbuf = nullptr;
    auto drv_mbuf_ret = halMbufAlloc(data_size, &mbuf);
    if (drv_mbuf_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halMbufAlloc failed, drvRet=%d, data_size=%lu.", drv_mbuf_ret, data_size);
        return HICAID_FAILED;
    }
    auto mbuf_deleter = [](Mbuf *tmp_buf) { (void)halMbufFree(tmp_buf); };
    std::unique_ptr<Mbuf, decltype(mbuf_deleter)> mbufGuard(mbuf, mbuf_deleter);
    drv_mbuf_ret = halMbufSetDataLen(mbuf, data_size);
    if (drv_mbuf_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halMbufSetDataLen failed, drvRet=%d, data_size=%lu.", drv_mbuf_ret, data_size);
        return HICAID_FAILED;
    }
    uint32_t head_size = 0U;
    void *head_data = nullptr;
    drv_mbuf_ret = halMbufGetPrivInfo(mbuf, &head_data, &head_size);
    if (drv_mbuf_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halMbufGetPrivInfo failed, drvRet=%d.", drv_mbuf_ret);
        return HICAID_FAILED;
    }
    void *data = nullptr;
    drv_mbuf_ret = halMbufGetBuffAddr(mbuf, &data);
    if (drv_mbuf_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halMbufGetBuffAddr failed, drvRet=%d.", drv_mbuf_ret);
        return HICAID_FAILED;
    }
    const size_t total_len = sizeof(buff_iovec) + sizeof(iovec_info);
    const std::unique_ptr<uint8_t[]> vec_uptr(new(std::nothrow) uint8_t[total_len]);
    if (vec_uptr == nullptr) {
        HICAID_LOG_ERROR("Failed to alloc  buff_iovec, size=%zu.", total_len);
        return HICAID_FAILED;
    }
    auto vec_ptr = reinterpret_cast<buff_iovec *>(vec_uptr.get());
    vec_ptr->context_base = head_data;
    vec_ptr->context_len = head_size;
    vec_ptr->count = 1U;
    vec_ptr->ptr[0].iovec_base = data;
    vec_ptr->ptr[0].len = data_size;
    drv_queue_ret = halQueueDeQueueBuff(device_id_, queue_id_, vec_ptr, timeout);
    if (drv_queue_ret != DRV_ERROR_NONE) {
        HICAID_LOG_ERROR("halQueueDeQueueBuff failed, drvRet=%d, deviceId=%u, queueId=%u, timeout=%d(ms).",
            drv_queue_ret, device_id_, queue_id_, timeout);
        return HICAID_FAILED;
    }
    buf = mbuf;
    mbufGuard.release();
    HICAID_LOG_DEBUG("Dequeue success, deviceId=%u, queueId=%u", device_id_, queue_id_);
    return HICAID_SUCCESS;
}

int32_t ProxyQueueWrapper::DiscardMbuffOnce() {
    Mbuf *tmp_buf = nullptr;
    const int32_t ret = DequeueWithTimeout(tmp_buf, 0);
    if (ret == HICAID_SUCCESS) {
        (void)halMbufFree(tmp_buf);
    }
    return ret;
}
}