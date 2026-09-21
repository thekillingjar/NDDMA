/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HICAID_DATA_ALIGNER_H
#define HICAID_DATA_ALIGNER_H

#include <map>
#include <set>
#include <vector>
#include <queue>
#include <chrono>
#include <mutex>
#include "ascend_hal.h"
#include "flow_func/flow_func_defines.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY CachedData {
public:
    explicit CachedData(std::vector<size_t> &queue_cache_num) : queue_cache_num_(queue_cache_num),
        queue_data_(queue_cache_num.size())
    {}

    ~CachedData();

    CachedData(const CachedData &other) = delete;

    CachedData &operator=(const CachedData &other) = delete;

    CachedData(CachedData &&other) noexcept = default;

    CachedData &operator=(CachedData &&other) noexcept = default;

    bool IsExpire(const std::chrono::steady_clock::time_point &current_time, int32_t timeout) const;

    bool IsComplete() const;

    bool IsEmpty() const;

    /**
     * @brief push to cache queue.
     * caller must make sure idx valid
     * @param idx queue index.
     * @param buf cache mbuf.
     */
    void Push(size_t idx, Mbuf *buf);

    void Take(std::vector<Mbuf *> &data);

private:
    std::vector<size_t> &queue_cache_num_;
    std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
    std::vector<std::queue<Mbuf *>> queue_data_;
};

class FLOW_FUNC_VISIBILITY DataAligner {
public:
    DataAligner(size_t queue_num, uint32_t align_max_cache_num, int32_t align_timeout, bool drop_when_not_align) :
        queue_num_(queue_num), align_max_cache_num_(align_max_cache_num), align_timeout_(align_timeout),
        drop_when_not_align_(drop_when_not_align), queue_cache_num_(queue_num) {
    }

    ~DataAligner() = default;

    /**
     * @brief save and try take aligned data.
     * @param idx index
     * @param mbuf save mbuf
     * @param aligned_data aligned data
     * @return HICAID_SUCCESS:save success, other: save failed.
     */
    int32_t PushAndAlignData(size_t idx, Mbuf *mbuf, std::vector<Mbuf *> &aligned_data);

    /**
     * @brief select the next queue to dequeue.
     * @return queue index.
     */
    size_t SelectNextIndex() const;

    /**
     * @brief take data from cache.
     * take priority is expired data > over limit data.
     * @param data: data be taken. empty means no data need to be taken.
     */
    void TryTakeExpiredOrOverLimitData(std::vector<Mbuf *> &data);

    size_t GetWaitAlignDataSize() const {
        return wait_align_data_.size();
    }

    void Reset();

    void AddExceptionTransId(uint64_t trans_id);

    void DeleteExceptionTransId(uint64_t trans_id);

    static int32_t GetTransIdAndDataLabel(Mbuf *mbuf, uint64_t &trans_id, uint32_t &data_label);
private:
    void TryTakeExpired(std::vector<Mbuf *> &data);

    void TryTakeOverLimit(std::vector<Mbuf *> &data);

private:
    const size_t queue_num_ = 0;
    const uint32_t align_max_cache_num_ = 0; // 0 means align not enable
    const int32_t align_timeout_ = -1;  // -1 means never timeout
    const bool drop_when_not_align_ = false;
    std::vector<size_t> queue_cache_num_;
    std::set<uint64_t> exception_trans_id_;
    std::map<std::pair<uint64_t, uint32_t>, CachedData> wait_align_data_;
    std::mutex mutex_;
};
}
#endif // HICAID_DATA_ALIGNER_H
