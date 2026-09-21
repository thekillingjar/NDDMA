/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_aligner.h"
#include <set>
#include "common/inner_error_codes.h"
#include "common/udf_log.h"

namespace FlowFunc {
namespace {
// -1 means never timeout
constexpr int64_t kDataNeverExpireTime = -1;

#pragma pack(push, 1)

struct MbufHeadMsg {
    uint64_t trans_id;
    uint16_t version;
    uint16_t msg_type;
    int32_t ret_code;
    uint64_t start_time;
    uint64_t end_time;
    uint32_t flags;
    uint8_t data_flag;
    uint8_t rsv0[3];
    int32_t worker_id;       // ES使用，占位
    uint32_t step_id;
    uint8_t rsv[8];
    uint32_t data_label;     // use for data align
    uint32_t route_label;    // use for data route
};
#pragma pack(pop)
}

CachedData::~CachedData() {
    for (size_t idx = 0; idx < queue_data_.size(); ++idx) {
        auto &queue_mbuf = queue_data_[idx];
        while (!queue_mbuf.empty()) {
            auto mbuf = queue_mbuf.front();
            if (mbuf != nullptr) {
                halMbufFree(mbuf);
            }
            queue_mbuf.pop();
            --queue_cache_num_[idx];
        }
    }
    queue_data_.clear();
}

bool CachedData::IsExpire(const std::chrono::steady_clock::time_point &current_time, int32_t timeout) const {
    std::chrono::duration<double> elapsed = current_time - start_time_;
    int64_t elapsed_second = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    if (elapsed_second >= timeout) {
        HICAID_LOG_WARN("data elapsed %ld ms, reach expire time %d ms", elapsed_second, timeout);
        return true;
    }
    return false;
}

bool CachedData::IsComplete() const {
    if (queue_data_.empty()) {
        return false;
    }
    for (const auto &queue_buf : queue_data_) {
        if (queue_buf.empty()) {
            return false;
        }
    }
    return true;
}

bool CachedData::IsEmpty() const {
    for (const auto &queue_buf : queue_data_) {
        if (!queue_buf.empty()) {
            return false;
        }
    }
    return true;
}

void CachedData::Push(size_t idx, Mbuf *buf) {
    queue_data_[idx].push(buf);
    ++queue_cache_num_[idx];
}

void CachedData::Take(std::vector<Mbuf *> &data) {
    data.resize(queue_data_.size());
    for (size_t i = 0; i < queue_data_.size(); ++i) {
        if (queue_data_[i].empty()) {
            continue;
        }
        data[i] = queue_data_[i].front();
        queue_data_[i].pop();

        --queue_cache_num_[i];
    }
}

int32_t DataAligner::GetTransIdAndDataLabel(Mbuf *mbuf, uint64_t &trans_id, uint32_t &data_label) {
    if (mbuf == nullptr) {
        HICAID_LOG_ERROR("get trans_id failed as mbuf is null.");
        return HICAID_ERR_PARAM_INVALID;
    }
    void *head_buf = nullptr;
    uint32_t head_buf_len = 0U;
    int32_t drv_ret = halMbufGetPrivInfo(mbuf, &head_buf, &head_buf_len);
    if (drv_ret != DRV_ERROR_NONE || head_buf == nullptr) {
        HICAID_LOG_ERROR("Failed to get mbuf priv info, ret[%d].", drv_ret);
        return HICAID_FAILED;
    }
    if (head_buf_len < sizeof(MbufHeadMsg)) {
        HICAID_LOG_ERROR("mbuf priv info len=%u can't be less than to sizeof(MbufHeadMsg)=%zu.",
            head_buf_len, sizeof(MbufHeadMsg));
        return HICAID_FAILED;
    }

    const auto *head_msg = reinterpret_cast<MbufHeadMsg *>(static_cast<uint8_t *>(head_buf) +
                                                          (head_buf_len - sizeof(MbufHeadMsg)));
    trans_id = head_msg->trans_id;
    data_label = head_msg->data_label;

    return HICAID_SUCCESS;
}

size_t DataAligner::SelectNextIndex() const {
    size_t next_take_idx = 0;
    size_t min_cache_size = std::numeric_limits<size_t>::max();
    for (size_t idx = 0; idx < queue_cache_num_.size(); ++idx) {
        if (min_cache_size > queue_cache_num_[idx]) {
            min_cache_size = queue_cache_num_[idx];
            next_take_idx = idx;
        }
    }
    return next_take_idx;
}

int32_t DataAligner::PushAndAlignData(size_t idx, Mbuf *mbuf, std::vector<Mbuf *> &aligned_data) {
    if (idx >= queue_num_) {
        HICAID_LOG_ERROR("save failed as idx=%zu must be less than %zu.", idx, queue_num_);
        return HICAID_ERR_PARAM_INVALID;
    }
    uint64_t trans_id = 0;
    uint32_t data_label = 0;
    int32_t get_ret = GetTransIdAndDataLabel(mbuf, trans_id, data_label);
    if (get_ret != HICAID_SUCCESS) {
        HICAID_LOG_ERROR("get trans_id and data label failed, idx=%zu.", idx);
        return get_ret;
    }

     std::unique_lock<std::mutex> guard(mutex_);
    if (exception_trans_id_.find(trans_id) != exception_trans_id_.cend()) {
        HICAID_LOG_WARN("trans_id=%lu is in exception list, drop it, data_label=%u.", trans_id, data_label);
        halMbufFree(mbuf);
        return HICAID_SUCCESS;
    }

    std::pair<uint64_t, uint32_t> trans_id_and_data_label(trans_id, data_label);
    auto find_ret = wait_align_data_.find(trans_id_and_data_label);
    if (find_ret == wait_align_data_.end()) {
        auto insertRet = wait_align_data_.emplace(trans_id_and_data_label, CachedData(queue_cache_num_));
        if (!insertRet.second) {
            HICAID_LOG_ERROR("emplace trans_id=%lu data_label[%u] failed.", trans_id_and_data_label.first,
                trans_id_and_data_label.second);
            return HICAID_FAILED;
        }
        find_ret = insertRet.first;
    }
    auto &cache_data = find_ret->second;
    cache_data.Push(idx, mbuf);

    HICAID_LOG_DEBUG("save trans_id[%lu], data_label[%u] to the [%zu]th queue success.", trans_id, data_label, idx);
    if (cache_data.IsComplete()) {
        HICAID_LOG_INFO("trans_id[%lu], data_label[%u] is complete.", trans_id, data_label);
        cache_data.Take(aligned_data);
        if (cache_data.IsEmpty()) {
            wait_align_data_.erase({trans_id, data_label});
        }
    }
    return HICAID_SUCCESS;
}

void DataAligner::TryTakeExpiredOrOverLimitData(std::vector<Mbuf *> &data) {
    std::unique_lock<std::mutex> guard(mutex_);
    // second take expired data
    TryTakeExpired(data);
    if (!data.empty()) {
        return;
    }
    // last take over limit data
    TryTakeOverLimit(data);
}

void DataAligner::TryTakeExpired(std::vector<Mbuf *> &data) {
    if (wait_align_data_.empty()) {
        return;
    }
    if (align_timeout_ == kDataNeverExpireTime) {
        return;
    }
    const auto current_time = std::chrono::steady_clock::now();

    for (auto iter = wait_align_data_.begin(); iter != wait_align_data_.end();) {
        if (iter->second.IsExpire(current_time, align_timeout_)) {
            if (drop_when_not_align_) {
                HICAID_LOG_WARN("data trans_id=%lu, data_label=%u is expire, need drop it.", iter->first.first,
                    iter->first.second);
                iter = wait_align_data_.erase(iter);
                continue;
            }

            iter->second.Take(data);
            if (iter->second.IsEmpty()) {
                HICAID_LOG_WARN("data trans_id=%lu, data_label=%u is expire, and handle finish.", iter->first.first,
                    iter->first.second);
                (void)wait_align_data_.erase(iter);
            } else {
                HICAID_LOG_WARN(
                    "data trans_id=%lu, data_label=%u is expire, but has data left after take, need take next time.",
                    iter->first.first, iter->first.second);
            }
            break;
        } else {
            ++iter;
        }
    }
}

void DataAligner::TryTakeOverLimit(std::vector<Mbuf *> &data) {
    if (wait_align_data_.size() <= align_max_cache_num_) {
        return;
    }

    if (!drop_when_not_align_) {
        auto begin = wait_align_data_.begin();
        begin->second.Take(data);
        if (begin->second.IsEmpty()) {
            HICAID_LOG_WARN("cache size=%zu is over limit size %u, take trans_id=%lu, data_label=%u finish.",
                wait_align_data_.size(), align_max_cache_num_, begin->first.first, begin->first.second);
            (void)wait_align_data_.erase(begin);
        } else {
            HICAID_LOG_WARN("cache size=%zu is over limit size %u, take trans_id=%lu, data_label=%u and has data left, "
                            "need take next time.",
                wait_align_data_.size(), align_max_cache_num_, begin->first.first, begin->first.second);
        }
        return;
    }

    while (wait_align_data_.size() > align_max_cache_num_) {
        HICAID_LOG_WARN("cache size=%zu is over limit size %u, drop trans_id=%lu, data_label=%u.",
            wait_align_data_.size(), align_max_cache_num_, wait_align_data_.begin()->first.first,
            wait_align_data_.begin()->first.second);
        wait_align_data_.erase(wait_align_data_.begin());
    }
}

void DataAligner::AddExceptionTransId(uint64_t trans_id) {
    std::unique_lock<std::mutex> guard(mutex_);
    (void)exception_trans_id_.emplace(trans_id);
    auto iter = wait_align_data_.begin();
    size_t drop_cache_cnt = 0;
    while (iter != wait_align_data_.cend()) {
        if (iter->first.first == trans_id) {
            iter = wait_align_data_.erase(iter);
            ++drop_cache_cnt;
        } else {
            ++iter;
        }
    }
    HICAID_LOG_INFO("add exception trans_id end, trans_id=%lu, total exception trans_id count=%zu, drop_cache_cnt=%zu.",
        trans_id, exception_trans_id_.size(), drop_cache_cnt);
}

void DataAligner::DeleteExceptionTransId(uint64_t trans_id) {
    std::unique_lock<std::mutex> guard(mutex_);
    (void)exception_trans_id_.erase(trans_id);
}

void DataAligner::Reset() {
    std::unique_lock<std::mutex> guard(mutex_);
    wait_align_data_.clear();
    exception_trans_id_.clear();
    queue_cache_num_.assign(queue_cache_num_.size(), 0);
}
}