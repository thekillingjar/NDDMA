/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mbuf_reader.h"
#include <sstream>
#include "proxy_queue_wrapper.h"
#include "common/udf_log.h"

namespace FlowFunc {
namespace {
template<typename T>
std::string VecToStr(const std::vector<T> &v) {
    bool first = true;
    std::stringstream ss;
    ss << "[";
    for (const T &x : v) {
        if (first) {
            first = false;
            ss << x;
        } else {
            ss << ", " << x;
        }
    }
    ss << "]";
    return ss.str();
}
}
MbufReader::MbufReader(const std::string &name, std::vector<std::shared_ptr<QueueWrapper>> &queue_wrappers,
    DATA_CALLBACK_FUNC data_callback, std::unique_ptr<DataAligner> data_aligner)
    : BaseReader(), name_(name), queue_wrappers_(queue_wrappers), read_data_(queue_wrappers.size()),
    data_callback_(std::move(data_callback)), data_aligner_(std::move(data_aligner)) {
}

MbufReader::~MbufReader() {
    ClearData(read_data_);
}

void MbufReader::QueryQueueSize(std::vector<int32_t> &queue_size_list, uint32_t &not_empty_queue_num) const {
    for (size_t idx = 0; idx < queue_wrappers_.size(); ++idx) {
        int32_t queue_size = queue_wrappers_[idx]->QueryQueueSize();
        if (queue_size > 0) {
            ++not_empty_queue_num;
        }
        queue_size_list.emplace_back(queue_size);
    }
}

void MbufReader::DumpReaderStatus() const {
    if (queue_wrappers_.empty()) {
        return;
    }
    std::string align_status_info;
    size_t next_dequeue_idx = 0;
    bool some_data_cached = false;
    if (data_aligner_ != nullptr) {
        next_dequeue_idx = data_aligner_->SelectNextIndex();
        size_t wait_align_data_size = data_aligner_->GetWaitAlignDataSize();
        some_data_cached = wait_align_data_size > 0;
        align_status_info = ", cached not aligned data size=" + std::to_string(wait_align_data_size);
    } else {
        for (size_t idx = 0; idx < read_data_.size(); ++idx) {
            // read data is in order.
            if (read_data_[idx] == nullptr) {
                next_dequeue_idx = idx;
                break;
            } else {
                some_data_cached = true;
            }
        }
    }

    std::string queue_status_info =
        ", wait to dequeue [" + std::to_string(next_dequeue_idx) + "]th queue[" +
        queue_wrappers_[next_dequeue_idx]->GetQueueInfo() + "]";
    if (queue_failed_) {
        queue_status_info += ", but dequeue failed";
    }
    if (queue_empty_) {
        queue_status_info += ", but queue is empty";
        uint32_t not_empty_queue_num = 0;
        std::vector<int32_t> queue_size_list;
        queue_size_list.reserve(queue_wrappers_.size());
        QueryQueueSize(queue_size_list, not_empty_queue_num);
        if ((not_empty_queue_num > 0) || some_data_cached) {
            queue_status_info += ", may be miss data, current queue size=" + VecToStr(queue_size_list);
        }
    }

    HICAID_RUN_LOG_INFO("%s: input queue num=%zu%s%s.", name_.c_str(), queue_wrappers_.size(), queue_status_info.c_str(),
        align_status_info.c_str());
}

void MbufReader::ReportData(std::vector<Mbuf *> &data) const {
    HICAID_LOG_DEBUG("begin callback");
    data_callback_(data);
    HICAID_LOG_DEBUG("end callback");

    // clear data.
    ClearData(data);
}

void MbufReader::ReadWithDataAlign() {
    queue_empty_ = false;
    std::vector<Mbuf *> tmp_data;
    size_t dequeue_times = 0;
    size_t max_dequeue_times = queue_wrappers_.size();
    do {
        // check whether have expire or over limit data.
        data_aligner_->TryTakeExpiredOrOverLimitData(tmp_data);
        if (!tmp_data.empty()) {
            HICAID_LOG_INFO("will report not aligned data.");
            ReportData(tmp_data);
            return;
        }

        size_t next_idx = data_aligner_->SelectNextIndex();
        Mbuf *tmp_buf = nullptr;
        int32_t ret = queue_wrappers_[next_idx]->Dequeue(tmp_buf);
        if (ret == HICAID_ERR_QUEUE_EMPTY) {
            queue_empty_ = true;
            need_retry_ = queue_wrappers_[next_idx]->NeedRetry();
            return;
        }
        if (ret != HICAID_SUCCESS) {
            // queueWrapper has logged
            queue_failed_ = true;
            need_retry_ = false;
            return;
        }

        int32_t cache_ret = data_aligner_->PushAndAlignData(next_idx, tmp_buf, tmp_data);
        if (cache_ret != HICAID_SUCCESS) {
            halMbufFree(tmp_buf);
            queue_failed_ = true;
            need_retry_ = false;
            return;
        }

        if (!tmp_data.empty()) {
            ReportData(tmp_data);
            return;
        }
    } while (++dequeue_times < max_dequeue_times);
    HICAID_LOG_INFO("dequeue %zu times, need dequeue next time.", dequeue_times);
}

void MbufReader::ReadMessage() {
    need_retry_ = true;
    if (queue_wrappers_.empty()) {
        return;
    }
    if (data_aligner_ != nullptr) {
        ReadWithDataAlign();
        return;
    }

    read_data_.resize(queue_wrappers_.size());
    bool read_finish = false;
    ReadAllData(read_finish);
    if (read_finish) {
        ReportData(read_data_);
        return;
    }
}

bool MbufReader::StatusOk() const {
    return !queue_failed_;
}

void MbufReader::ClearData(std::vector<Mbuf *> &mbuf_vec) noexcept {
    for (auto &data : mbuf_vec) {
        if (data != nullptr) {
            (void)halMbufFree(data);
            data = nullptr;
        }
    }
}

void MbufReader::ReadAllData(bool &read_finish) {
    read_finish = true;
    queue_empty_ = false;
    queue_failed_ = false;
    for (size_t i = 0UL; i < read_data_.size(); ++i) {
        if (read_data_[i] != nullptr) {
            continue;
        }
        Mbuf *tmp_buf = nullptr;
        int32_t ret = queue_wrappers_[i]->Dequeue(tmp_buf);
        if (ret == HICAID_SUCCESS) {
            read_data_[i] = tmp_buf;
        } else if (ret == HICAID_ERR_QUEUE_EMPTY) {
            read_finish = false;
            queue_empty_ = true;
            need_retry_ = queue_wrappers_[i]->NeedRetry();
            break;
        } else {
            // queueWrapper has logged
            read_finish = false;
            queue_failed_ = true;
            need_retry_ = false;
            break;
        }
    }
    HICAID_LOG_DEBUG("ReadAllData end, read_finish=%d", static_cast<int32_t>(read_finish));
}

void MbufReader::DiscardAllInputData() {
    queue_failed_ = false;
    for (size_t i = 0UL; i < queue_wrappers_.size(); ++i) {
        if (queue_wrappers_[i]->DiscardMbuf() != HICAID_SUCCESS) {
            queue_failed_ = true;
            HICAID_LOG_ERROR("Discard all input data failed of index = %zu.", i);
            return;
        }
    }
}

void MbufReader::Reset() {
    ClearData(read_data_);
    if (data_aligner_ != nullptr) {
        data_aligner_->Reset();
    }
}
void MbufReader::DeleteExceptionTransId(uint64_t trans_id) {
    if (data_aligner_ != nullptr) {
        data_aligner_->DeleteExceptionTransId(trans_id);
    }
}

void MbufReader::AddExceptionTransId(uint64_t trans_id) {
    if (data_aligner_ != nullptr) {
        data_aligner_->AddExceptionTransId(trans_id);
    }
}
}