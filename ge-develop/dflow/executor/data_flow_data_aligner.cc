/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_flow_data_aligner.h"
#include "securec.h"
#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"
#include "data_flow_info_utils.h"

namespace ge {
AlignCacheData::~AlignCacheData() {
  for (size_t idx = 0; idx < queue_data_.size(); ++idx) {
    auto &que = queue_data_[idx];
    while (!que.empty()) {
      que.pop();
      --cache_nums_[idx];
    }
  }
  queue_data_.clear();
}

Status AlignCacheData::Push(size_t idx, TensorWithHeader data) {
  queue_data_[idx].push(std::move(data));
  ++cache_nums_[idx];
  return SUCCESS;
}

bool AlignCacheData::IsComplete() const {
  if (queue_data_.empty()) {
    return false;
  }
  for (const auto &que : queue_data_) {
    if (que.empty()) {
      return false;
    }
  }
  return true;
}

bool AlignCacheData::IsEmpty() const {
  for (const auto &que : queue_data_) {
    if (!que.empty()) {
      return false;
    }
  }
  return true;
}

bool AlignCacheData::IsExpire(const std::chrono::steady_clock::time_point &current_time, int32_t timeout) const {
  std::chrono::duration<double> elapsed = current_time - start_time_;
  int64_t elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  if (elapsed_milliseconds >= timeout) {
    GELOGW("data elapsed %" PRId64 " ms, reach expired time %d ms", elapsed_milliseconds, timeout);
    return true;
  }
  return false;
}

Status AlignCacheData::Take(std::vector<GeTensor> &data, DataFlowInfo &info) {
  bool need_set_data_flow_info = true;
  Status ret = SUCCESS;
  for (size_t idx = 0; idx < queue_data_.size(); ++idx) {
    auto &que = queue_data_[idx];
    if (que.empty()) {
      continue;
    }
    auto &tensor_with_head = que.front();
    const auto &msg_info = tensor_with_head.msg_info;
    if ((msg_info.ret_code == 0) && ((msg_info.data_flag & kNullDataFlagBit) == 0U)) {
      data.emplace_back(std::move(tensor_with_head.tensor));
    }
    if ((ret == SUCCESS) && (msg_info.ret_code != 0)) {
      ret = msg_info.ret_code;
    }
    if (need_set_data_flow_info) {
      DataFlowInfoUtils::InitDataFlowInfoByMsgInfo(info, msg_info);
      (void)info.SetUserData(tensor_with_head.user_data, kMaxUserDataSize);
      need_set_data_flow_info = false;
    }
    que.pop();
    --cache_nums_[idx];
  }
  return ret;
}

DataFlowDataAligner::DataFlowDataAligner(const std::vector<uint32_t> &queue_idxes, InputAlignAttrs input_align_attrs,
                                         const CheckIgnoreTransIdFunc &check_ignore_trans_id_func)
    : queue_idxes_(queue_idxes),
      align_attrs_(input_align_attrs),
      check_ignore_trans_id_func_(check_ignore_trans_id_func),
      cache_nums_(queue_idxes.size()) {
  for (size_t i = 0; i < queue_idxes_.size(); ++i) {
    queue_idx_order_[queue_idxes_[i]] = i;
  }
}

DataFlowDataAligner::~DataFlowDataAligner() {
  std::lock_guard<std::mutex> guard(mt_);
  if (!wait_align_data_.empty()) {
    GELOGW("data aligner has data not aligned, queue index=%s, left data nums=%s", ToString(queue_idxes_).c_str(),
           ToString(cache_nums_).c_str());
  }
}

Status DataFlowDataAligner::PushAndAlignData(uint32_t queue_idx, TensorWithHeader tensor_with_header,
                                             std::vector<GeTensor> &output, DataFlowInfo &info, bool &is_aligned) {
  is_aligned = false;
  std::lock_guard<std::mutex> guard(mt_);
  auto queue_find_ret = queue_idx_order_.find(queue_idx);
  GE_ASSERT_TRUE(queue_find_ret != queue_idx_order_.end(), "queue idx is invalid, queue_idx=%u, valid idx list=%s",
                 queue_idx, ToString(queue_idxes_).c_str());

  size_t idx = queue_find_ret->second;
  uint64_t trans_id = tensor_with_header.msg_info.trans_id;
  uint32_t data_label = tensor_with_header.msg_info.data_label;
  if ((check_ignore_trans_id_func_ != nullptr) && check_ignore_trans_id_func_(trans_id)) {
    GELOGW("trans_id=%" PRIu64 ", data_label=%u is dropped as it is ignored.", trans_id, data_label);
    return SUCCESS;
  }
  std::pair<uint64_t, uint32_t> trans_id_and_data_label(trans_id, data_label);
  auto cache_find_ret = wait_align_data_.find(trans_id_and_data_label);
  if (cache_find_ret == wait_align_data_.end()) {
    AlignCacheData tmp_cache_data(cache_nums_);
    auto emplace_ret = wait_align_data_.emplace(trans_id_and_data_label, std::move(tmp_cache_data));
    GE_ASSERT_TRUE(emplace_ret.second, "add cache data failed, trans_id=%" PRIu64 ", data_label=%u",
      trans_id, data_label);
    cache_find_ret = emplace_ret.first;
  }
  auto &cache_data = cache_find_ret->second;
  // when first push failed, or take empty, need erase it.
  ScopeGuard scope_guard([this, &cache_data, &trans_id_and_data_label]() {
    if (cache_data.IsEmpty()) {
      (void)wait_align_data_.erase(trans_id_and_data_label);
    }
  });
  GE_CHK_STATUS_RET(cache_data.Push(idx, std::move(tensor_with_header)),
                    "save queue_idx[%u] trans_id[%" PRIu64 "] data_label[%u] to the [%zu]th cache queue failed",
                    queue_idx, trans_id, data_label, idx);
  GELOGD("save queue_idx[%u] trans_id[%" PRIu64 "] data_label[%u] to the [%zu]th cache queue success.",
         queue_idx, trans_id, data_label, idx);
  if (cache_data.IsComplete()) {
    GELOGI("trans_id[%" PRIu64 "] data_label[%u] align complete.", trans_id, data_label);
    is_aligned = true;
    return cache_data.Take(output, info);
  }
  return SUCCESS;
}

uint32_t DataFlowDataAligner::SelectNextQueueIdx() {
  std::lock_guard<std::mutex> guard(mt_);
  size_t next_take_idx = 0;
  size_t min_cache_size = std::numeric_limits<size_t>::max();
  for (size_t idx = 0; idx < cache_nums_.size(); ++idx) {
    if (min_cache_size > cache_nums_[idx]) {
      min_cache_size = cache_nums_[idx];
      next_take_idx = idx;
    }
  }
  return queue_idxes_[next_take_idx];
}

void DataFlowDataAligner::ClearCacheByTransId(uint64_t trans_id) {
  std::lock_guard<std::mutex> guard(mt_);
  if (wait_align_data_.empty()) {
    return;
  }
  auto iter = wait_align_data_.begin();
  size_t drop_cache_cnt = 0;
  while (iter != wait_align_data_.end()) {
    if (iter->first.first == trans_id) {
      iter = wait_align_data_.erase(iter);
      ++drop_cache_cnt;
    } else {
      ++iter;
    }
  }
  GELOGI("clear cache by trans id=%" PRIu64 ", drop cache cnt=%zu, queue idxes=%s", trans_id, drop_cache_cnt,
         ToString(queue_idxes_).c_str());
}

Status DataFlowDataAligner::TryTakeExpiredOrOverLimitData(std::vector<GeTensor> &data, DataFlowInfo &info, bool &has_output) {
  has_output = false;
  Status ret = TryTakeExpired(data, info, has_output);
  // when null data, data may be empty.
  if ((ret != SUCCESS) || has_output) {
    return ret;
  }
  return TryTakeOverLimit(data, info, has_output);
}

Status DataFlowDataAligner::TryTakeOverLimit(std::vector<GeTensor> &data, DataFlowInfo &info, bool &has_output) {
  std::lock_guard<std::mutex> guard(mt_);
  if (wait_align_data_.size() <= align_attrs_.align_max_cache_num) {
    return SUCCESS;
  }
  if (!align_attrs_.drop_when_not_align) {
    auto begin = wait_align_data_.begin();
    Status ret = begin->second.Take(data, info);
    if (begin->second.IsEmpty()) {
      GELOGW("cache size=%zu is over limit size %u, take trans id=%" PRIu64 ", data label=%u finish.",
        wait_align_data_.size(), align_attrs_.align_max_cache_num, begin->first.first, begin->first.second);
      (void)wait_align_data_.erase(begin);
    } else {
      GELOGW("cache size=%zu is over limit size %u, take trans id=%" PRIu64 ", "
             "data label=%u not finish, need take next time.",
             wait_align_data_.size(), align_attrs_.align_max_cache_num, begin->first.first, begin->first.second);
    }
    has_output = true;
    return ret;
  }
  while (wait_align_data_.size() > align_attrs_.align_max_cache_num) {
    GELOGW("cache size=%zu is over limit size %u, drop trans id=%" PRIu64 ", data label=%u.", wait_align_data_.size(),
           align_attrs_.align_max_cache_num, wait_align_data_.begin()->first.first,
           wait_align_data_.begin()->first.second);
    (void)wait_align_data_.erase(wait_align_data_.begin());
  }
  return SUCCESS;
}

Status DataFlowDataAligner::TryTakeExpired(std::vector<GeTensor> &data, DataFlowInfo &info, bool &has_output) {
  constexpr const int32_t kAlignNeverTimeout = -1;
  if (align_attrs_.align_timeout == kAlignNeverTimeout) {
    return SUCCESS;
  }
  std::lock_guard<std::mutex> guard(mt_);
  if (wait_align_data_.empty()) {
    return SUCCESS;
  }
  const auto current_time = std::chrono::steady_clock::now();
  for (auto iter = wait_align_data_.begin(); iter != wait_align_data_.end();) {
    if (!iter->second.IsExpire(current_time, align_attrs_.align_timeout)) {
      ++iter;
      continue;
    }
    if (align_attrs_.drop_when_not_align) {
      GELOGW("data trans id=%" PRIu64 ", data label=%u is expire, need drop it.", iter->first.first, iter->first.second);
      iter = wait_align_data_.erase(iter);
      continue;
    }
    Status ret = iter->second.Take(data, info);
    if (iter->second.IsEmpty()) {
      GELOGW("data trans id=%" PRIu64 ", data label=%u is expire, and take finish.",
        iter->first.first, iter->first.second);
      (void)wait_align_data_.erase(iter);
    } else {
      GELOGW("data trans id=%" PRIu64 ", data label=%u is expire, and take not finish.",
        iter->first.first, iter->first.second);
    }
    has_output = true;
    return ret;
  }
  return SUCCESS;
}

void DataFlowDataAligner::ClearCache() {
  std::lock_guard<std::mutex> guard(mt_);
  wait_align_data_.clear();
}
}  // namespace ge
