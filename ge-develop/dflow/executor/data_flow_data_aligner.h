/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_DATAFLOW_DATA_ALIGNER_H_
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_DATAFLOW_DATA_ALIGNER_H_

#include <chrono>
#include <queue>
#include <vector>
#include <map>
#include <mutex>
#include "graph/ge_tensor.h"
#include "ge/ge_data_flow_api.h"
#include "common/ge_common/ge_types.h"
#include "dflow/base/deploy/exchange_service.h"

namespace ge {
struct TensorWithHeader {
  GeTensor tensor;
  ExchangeService::MsgInfo msg_info{};
  int8_t user_data[kMaxUserDataSize]{};
};
class AlignCacheData {
 public:
  explicit AlignCacheData(std::vector<size_t> &cache_nums) : cache_nums_(cache_nums), queue_data_(cache_nums.size()) {}
  ~AlignCacheData();
  AlignCacheData(const AlignCacheData &context) = delete;
  AlignCacheData &operator=(const AlignCacheData &context) & = delete;
  AlignCacheData(AlignCacheData &&context) noexcept = default;
  AlignCacheData &operator=(AlignCacheData &&context) &noexcept = delete;
  bool IsComplete() const;
  bool IsExpire(const std::chrono::steady_clock::time_point &current_time, int32_t timeout) const;
  bool IsEmpty() const;

  Status Push(size_t idx, TensorWithHeader data);

  Status Take(std::vector<GeTensor> &data, DataFlowInfo &info);

 private:
  std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
  std::vector<size_t> &cache_nums_;
  std::vector<std::queue<TensorWithHeader>> queue_data_;
};

class DataFlowDataAligner {
 public:
  using CheckIgnoreTransIdFunc = std::function<bool(uint64_t trans_id)>;
  DataFlowDataAligner(const std::vector<uint32_t> &queue_idxes, InputAlignAttrs input_align_attrs,
                      const CheckIgnoreTransIdFunc &check_ignore_trans_id_func);
  ~DataFlowDataAligner();

  Status PushAndAlignData(uint32_t queue_idx, TensorWithHeader tensor_with_header, std::vector<GeTensor> &output,
                          DataFlowInfo &info, bool &is_aligned);
  uint32_t SelectNextQueueIdx();

  Status TryTakeExpiredOrOverLimitData(std::vector<GeTensor> &data, DataFlowInfo &info, bool &has_output);

  void ClearCacheByTransId(uint64_t trans_id);
  void ClearCache();
  const std::vector<uint32_t> &GetQueueIdxes() const {
    return queue_idxes_;
  }

 private:
  Status TryTakeExpired(std::vector<GeTensor> &data, DataFlowInfo &info, bool &has_output);
  Status TryTakeOverLimit(std::vector<GeTensor> &data, DataFlowInfo &info, bool &has_output);

  std::mutex mt_;
  const std::vector<uint32_t> queue_idxes_;
  // the index of queue_idx in queue_idxes_
  std::map<uint32_t, size_t> queue_idx_order_;
  const InputAlignAttrs align_attrs_;
  CheckIgnoreTransIdFunc check_ignore_trans_id_func_;
  std::vector<size_t> cache_nums_;
  std::map<std::pair<uint64_t, uint32_t>, AlignCacheData> wait_align_data_;
  std::set<uint64_t> exception_trans_id_set_;
};
}  // namespace ge
#endif  // EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_DATAFLOW_DATA_ALIGNER_H_
