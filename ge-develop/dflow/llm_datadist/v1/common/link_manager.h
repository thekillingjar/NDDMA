/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_COMMON_LINK_MANAGER_H_
#define AIR_RUNTIME_LLM_ENGINE_COMMON_LINK_MANAGER_H_

#include <vector>
#include "ge/ge_api_types.h"
#include "common/llm_inner_types.h"

namespace llm {
enum class LinkOperator : uint32_t { ClientUnlink = 0U, Link, ServerUnlink };
class LinkManager {
 public:
  virtual ~LinkManager() = default;
  void Initialize(const std::vector<uint32_t> &link_input_indices, const std::vector<uint32_t> &link_output_indices,
                  const size_t device_num, uint64_t cluster_id);
  void SetDifferentUnlinkIndices(const std::vector<uint32_t> &unlink_input_indices,
                                 const std::vector<uint32_t> &unlink_output_indices);
  ge::Status LinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                          const int32_t timeout);

  ge::Status UnlinkClusters(const std::vector<ClusterInfo> &clusters, const LinkOperator operator_type,
                            std::vector<ge::Status> &rets, const int32_t timeout, bool force_flag = false);

 protected:
  virtual ge::Status FeedInputs(const uint32_t graph_id,
                                const std::vector<uint32_t> &indices,
                                const std::vector<ge::Tensor> &inputs,
                                const int32_t timeout,
                                bool is_link);
  virtual ge::Status FetchOutputs(const uint32_t graph_id,
                                  const std::vector<uint32_t> &indices,
                                  std::vector<ge::Tensor> &outputs,
                                  int64_t timeout,
                                  uint64_t transaction_id);
 protected:
  uint64_t link_transaction_id_ = 1U;
  uint64_t unlink_transaction_id_ = 1U;
 private:
  ge::Status PrepareLinkInfoTensors(const std::vector<ClusterInfo> &clusters, const LinkOperator operator_type,
                                    const size_t sliced_num, const int32_t timeout, bool force_flag);
  ge::Status LinkOrUnlinkAsync(int32_t timeout, bool is_link);
  void HandleFetchEmpty(ge::Status status, const std::vector<ge::Tensor> &outputs, std::vector<ge::Status> &rets) const;
  ge::Status GetLinkResult(const std::vector<ClusterInfo> &clusters, int64_t left_timeout,
                           std::vector<ge::Status> &rets, std::vector<ClusterInfo> &need_rollback_clusters,
                           uint64_t transaction_id);
  ge::Status GetUnLinkResult(int64_t left_timeout, std::vector<ge::Status> &rets, uint64_t transaction_id);
  ge::Status ParseOutputRetStatus(const ge::Tensor &output, std::vector<ge::Status> &result) const;
  ge::Status CheckClusterInfo(const std::vector<ClusterInfo> &clusters, const size_t sliced_num) const;

  size_t cluster_num_;
  std::vector<ge::Tensor> link_inputs_;
  std::vector<uint32_t> link_input_indices_;
  std::vector<uint32_t> link_output_indices_;
  std::vector<uint32_t> unlink_input_indices_;
  std::vector<uint32_t> unlink_output_indices_;
  std::timed_mutex link_mtx_;
  size_t device_num_;
  uint64_t cluster_id_;
  bool need_extra_host_timeout_ = true;
};
}  // namespace llm
#endif  // AIR_RUNTIME_LLM_ENGINE_COMMON_LINK_MANAGER_H_
