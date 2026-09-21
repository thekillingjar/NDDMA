/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_LLM_COMMON_CLUSTER_MANAGER_H_
#define BUILT_IN_LLM_COMMON_CLUSTER_MANAGER_H_

#include <mutex>
#include "flow_func/meta_params.h"
#include "flow_func/meta_run_context.h"
#include "fsm/state_define.h"
#include "llm_common/llm_common.h"

namespace FlowFunc {
class ClusterManager {
public:
    int32_t Initialize(const std::shared_ptr<MetaParams> &params,
                       int32_t device_id,
                       uint32_t unlink_output_idx,
                       uint32_t update_link_out_idx,
                       bool is_prompt);
    void Finalize();

    int32_t UpdateLink(const std::shared_ptr<MetaRunContext> &run_context,
                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t UnlinkData(const std::shared_ptr<MetaRunContext> &run_context,
                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t SyncKvData(const std::shared_ptr<MetaRunContext> &run_context,
                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) const;

    static FsmStatus PullCacheFromPrompt(PullKvReqInfo *req_info,
                                         bool enable_paged_attention,
                                         uint64_t start_tick,
                                         const CacheEntry *cache_entry = nullptr);

    static FsmStatus TransferCache(TransferKvReqInfo *req_info,
                                   bool enable_paged_attention,
                                   uint64_t start_tick,
                                   const CacheEntry *cache_entry = nullptr);

    static FsmStatus CheckPullReqSize(uint64_t msg_data_size, const PullKvReqInfo *req_info);
    template<typename T>
    static FsmStatus CheckPullAndTransferReqSize(uint64_t msg_data_size, T *req_info, uint32_t additional_block_count = 0U);
    template<typename T>
    static FsmStatus CheckBlockIndices(T *req_info, uint64_t max_block_index, uint32_t start_index = 0U);
private:
    static int32_t InitServerConn(const std::shared_ptr<MetaParams> &params);

    uint32_t unlink_output_idx_ = 0;
    uint32_t update_link_output_idx_ = 0;
    int64_t device_index_ = 0;
    std::vector<uint64_t> mem_addrs_;
    std::shared_ptr<FlowMsg> update_link_output_ = nullptr;
    std::shared_ptr<FlowMsg> unlink_output_ = nullptr;
};

template<typename T>
FsmStatus ClusterManager::CheckPullAndTransferReqSize(uint64_t msg_data_size,
                                                      T *req_info,
                                                      uint32_t additional_block_count) {
    uint64_t block_info_byte_size = msg_data_size - sizeof(T);
    const auto num_blocks = req_info->block_count + req_info->prompt_block_count + additional_block_count;
    if (num_blocks != (block_info_byte_size / sizeof(uint64_t))) {
        UDF_LOG_ERROR("Invalid req size, blockCount:%u, promptBlockCount:%u, real count:%lu.",
                      req_info->block_count, req_info->prompt_block_count, block_info_byte_size / sizeof(uint64_t));
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}

template<typename T>
FsmStatus ClusterManager::CheckBlockIndices(T *req_info, uint64_t max_block_index, uint32_t start_index) {
    for (uint32_t i = 0; i < req_info->block_count; ++i) {
        if (req_info->block_indices[start_index + i] >= max_block_index) {
            UDF_LOG_ERROR("Invalid param, cacheId:%ld, blockIndex (%lu) out of range [0, %zu).",
                          req_info->cache_id, req_info->block_indices[start_index + i], max_block_index);
            return FsmStatus::kFsmParamInvalid;
        }
    }
    return FsmStatus::kFsmSuccess;
}
}  // namespace FlowFunc

#endif  // BUILT_IN_LLM_COMMON_CLUSTER_MANAGER_H_

