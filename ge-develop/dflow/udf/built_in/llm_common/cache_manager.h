/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_LLM_COMMON_CACHE_MANAGER_H_
#define BUILT_IN_LLM_COMMON_CACHE_MANAGER_H_

#include <mutex>
#include "flow_func/meta_multi_func.h"
#include "flow_func/meta_params.h"
#include "flow_func/meta_run_context.h"
#include "flow_func/tensor_data_type.h"
#include "llm_common/llm_common.h"
#include "llm_common/cluster_manager.h"

namespace FlowFunc {
using CacheIndex = std::pair<uint64_t, uint64_t>;  // reqId or prefixId, modelId

class CacheManager {
public:
    static CacheManager &GetInstance();
    FsmStatus AllocateCache(const std::shared_ptr<MetaRunContext> &run_context,
                            const AllocateCacheReqInfo &req_info,
                            uint64_t *data_addresses);
    FsmStatus DeallocateCache(int64_t cache_id);
    FsmStatus RemoveCacheIndex(const CacheIndex &cache_index,
                               bool is_prefix,
                               uint32_t num_tensor_indices = 0,
                               const TransferInfo *tensor_indices = nullptr);
    FsmStatus CopyBlockCache(const CopyCacheReqInfo &req_info) const;
    FsmStatus CopyCache(const CopyCacheReqInfo &req_info) const;
    FsmStatus PullCache(PullKvReqInfo *req_info, ClusterManager &cluster_manager, uint64_t start_tick) const;
    FsmStatus TransferCache(TransferKvReqInfo *req_info, ClusterManager &cluster_manager, uint64_t start_tick) const;
    bool GetCacheEntry(const CacheIndex &cache_index, CacheEntry &cache_entry, bool is_prefix) const;
    bool GetCacheEntry(int64_t cache_id, CacheEntry &cache_entry) const;
    bool GetCacheIndex(const std::pair<int64_t, uint32_t> &cache_id_and_batch_index, CacheIndex &cache_index) const;
    void Enable();
    bool IsEnabled() const;
    FsmStatus CheckAddr(uint64_t addr, uint64_t check_size);

private:
    void RemoveAddrToSize(const CacheEntry &cache_entry);
    static FsmStatus CheckCopyCacheParam(const CacheEntry &src_cache_entry, const CacheEntry &dst_cache_entry,
                                         const CopyCacheReqInfo &req_info, uint64_t &copy_size);
    void AddCacheIndices(CacheEntry &cache_entry, const AllocateCacheReqInfo &req_info,
                         const std::vector<CacheIndex> &cache_indices);
    void AddCacheIndicesForBlocks(const AllocateCacheReqInfo &req_info,
                                  const std::vector<CacheIndex> &cache_indices);
    FsmStatus CheckAndCreateCacheIndices(const AllocateCacheReqInfo &req_info,
                                         std::vector<CacheIndex> &cache_indices) const;
    const CacheEntry *DoGetCacheEntry(int64_t cache_id) const;
    static FsmStatus CheckTensorIndices(const PullKvReqInfo &req_info,
                                        const CacheEntry &cache_entry,
                                        bool enable_page_attention);
    static FsmStatus CheckReqInfo(const CacheEntry &cache_entry, PullKvReqInfo *req_info, bool &enable_page_attention);

    mutable std::mutex mu_;
    bool enabled_ = false;
    std::map<int64_t, CacheEntry> cache_id_to_entry_;
    std::map<CacheIndex, int64_t> cache_index_to_id_;
    std::map<CacheIndex, int64_t> prefix_index_to_id_;
    std::map<std::pair<int64_t, uint32_t>, CacheIndex> cache_id_and_batch_id_to_cache_index_;
    std::map<uint64_t, uint64_t> addr_to_size_;
};
}  // namespace FlowFunc

#endif  // BUILT_IN_LLM_COMMON_CACHE_MANAGER_H_