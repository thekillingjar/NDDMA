/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_LLM_COMMON_KV_CACHE_MANAGER_H
#define BUILT_IN_LLM_COMMON_KV_CACHE_MANAGER_H

#include <mutex>
#include <unordered_map>
#include "fsm/state_define.h"
#include "flow_func/flow_msg.h"
#include "flow_func/meta_run_context.h"
#include "llm_common/llm_common.h"

namespace FlowFunc {
constexpr uint64_t kHashShiftLeft = 32UL;

class KvCacheManager {
public:
    struct ReqKey {
        uint64_t req_id;
        uint64_t model_id;
        bool operator==(const ReqKey &req_key) const {
            return ((req_id == req_key.req_id) && (model_id == req_key.model_id));
        }
    };
    struct ReqKeyHash {
        std::size_t operator()(const ReqKey &req_key) const {
            std::hash<uint64_t> hash_fn;
            return hash_fn(req_key.req_id) ^ hash_fn(req_key.model_id << kHashShiftLeft);
        }
    };
    struct PrefixReqKey {
        uint64_t prefix_id;
        uint64_t model_id;
        bool operator==(const PrefixReqKey &req_key) const {
            return ((prefix_id == req_key.prefix_id) && (model_id == req_key.model_id));
        }
    };
    struct PrefixReqKeyHash {
        std::size_t operator()(const PrefixReqKey &req_key) const {
            std::hash<uint64_t> hash_fn;
            return hash_fn(req_key.prefix_id) ^ hash_fn(req_key.model_id << kHashShiftLeft);
        }
    };

    static KvCacheManager &GetInstance();
    virtual ~KvCacheManager();
    std::shared_ptr<FlowMsg> QueryDecoderKvCache(const ReqKey &req_key);
    FsmStatus SaveDecoderKvCache(const ReqKey &req_key, const std::shared_ptr<FlowMsg> &kv_cache);
    FsmStatus ReleaseDecoderKvCache(const ReqKey &req_key);
    FsmStatus ReleaseDecoderKvCaches();
    void SetDecoderKvCache(const std::vector<std::shared_ptr<FlowMsg>> &kv_caches);
    std::shared_ptr<FlowMsg> GetDecoderKvCache(uint64_t model_id) const;
    void ClearDecoderKvCache();
    uint64_t GetPromptKvCacheCount();
    uint64_t GetDecoderKvCacheCount(const ReqKey &req_key);
    std::vector<KvTensor> QueryPromptKvCache(const ReqKey &req_key);
    std::vector<KvTensor> QueryPromptPrefixKvCache(const PrefixReqKey &prefix_req_key);
    FsmStatus ShrinkKvCacheForPrefix(const PrefixReqKey &prefix_req_key,
                                     const std::shared_ptr<MetaRunContext> &run_context);
    FsmStatus ReleaseKvCacheForPrompt(const ReqKey &req_key);
    FsmStatus ReleaseKvCacheForPrompt(uint64_t req_id);
    FsmStatus ReleaseKvCacheForPrefix(const PrefixReqKey &prefix_req_key);
    uint64_t AddKvCache(size_t batch_num, std::vector<std::shared_ptr<FlowMsg>> kv_tensors);
    void SaveKvCacheForPrompt(const ReqKey &req_key, uint64_t cache_id, std::pair<int64_t, int64_t> offset_and_size);
    void SaveKvCacheForPromptPrefix(const PrefixReqKey &prefix_req_key,
                                    uint64_t cache_id,
                                    std::pair<int64_t, int64_t> offset_and_size);
    void InitBlockTensors(uint64_t model_id);
    void SaveKvBlocksTensor(uint64_t model_id, std::shared_ptr<FlowMsg> &kv_blocks_tensor);
    std::vector<std::shared_ptr<FlowMsg>> &QueryKvBlocksTensors(uint64_t model_id);
    void ClearKvBlocksTensors();
    void ClearPromptKvCache();
    KvCacheManager(const KvCacheManager &) = delete;
    KvCacheManager(const KvCacheManager &&) = delete;
    KvCacheManager &operator=(const KvCacheManager &) = delete;
    KvCacheManager &operator=(const KvCacheManager &&) = delete;

private:
    KvCacheManager() = default;
    std::vector<KvTensor> QueryKvTensor(uint64_t cache_id, uint64_t tensor_id);
    FsmStatus ReleaseKvCacheForPromptByKey(const ReqKey &req_key);

private:
    struct CacheEntry {
        size_t batch_num;
        std::vector<std::shared_ptr<FlowMsg>> kv_tensors;
        std::unordered_map<uint64_t, std::pair<int64_t, int64_t>> id_to_offset_and_size;
    };

    // only one reserved kv cache for decoder
    std::vector<std::shared_ptr<FlowMsg>> decoder_kv_caches_;
    // mutex for decoder and prompt kv map
    std::mutex mutex_;
    // synced kv cache for decoder
    std::unordered_map<ReqKey, std::shared_ptr<FlowMsg>, ReqKeyHash> decoder_req_to_kv_cache_;
    // kv physical blocks tensor list, count = layerNum * 2
    std::vector<std::vector<std::shared_ptr<FlowMsg>>> kv_blocks_tensors_;

    uint64_t cache_id_gen_{0};
    std::unordered_map<uint64_t, CacheEntry> cache_id_to_cache_entry_;
    std::unordered_map<ReqKey, uint64_t, ReqKeyHash> req_id_to_cache_id_;
    std::unordered_map<PrefixReqKey, uint64_t, PrefixReqKeyHash> prefix_id_to_cache_id_;
};
}  // namespace FlowFunc

#endif  // BUILT_IN_LLM_COMMON_KV_CACHE_MANAGER_H