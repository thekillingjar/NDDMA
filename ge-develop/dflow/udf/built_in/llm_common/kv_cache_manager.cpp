/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_common/kv_cache_manager.h"
#include <set>
#include <memory>
#include <vector>
#include "ascend_hal.h"

namespace FlowFunc {
KvCacheManager &KvCacheManager::GetInstance() {
    static KvCacheManager manager;
    return manager;
}

KvCacheManager::~KvCacheManager() {
    ClearDecoderKvCache();
    ClearPromptKvCache();
    decoder_req_to_kv_cache_.clear();
    kv_blocks_tensors_.clear();
}

std::shared_ptr<FlowMsg> KvCacheManager::QueryDecoderKvCache(const ReqKey &req_key) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto iter = decoder_req_to_kv_cache_.find(req_key);
    if (iter == decoder_req_to_kv_cache_.end()) {
        uint64_t cached_req_id = decoder_req_to_kv_cache_.empty() ? UINT64_MAX : decoder_req_to_kv_cache_.begin()->first.req_id;
        UDF_LOG_ERROR("Can not find kv cache for req_id:%lu, current cached req_id:%lu.", req_key.req_id, cached_req_id);
        return nullptr;
    }
    return iter->second;
}

FsmStatus KvCacheManager::SaveDecoderKvCache(const ReqKey &req_key, const std::shared_ptr<FlowMsg> &kv_cache) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto iter = decoder_req_to_kv_cache_.find(req_key);
    if (iter != decoder_req_to_kv_cache_.end()) {
        UDF_LOG_ERROR("Exist kv cache in decoder, req_id:%lu, model_id:%lu.", req_key.req_id, req_key.model_id);
    }
    decoder_req_to_kv_cache_[req_key] = kv_cache;
    UDF_LOG_INFO("Success to save kv cache for decoder, req_id:%lu, model_id:%lu.", req_key.req_id, req_key.model_id);
    return FsmStatus::kFsmSuccess;
}

FsmStatus KvCacheManager::ReleaseDecoderKvCache(const ReqKey &req_key) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto iter = decoder_req_to_kv_cache_.find(req_key);
    if (iter == decoder_req_to_kv_cache_.end()) {
        UDF_LOG_INFO("Not exist kv cache in decoder, req_id:%lu, model_id:%lu.", req_key.req_id, req_key.model_id);
        return FsmStatus::kFsmSuccess;
    }
    (void) decoder_req_to_kv_cache_.erase(iter);
    return FsmStatus::kFsmSuccess;
}

FsmStatus KvCacheManager::ReleaseDecoderKvCaches() {
    std::unique_lock<std::mutex> lock(mutex_);
    (void) decoder_req_to_kv_cache_.clear();
    return FsmStatus::kFsmSuccess;
}

void KvCacheManager::SaveKvCacheForPrompt(const ReqKey &req_key,
                                          uint64_t cache_id,
                                          std::pair<int64_t, int64_t> offset_and_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    UDF_LOG_INFO("KV Cache save for prompt, req_id = %lu, model_id = %lu, cache_id = %lu, offset = %ld, size = %ld",
                 req_key.req_id, req_key.model_id, cache_id, offset_and_size.first, offset_and_size.second);
    auto &cache_entry = cache_id_to_cache_entry_[cache_id];
    cache_entry.id_to_offset_and_size[req_key.req_id] = std::move(offset_and_size);
    req_id_to_cache_id_[req_key] = cache_id;
}

void KvCacheManager::SaveKvCacheForPromptPrefix(const PrefixReqKey &prefix_req_key, uint64_t cache_id,
                                                std::pair<int64_t, int64_t> offset_and_size) {
    UDF_LOG_INFO("KV Cache save for prefix, prefix_id = %lu, cache_id = %lu, offset = %ld, size = %ld",
                 prefix_req_key.prefix_id, cache_id, offset_and_size.first, offset_and_size.second);
    std::unique_lock<std::mutex> lock(mutex_);
    const auto iter = prefix_id_to_cache_id_.find(prefix_req_key);
    if (iter != prefix_id_to_cache_id_.end()) {
        UDF_LOG_INFO("Exist kv cache in prompt, prefix_id:%lu.", prefix_req_key.prefix_id);
    }
    auto &cache_entry = cache_id_to_cache_entry_[cache_id];
    cache_entry.id_to_offset_and_size[prefix_req_key.prefix_id] = std::move(offset_and_size);
    prefix_id_to_cache_id_[prefix_req_key] = cache_id;
}

FsmStatus KvCacheManager::ReleaseKvCacheForPromptByKey(const ReqKey &req_key) {
    UDF_LOG_INFO("Release KV Cache for prompt, req_id=%lu, model_id=%lu", req_key.req_id, req_key.model_id);
    const auto iter = req_id_to_cache_id_.find(req_key);
    if (iter == req_id_to_cache_id_.end()) {
        UDF_LOG_INFO("Not exist kv cache in prompt, req_id:%lu.", req_key.req_id);
        return FsmStatus::kFsmSuccess;
    }
    const auto cache_id = iter->second;
    if (cache_id_to_cache_entry_.count(cache_id) > 0UL) {
        auto &cache_entry = cache_id_to_cache_entry_[cache_id];
        if (cache_entry.id_to_offset_and_size.count(req_key.req_id) > 0UL) {
            (void) cache_entry.id_to_offset_and_size.erase(req_key.req_id);
        }
        if (cache_entry.id_to_offset_and_size.empty()) {
            (void) cache_id_to_cache_entry_.erase(cache_id);
            UDF_LOG_INFO("KV Cache removed, cache id = %lu", cache_id);
        }
    }
    (void) req_id_to_cache_id_.erase(iter);
    return FsmStatus::kFsmSuccess;
}

FsmStatus KvCacheManager::ReleaseKvCacheForPrompt(const ReqKey &req_key) {
    std::unique_lock<std::mutex> lock(mutex_);
    return ReleaseKvCacheForPromptByKey(req_key);
}

FsmStatus KvCacheManager::ReleaseKvCacheForPrompt(uint64_t req_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    UDF_LOG_INFO("Release KV Cache for prompt, req_id=%lu", req_id);
    std::vector<ReqKey> need_delete_req_keys;
    for (const auto &it: req_id_to_cache_id_) {
        if (it.first.req_id == req_id) {
            need_delete_req_keys.emplace_back(it.first);
        }
    }
    for (const auto &req_key: need_delete_req_keys) {
        (void) ReleaseKvCacheForPromptByKey(req_key);
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus KvCacheManager::ReleaseKvCacheForPrefix(const PrefixReqKey &prefix_req_key) {
    std::unique_lock<std::mutex> lock(mutex_);
    UDF_LOG_INFO("Release KV Cache for prefix, prefix_id=%lu, model_id=%lu", prefix_req_key.prefix_id, prefix_req_key.model_id);
    const auto iter = prefix_id_to_cache_id_.find(prefix_req_key);
    if (iter == prefix_id_to_cache_id_.end()) {
        UDF_LOG_INFO("Not exist kv cache in prompt, prefix_id:%lu.", prefix_req_key.prefix_id);
        return FsmStatus::kFsmSuccess;
    }
    const auto cache_id = iter->second;
    auto &cache_entry = cache_id_to_cache_entry_[cache_id];
    (void) cache_entry.id_to_offset_and_size.erase(prefix_req_key.prefix_id);
    if (cache_entry.id_to_offset_and_size.empty()) {
        (void) cache_id_to_cache_entry_.erase(cache_id);
        UDF_LOG_INFO("KV Cache removed, cache id = %lu", cache_id);
    }
    (void) prefix_id_to_cache_id_.erase(iter);
    return FsmStatus::kFsmSuccess;
}

void KvCacheManager::SetDecoderKvCache(const std::vector<std::shared_ptr<FlowMsg>> &kv_caches) {
    decoder_kv_caches_ = kv_caches;
}

std::shared_ptr<FlowMsg> KvCacheManager::GetDecoderKvCache(uint64_t model_id) const {
    return decoder_kv_caches_[model_id];
}

void KvCacheManager::ClearDecoderKvCache()
{
    decoder_kv_caches_.clear();
}

uint64_t KvCacheManager::GetPromptKvCacheCount() {
    std::unique_lock<std::mutex> lock(mutex_);
    uint64_t count = 0;
    std::set<uint64_t> prompt_cache_ids;
    for (const auto &req_id_and_cache_id: req_id_to_cache_id_) {
        prompt_cache_ids.insert(req_id_and_cache_id.second);
    }
    for (const auto cache_id: prompt_cache_ids) {
        const auto batch_num = cache_id_to_cache_entry_[cache_id].batch_num;
        UDF_LOG_INFO("cache id = %lu, batch_num = %zu", cache_id, batch_num);
        count += batch_num;
    }
    return count;
}

uint64_t KvCacheManager::GetDecoderKvCacheCount(const ReqKey &req_key) {
    std::unique_lock<std::mutex> lock(mutex_);
    return decoder_req_to_kv_cache_.count(req_key);
}

std::vector<KvTensor> KvCacheManager::QueryKvTensor(uint64_t cache_id, uint64_t tensor_id) {
    auto &cache_entry = cache_id_to_cache_entry_[cache_id];
    const auto &offset_and_size = cache_entry.id_to_offset_and_size[tensor_id];
    std::vector<KvTensor> kv_tensors;
    if (!cache_entry.kv_tensors.empty()) {
        kv_tensors.reserve(cache_entry.kv_tensors.size() * cache_entry.kv_tensors[0]->GetTensorList().size());
    }
    for (const auto &kv_tensor_buffer: cache_entry.kv_tensors) {
        const auto tensor_list = kv_tensor_buffer->GetTensorList();
        for (const auto &tensor : tensor_list) {
            KvTensor kv_tensor;
            kv_tensor.tensor_buffer = kv_tensor_buffer;
            kv_tensor.data_addr = static_cast<uint8_t *>(tensor->GetData()) + offset_and_size.first;
            kv_tensor.data_size = offset_and_size.second;
            kv_tensor.block_len = kv_tensor.data_size;
            kv_tensors.emplace_back(std::move(kv_tensor));
        }
    }
    return kv_tensors;
}

std::vector<KvTensor> KvCacheManager::QueryPromptKvCache(const ReqKey &req_key) {
    static std::vector<KvTensor> empty_vector = {};
    std::unique_lock<std::mutex> lock(mutex_);
    const auto iter = req_id_to_cache_id_.find(req_key);
    if (iter == req_id_to_cache_id_.end()) {
        UDF_LOG_INFO("Not exist kv cache in prompt, req_id:%lu.", req_key.req_id);
        return empty_vector;
    }
    const auto cache_id = iter->second;
    return QueryKvTensor(cache_id, req_key.req_id);
}

std::vector<KvTensor> KvCacheManager::QueryPromptPrefixKvCache(const PrefixReqKey &prefix_req_key) {
    static std::vector<KvTensor> empty_vector = {};
    std::unique_lock<std::mutex> lock(mutex_);
    const auto iter = prefix_id_to_cache_id_.find(prefix_req_key);
    if (iter == prefix_id_to_cache_id_.end()) {
        UDF_LOG_INFO("Not exist kv cache in prompt, req_id:%lu.", prefix_req_key.prefix_id);
        return empty_vector;
    }
    const auto cache_id = iter->second;
    return QueryKvTensor(cache_id, prefix_req_key.prefix_id);
}

void KvCacheManager::InitBlockTensors(uint64_t model_id) {
    kv_blocks_tensors_.resize(model_id, std::vector<std::shared_ptr<FlowMsg>>());
}

void KvCacheManager::SaveKvBlocksTensor(uint64_t model_id, std::shared_ptr<FlowMsg> &kv_blocks_tensor) {
    kv_blocks_tensors_[model_id].emplace_back(kv_blocks_tensor);
}

std::vector<std::shared_ptr<FlowMsg>> &KvCacheManager::QueryKvBlocksTensors(uint64_t model_id) {
    return kv_blocks_tensors_[model_id];
}

void KvCacheManager::ClearKvBlocksTensors() {
    kv_blocks_tensors_.clear();
}

uint64_t KvCacheManager::AddKvCache(size_t batch_num, std::vector<std::shared_ptr<FlowMsg>> kv_tensors) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto cache_id = cache_id_gen_++;
    CacheEntry cache_entry;
    cache_entry.batch_num = batch_num;
    cache_entry.kv_tensors = std::move(kv_tensors);
    cache_id_to_cache_entry_.emplace(cache_id, std::move(cache_entry));
    UDF_LOG_INFO("KV Cache added, cache id = %lu, batch_num = %zu", cache_id, batch_num);
    return cache_id;
}

FsmStatus KvCacheManager::ShrinkKvCacheForPrefix(const PrefixReqKey &prefix_req_key,
                                                 const std::shared_ptr<MetaRunContext> &run_context) {
    UDF_LOG_INFO("Shrink start, prefix_id:%lu.", prefix_req_key.prefix_id);
    std::vector<std::shared_ptr<FlowMsg>> shrinked_kv_tensors;
    int64_t tensor_size = -1;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto &cache_id = prefix_id_to_cache_id_[prefix_req_key];
        auto &cache_entry = cache_id_to_cache_entry_[cache_id];
        const auto &offset_and_size = cache_entry.id_to_offset_and_size[prefix_req_key.prefix_id];
        tensor_size = offset_and_size.second;
        shrinked_kv_tensors.reserve(cache_entry.kv_tensors.size());
        for (auto &kv_tensor : cache_entry.kv_tensors) {
            auto new_kv_tensor = run_context->AllocTensorMsg({tensor_size}, TensorDataType::DT_UINT8);
            if (new_kv_tensor == nullptr) {
                UDF_LOG_ERROR("Shink kv cache failed. prefix_id = %lu, Currently cache size:%lu",
                              prefix_req_key.prefix_id, KvCacheManager::GetInstance().GetPromptKvCacheCount());
                return FsmStatus::kFsmOutOfMemory;
            }
            auto rt_ret = 
                halSdmaCopy(reinterpret_cast<uintptr_t>(new_kv_tensor->GetTensor()->GetData()),
                            tensor_size,
                            reinterpret_cast<uintptr_t>(static_cast<uint8_t *>(kv_tensor->GetTensor()->GetData()) +
                                                                            offset_and_size.first),
                            tensor_size);
            if (rt_ret != DRV_ERROR_NONE) {
                UDF_LOG_ERROR("Fail to call halSdmaCopy, kv cache size:%ld, ret:%d.", tensor_size, rt_ret);
                return FsmStatus::kFsmFailed;
            }
            shrinked_kv_tensors.emplace_back(std::move(new_kv_tensor));
            kv_tensor.reset();
        }
    }
    UDF_LOG_INFO("copy tensors ended, prefix_id:%lu.", prefix_req_key.prefix_id);
    auto new_cache_id = AddKvCache(1U, shrinked_kv_tensors);
    (void) ReleaseKvCacheForPrefix(prefix_req_key);
    SaveKvCacheForPromptPrefix(prefix_req_key, new_cache_id, std::pair<int64_t, int64_t>(0L, tensor_size));
    return FsmStatus::kFsmSuccess;
}

void KvCacheManager::ClearPromptKvCache() {
    std::unique_lock<std::mutex> lock(mutex_);
    cache_id_to_cache_entry_.clear();
    req_id_to_cache_id_.clear();
    prefix_id_to_cache_id_.clear();
}
}  // namespace FlowFunc