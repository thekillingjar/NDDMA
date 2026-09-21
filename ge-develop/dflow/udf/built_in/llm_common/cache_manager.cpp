/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_common/cache_manager.h"
#include <numeric>
#include "flow_func/flow_func_log.h"
#include "flow_func/meta_flow_func.h"
#include "common/data_utils.h"
#include "ascend_hal.h"

namespace FlowFunc {
namespace {
constexpr uint32_t kMemAlign = 512U;
}  // namespace

CacheManager &CacheManager::GetInstance() {
    static CacheManager manager;
    return manager;
}

FsmStatus CacheManager::AllocateCache(const std::shared_ptr<MetaRunContext> &run_context,
                                      const AllocateCacheReqInfo &req_info,
                                      uint64_t *data_addresses) {
    std::lock_guard<std::mutex> lk(mu_);
    if (req_info.num_tensors == 0U) {
        return FsmStatus::kFsmParamInvalid;
    }
    CacheEntry cache_entry;
    std::vector<CacheIndex> cache_indices;
    const auto ret = CheckAndCreateCacheIndices(req_info, cache_indices);
    if (ret != FsmStatus::kFsmSuccess) {
        return ret;
    }
    // do allocate
    std::vector<int64_t> shape(req_info.dims, req_info.dims + req_info.num_dims);
    auto dtype = static_cast<TensorDataType>(req_info.dtype);
    int64_t tensor_size = CalcDataSize(shape, dtype);
    if (tensor_size <= 0) {
        UDF_LOG_ERROR("[cache_id:%ld][Allocate] failed, tensor desc is invalid, shape=%s, dtype=%d",
                      req_info.cache_id, ToString(shape).c_str(), static_cast<int32_t>(dtype));
        return FsmStatus::kFsmParamInvalid;
    }
    cache_entry.ext_ref_count = 1U;
    cache_entry.tensor_size = static_cast<size_t>(tensor_size);
    cache_entry.batch_size = shape.empty() ? 1U : static_cast<uint32_t>(shape.front());
    cache_entry.batch_stride = tensor_size / cache_entry.batch_size;
    cache_entry.seq_len_dim_index = req_info.seq_len_dim_index;
    UDF_LOG_INFO("[cache_id:%ld][Allocate] allocate start, tensorNum = %u, tensor_size = %lu, "
                 "batch_size = %u, batch_stride = %lu, seq_len_dim_index=%d, cache_indices:%zu.",
                 req_info.cache_id, req_info.num_tensors, cache_entry.tensor_size,
                 cache_entry.batch_size, cache_entry.batch_stride, req_info.seq_len_dim_index, cache_indices.size());
    const auto num_tensors = static_cast<size_t>(req_info.num_tensors);
    cache_entry.tensors.resize(num_tensors);
    for (size_t i = 0U; i < num_tensors; ++i) {
        auto cache_tensor = run_context->AllocTensorMsgWithAlign(shape, dtype, kMemAlign);
        if (cache_tensor == nullptr) {
            UDF_LOG_ERROR("Allocate cache failed. cache_id = %ld, tensorNum = %u, tensor_size = %lu, "
                          "currentCacheNum = %zu",
                          req_info.cache_id, req_info.num_tensors, cache_entry.tensor_size, cache_id_to_entry_.size());
            return FsmStatus::kFsmOutOfMemory;
        }
        data_addresses[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cache_tensor->GetTensor()->GetData()));
        UDF_LOG_INFO("addr_to_size_ emplace data:%lu, size:%lu", data_addresses[i], cache_entry.tensor_size);
        addr_to_size_[data_addresses[i]] = cache_entry.tensor_size;
        cache_entry.tensors[i] = std::move(cache_tensor);
    }
    cache_entry.block_len = cache_entry.batch_stride;
    if (req_info.is_allocate_blocks) {
        AddCacheIndicesForBlocks(req_info, cache_indices);
        cache_entry.is_blocks = true;
        if (!cache_indices.empty()) {
            cache_entry.blocks_cache_index = cache_indices[0];
        }
    } else {
        AddCacheIndices(cache_entry, req_info, cache_indices);
    }
    cache_id_to_entry_[req_info.cache_id] = std::move(cache_entry);
    UDF_LOG_INFO("[cache_id:%ld][Allocate] allocate ended successfully", req_info.cache_id);
    return FsmStatus::kFsmSuccess;
}

void CacheManager::AddCacheIndices(CacheEntry &cache_entry, const AllocateCacheReqInfo &req_info,
                                   const std::vector<CacheIndex> &cache_indices) {
    int64_t cache_id = req_info.cache_id;
    uint32_t batch_index = 0U;
    std::vector<uint64_t> tensor_indices(cache_entry.tensors.size());
    std::iota(tensor_indices.begin(), tensor_indices.end(), 0);
    for (const auto &cache_index: cache_indices) {
        // Ensure emplace success by CheckAndCreateCacheIndices
        if (cache_index.first != UINT64_MAX) {
            if (req_info.is_prefix) {
                (void) prefix_index_to_id_.emplace(cache_index, cache_id);
            } else {
                (void) cache_index_to_id_.emplace(cache_index, cache_id);
            }
            cache_entry.id_to_batch_index_and_size[cache_index.first] = std::make_pair(batch_index, cache_entry.batch_stride);
            cache_entry.id_to_unpulled_tensor_indices[cache_index.first] =
                std::set<uint64_t>(tensor_indices.cbegin(), tensor_indices.cend());
            cache_id_and_batch_id_to_cache_index_[std::make_pair(cache_id, batch_index)] = cache_index;
            UDF_LOG_INFO("[cache_id:%ld][AddCacheIndex] cacheKey(%lu, %lu) added, batch_index = %u, size = %lu",
                         cache_id, cache_index.first, cache_index.second, batch_index, cache_entry.batch_stride);
        }
        ++batch_index;
    }
}

void CacheManager::AddCacheIndicesForBlocks(const AllocateCacheReqInfo &req_info,
                                            const std::vector<CacheIndex> &cache_indices) {
    int64_t cache_id = req_info.cache_id;
    for (const auto &cache_index: cache_indices) {
        // Ensure emplace success by CheckAndCreateCacheIndices
        (void) cache_index_to_id_.emplace(cache_index, cache_id);
        UDF_LOG_INFO("[cache_id:%ld][AddCacheIndicesForBlocks] cacheKey(%lu, %lu) added.",
                     cache_id,
                     cache_index.first,
                     cache_index.second);
    }
}

void CacheManager::RemoveAddrToSize(const CacheEntry &cache_entry) {
    for (const auto &tensor : cache_entry.tensors) {
        auto addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor->GetTensor()->GetData()));
        if (addr_to_size_.find(addr) != addr_to_size_.end()) {
            (void) addr_to_size_.erase(addr);
        }
    }
}

FsmStatus CacheManager::DeallocateCache(int64_t cache_id) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = cache_id_to_entry_.find(cache_id);
    if (it == cache_id_to_entry_.cend()) {
        UDF_LOG_INFO("[cache_id:%ld][Deallocate] failed, cache_id not exist", cache_id);
        return FsmStatus::kFsmKvNotExist;
    }
    auto &cache_entry = it->second;
    cache_entry.ext_ref_count = 0U;
    UDF_LOG_INFO("[cache_id:%ld][Deallocate] ext_ref_count was set to 0, ref_count remaining = %zu",
                 cache_id, cache_entry.id_to_batch_index_and_size.size());
    if (cache_entry.is_blocks) {
        (void) cache_index_to_id_.erase(cache_entry.blocks_cache_index);
        UDF_LOG_INFO("[cache_id:%ld][Deallocate] deallocate ended successfully", cache_id);
    }
    if (cache_entry.id_to_batch_index_and_size.empty()) {
        RemoveAddrToSize(cache_entry);
        (void) cache_id_to_entry_.erase(it);
        UDF_LOG_INFO("[cache_id:%ld][Deallocate] deallocate ended successfully", cache_id);
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus CacheManager::RemoveCacheIndex(const CacheIndex &cache_index,
                                         bool is_prefix,
                                         uint32_t num_tensor_indices,
                                         const TransferInfo *tensor_indices) {
    std::lock_guard<std::mutex> lk(mu_);
    auto &index_to_id = is_prefix ? prefix_index_to_id_ : cache_index_to_id_;
    const auto cache_id_it = index_to_id.find(cache_index);
    if (cache_id_it == index_to_id.cend()) {
        UDF_LOG_INFO("[RemoveIndex] cacheKey (%lu, %lu) not exist", cache_index.first, cache_index.second);
        return FsmStatus::kFsmKvNotExist;
    }
    const auto cache_id = cache_id_it->second;
    // 保证cache index指向的cache entry必然存在
    const auto it = cache_id_to_entry_.find(cache_id);
    auto &cache_entry = it->second;
    auto batch_index_and_size_it = cache_entry.id_to_batch_index_and_size.find(cache_index.first);
    if (batch_index_and_size_it != cache_entry.id_to_batch_index_and_size.cend()) {
        if (num_tensor_indices > 0 && num_tensor_indices < cache_entry.tensors.size()) {
            auto &unpulled = cache_entry.id_to_unpulled_tensor_indices[cache_index.first];
            for (uint32_t i = 0U; i < num_tensor_indices; ++i) {
                unpulled.erase(tensor_indices[i].tensor_index);
            }
            UDF_LOG_INFO("[RemoveIndex] cacheKey (%lu, %lu) partial pull, unpulled = %s",
                         cache_index.first, cache_index.second,
                         ToString(std::vector<uint64_t>(unpulled.cbegin(), unpulled.cend())).c_str());
            if (!unpulled.empty()) {
                return FsmStatus::kFsmSuccess;
            }
        }
        const auto batch_index = batch_index_and_size_it->second.first;
        (void) cache_id_and_batch_id_to_cache_index_.erase(std::make_pair(cache_id, batch_index));
        (void) cache_entry.id_to_batch_index_and_size.erase(batch_index_and_size_it);
    }
    UDF_LOG_INFO("[cache_id:%ld][RemoveIndex] cacheKey (%lu, %lu) was removed, ref_count = %zu, ext_ref_count = %zu",
                 cache_id, cache_index.first, cache_index.second,
                 cache_entry.id_to_batch_index_and_size.size(), cache_entry.ext_ref_count);
    const auto ref_count = cache_entry.id_to_batch_index_and_size.size() + cache_entry.ext_ref_count;
    (void) index_to_id.erase(cache_id_it);
    if (ref_count == 0) {
        RemoveAddrToSize(cache_entry);
        (void) cache_id_to_entry_.erase(it);
        UDF_LOG_INFO("[cache_id:%ld][Deallocate] ref_count = 0 and was deallocated", cache_id);
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus CacheManager::CopyBlockCache(const CopyCacheReqInfo &req_info) const {
    CacheEntry cache_entry;
    const auto cache_entry_exist = GetCacheEntry(req_info.dst_cache_id, cache_entry);
    if (!cache_entry_exist) {
        UDF_LOG_ERROR("[CopyCache](%ld->%ld) failed, dst_cache_exist = %d",
                      req_info.src_cache_id, req_info.dst_cache_id,
                      static_cast<int32_t>(cache_entry_exist));
        return FsmStatus::kFsmKvNotExist;
    }
    for (uint32_t i = 0; i < req_info.copy_block_count; ++i) {
        if ((req_info.block_copy_infos[i].src_block_index >= cache_entry.batch_size) ||
            (req_info.block_copy_infos[i].dst_block_index >= cache_entry.batch_size)) {
            UDF_LOG_ERROR("[cache_id:%ld][CopyBlockCache] copyBlockIndex (%lu, %lu) out of range [0, %zu).",
                          req_info.dst_cache_id, req_info.block_copy_infos[i].src_block_index,
                          req_info.block_copy_infos[i].dst_block_index, cache_entry.batch_size);
            return FsmStatus::kFsmParamInvalid;
        }
    }
    for (const auto &kv_tensor: cache_entry.tensors) {
        auto data_addr = static_cast<uint8_t *>(kv_tensor->GetTensor()->GetData());
        for (uint32_t i = 0; i < req_info.copy_block_count; ++i) {
            auto src_mem = data_addr + req_info.block_copy_infos[i].src_block_index * cache_entry.block_len;
            auto dst_mem = data_addr + req_info.block_copy_infos[i].dst_block_index * cache_entry.block_len;
            const auto copy_ret =
                halSdmaCopy(reinterpret_cast<uintptr_t>(dst_mem), cache_entry.block_len,
                            reinterpret_cast<uintptr_t>(src_mem), cache_entry.block_len);
            if (copy_ret != DRV_ERROR_NONE) {
                UDF_LOG_ERROR("[CopyCache][%ld] halSdmaCopy failed, tensor index = %zu, ret = %d",
                              req_info.dst_cache_id, i, copy_ret);
                return FsmStatus::kFsmCopyKvFailed;
            }
        }
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus CacheManager::CopyCache(const CopyCacheReqInfo &req_info) const {
    // 拷贝出来, 防止copy过程中tensor被销毁
    if (req_info.copy_block_count > 0U) {
        return CopyBlockCache(req_info);
    }
    CacheEntry src_cache_entry;
    CacheEntry dst_cache_entry;
    const auto src_cache_entry_exist = GetCacheEntry(req_info.src_cache_id, src_cache_entry);
    const auto dst_cache_entry_exist = GetCacheEntry(req_info.dst_cache_id, dst_cache_entry);
    if ((!src_cache_entry_exist) || (!dst_cache_entry_exist)) {
        UDF_LOG_ERROR("[CopyCache](%ld->%ld) failed, src_cache_exist = %d, dst_cache_exist = %d",
                      req_info.src_cache_id, req_info.dst_cache_id,
                      static_cast<int32_t>(src_cache_entry_exist), static_cast<int32_t>(dst_cache_entry_exist));
        return FsmStatus::kFsmKvNotExist;
    }
    uint64_t copy_size;
    if (CheckCopyCacheParam(src_cache_entry, dst_cache_entry, req_info, copy_size) != FsmStatus::kFsmSuccess) {
        return FsmStatus::kFsmParamInvalid;
    }
    const auto dst_offset = dst_cache_entry.batch_stride * req_info.dst_batch_index + req_info.offset;
    const auto src_offset = src_cache_entry.batch_stride * req_info.src_batch_index + req_info.offset;
    const auto dst_max = dst_cache_entry.batch_stride - req_info.offset;
    UDF_LOG_INFO("[CopyCache](%ld->%ld) start, num_tensors = %zu, copy_size = %lu",
                 req_info.src_cache_id, req_info.dst_cache_id, src_cache_entry.tensors.size(), copy_size);
    UDF_LOG_INFO("Src: [batch_index = %u, offset = %lu], Dst: [BatchIndex = %u, offset = %lu]",
                 req_info.src_batch_index, src_offset, req_info.dst_batch_index, dst_offset);
    for (size_t i = 0U; i < src_cache_entry.tensors.size(); ++i) {
        auto src_mem_base = static_cast<uint8_t *>(src_cache_entry.tensors[i]->GetTensor()->GetData());
        auto dst_mem_base = static_cast<uint8_t *>(dst_cache_entry.tensors[i]->GetTensor()->GetData());
        auto dst_mem = dst_mem_base + dst_offset;
        auto src_men = src_mem_base + src_offset;
        const auto copy_ret = D2DCopy(dst_mem, dst_max, src_men, copy_size);
        if (copy_ret != DRV_ERROR_NONE) {
            UDF_LOG_ERROR("[CopyCache][%ld->%ld] halSdmaCopy failed, tensor index = %zu, ret = %d",
                          req_info.src_cache_id, req_info.dst_cache_id, i, copy_ret);
            return FsmStatus::kFsmCopyKvFailed;
        }
    }
    UDF_LOG_INFO("[CopyCache](%ld->%ld) ended successfully", req_info.src_cache_id, req_info.dst_cache_id);
    return FsmStatus::kFsmSuccess;
}

FsmStatus CacheManager::CheckCopyCacheParam(const CacheEntry &src_cache_entry,
                                            const CacheEntry &dst_cache_entry,
                                            const CopyCacheReqInfo &req_info,
                                            uint64_t &copy_size) {
    if (dst_cache_entry.tensors.size() != src_cache_entry.tensors.size()) {
        UDF_LOG_ERROR("[CopyCache](%ld->%ld) failed, tensor number mismatches, src = %zu, dst = %zu",
                      req_info.src_cache_id,
                      req_info.dst_cache_id,
                      src_cache_entry.tensors.size(),
                      dst_cache_entry.tensors.size());
        return FsmStatus::kFsmCacheIncompatible;
    }
    if (req_info.src_batch_index >= src_cache_entry.batch_size || req_info.dst_batch_index >= dst_cache_entry.batch_size) {
        UDF_LOG_ERROR("[CopyCache](%ld->%ld) failed, batch_index out of range, "
                      "src: index = %u, range = [0, %u), dst: index = %u, range = [0, %u)",
                      req_info.src_cache_id, req_info.dst_cache_id,
                      req_info.src_batch_index, src_cache_entry.batch_size, req_info.dst_batch_index, dst_cache_entry.batch_size);
        return FsmStatus::kFsmParamInvalid;
    }
    if ((req_info.offset >= dst_cache_entry.batch_stride) || (req_info.offset >= src_cache_entry.batch_stride)) {
        UDF_LOG_ERROR("[CopyCache](%ld->%ld) failed, offset out of range, "
                      "offset=%lu, src_stride=%lu, dst_stride=%lu",
                      req_info.src_cache_id, req_info.dst_cache_id,
                      req_info.offset, src_cache_entry.batch_stride, dst_cache_entry.batch_stride);
        return FsmStatus::kFsmParamInvalid;
    }
    copy_size = req_info.copy_size != -1 ?
               static_cast<uint64_t>(req_info.copy_size) :
               src_cache_entry.batch_stride - req_info.offset;
    if ((copy_size > (src_cache_entry.batch_stride - req_info.offset)) ||
        (copy_size > (dst_cache_entry.batch_stride - req_info.offset))) {
        UDF_LOG_ERROR("[CopyCache](%ld->%ld) failed, copy_size out of range, "
                      "offset=%lu, src_stride=%lu, dst_stride=%lu, size=%ld",
                      req_info.src_cache_id, req_info.dst_cache_id,
                      req_info.offset, src_cache_entry.batch_stride, dst_cache_entry.batch_stride, req_info.copy_size);
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus CacheManager::CheckTensorIndices(const PullKvReqInfo &req_info,
                                           const CacheEntry &cache_entry,
                                           bool enable_page_attention) {
    const uint64_t *tensor_indices = LocalTensorIndices(req_info, enable_page_attention);
    for (uint32_t i = 0U; i < req_info.dst_tensor_index_count; ++i) {
      if (tensor_indices[i] >= cache_entry.tensors.size()) {
        UDF_LOG_ERROR("[cache_id:%ld][Pull] dstTensorIndex[%u] = %lu, out of range, tensorNum = %zu.",
                      req_info.cache_id, i, tensor_indices[i], cache_entry.tensors.size());
        return FsmStatus::kFsmParamInvalid;
      }
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus CacheManager::CheckReqInfo(const CacheEntry &cache_entry, PullKvReqInfo *req_info, bool &enable_page_attention) {
    if (req_info->batch_index >= cache_entry.batch_size) {
        UDF_LOG_ERROR("[cache_id:%ld][Pull] batch_index (%lu) out of range [0, %zu).",
                      req_info->cache_id, req_info->batch_index, cache_entry.batch_size);
        return FsmStatus::kFsmParamInvalid;
    }
    enable_page_attention = (req_info->block_count > 0U);
    if (enable_page_attention
        && ClusterManager::CheckBlockIndices(req_info, cache_entry.batch_size) != FsmStatus::kFsmSuccess) {
        return FsmStatus::kFsmParamInvalid;
    }
    if (CheckTensorIndices(*req_info, cache_entry, enable_page_attention) != FsmStatus::kFsmSuccess) {
        return FsmStatus::kFsmParamInvalid;
    }
    if (enable_page_attention && req_info->block_count != req_info->prompt_block_count) {
        UDF_LOG_ERROR("[cache_id:%ld][Pull] block_count(%lu) != prompt_block_count(%lu).",
                      req_info->cache_id, req_info->block_count, req_info->prompt_block_count);
        return FsmStatus::kFsmParamInvalid;
    }
    // check single block size < INT32_MAX
    if (enable_page_attention && (cache_entry.block_len > INT32_MAX)) {
        UDF_LOG_ERROR("[cache_id:%ld][Pull] block_len(%lu) is over limit.", req_info->cache_id, cache_entry.block_len);
        return FsmStatus::kFsmParamInvalid;
    }
    req_info->block_count = (req_info->block_count == 0U) ? 1U : req_info->block_count;
    if (req_info->block_len == 0U) {
        req_info->block_len = (cache_entry.batch_stride - req_info->dst_cache_offset);
    }
    if ((req_info->dst_cache_offset + req_info->block_len) > cache_entry.batch_stride) {
        UDF_LOG_ERROR("[cache_id:%ld][Pull] dst_cache_offset(%lu) or block_len(%lu) is invalid, cache batch stride(%lu).",
                      req_info->cache_id, req_info->dst_cache_offset, req_info->block_len, cache_entry.batch_stride);
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus CacheManager::PullCache(PullKvReqInfo *req_info, ClusterManager &cluster_manager, uint64_t start_tick) const {
    CacheEntry cache_entry;
    if (!GetCacheEntry(req_info->cache_id, cache_entry)) {
        UDF_LOG_ERROR("[cache_id:%ld][Pull] cache_id not found.", req_info->cache_id);
        return FsmStatus::kFsmKvNotExist;
    }
    bool enable_page_attention = false;
    auto ret = CheckReqInfo(cache_entry, req_info, enable_page_attention);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("[cache_id:%ld][Pull] check req info failed, ret = %d.",
                      req_info->cache_id, static_cast<int32_t>(ret));
        return ret;
    }
    UDF_LOG_INFO("[cache_id:%ld][Pull] start, %s, num_tensors=%zu, block_len = %lu",
                 req_info->cache_id, CacheKeyDebugString(*req_info).c_str(), cache_entry.tensors.size(), req_info->block_len);
    ret = cluster_manager.PullCacheFromPrompt(req_info, enable_page_attention, start_tick, &cache_entry);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("[cache_id:%ld][Pull] failed, ret = %d.", req_info->cache_id, static_cast<int32_t>(ret));
        return ret;
    }
    UDF_LOG_INFO("[cache_id:%ld][Pull] success, %s, num_tensors=%zu, block_len = %lu",
                 req_info->cache_id, CacheKeyDebugString(*req_info).c_str(), cache_entry.tensors.size(), req_info->block_len);
    return FsmStatus::kFsmSuccess;
}

FsmStatus CacheManager::TransferCache(TransferKvReqInfo *req_info, ClusterManager &cluster_manager,
                                      uint64_t start_tick) const {
    CacheEntry cache_entry;
    if (!GetCacheEntry(req_info->cache_id, cache_entry)) {
        UDF_LOG_ERROR("[cache_id:%ld][Transfer] cache_id not found.", req_info->cache_id);
        return FsmStatus::kFsmKvNotExist;
    }
    if (cache_entry.tensors.size() % req_info->tensor_num_per_layer != 0) {
        UDF_LOG_ERROR("[cache_id:%ld][Transfer] tensors num:%zu is not a multiple of %lu.", req_info->cache_id, cache_entry.tensors.size(),
                      req_info->tensor_num_per_layer);
        return FsmStatus::kFsmParamInvalid;
    }
    uint64_t max_layer_index = cache_entry.tensors.size() / req_info->tensor_num_per_layer;
    if (req_info->src_layer_index >= max_layer_index) {
        UDF_LOG_ERROR("[cache_id:%ld][Transfer] layer index:%lu out of range [0, %lu).", req_info->cache_id,
                      req_info->src_layer_index, max_layer_index);
        return FsmStatus::kFsmParamInvalid;
    }
    if (req_info->src_batch_index >= cache_entry.batch_size) {
        UDF_LOG_ERROR("[cache_id:%ld][Transfer] src_batch_index (%lu) out of range [0, %zu).",
                      req_info->cache_id, req_info->src_batch_index, cache_entry.batch_size);
        return FsmStatus::kFsmParamInvalid;
    }
    if (req_info->block_len > cache_entry.batch_stride) {
        UDF_LOG_ERROR("[cache_id:%ld][Transfer] block_len(%lu) > cached batch_stride(%lu).",
                      req_info->cache_id, req_info->block_len, cache_entry.batch_stride);
        return FsmStatus::kFsmParamInvalid;
    }
    UDF_LOG_INFO("[cache_id:%ld][Transfer] start, %s, num_tensors=%zu, block_len = %lu, dst_cache_id:%ld.", req_info->cache_id,
                 TransferReqDebugString(*req_info).c_str(), cache_entry.tensors.size(), req_info->block_len, req_info->dst_cache_id);
    bool enable_page_attention = (req_info->block_count > 0U);
    if (enable_page_attention && req_info->block_count != req_info->prompt_block_count) {
        UDF_LOG_ERROR("[cache_id:%ld][Transfer] block_count(%lu) != prompt_block_count(%lu).",
                      req_info->cache_id, req_info->block_count, req_info->prompt_block_count);
        return FsmStatus::kFsmParamInvalid;
    }
    if (enable_page_attention
        && ClusterManager::CheckBlockIndices(req_info, cache_entry.batch_size, req_info->block_count) != FsmStatus::kFsmSuccess) {
        return FsmStatus::kFsmParamInvalid;
    }
    req_info->block_count = (req_info->block_count == 0U) ? 1U : req_info->block_count;
    if (req_info->block_len == 0U) {
        req_info->block_len = cache_entry.batch_stride;
    }
    if (enable_page_attention && req_info->block_len != cache_entry.batch_stride) {
        UDF_LOG_ERROR("[cache_id:%ld][Transfer] block_len(%lu) != cached batch_stride(%lu).",
                      req_info->cache_id, req_info->block_len, cache_entry.batch_stride);
        return FsmStatus::kFsmParamInvalid;
    }
    auto ret = cluster_manager.TransferCache(req_info, enable_page_attention, start_tick, &cache_entry);
    if (ret != FsmStatus::kFsmSuccess) {
        UDF_LOG_ERROR("[cache_id:%ld][Transfer] failed, ret = %d.", req_info->cache_id, static_cast<int32_t>(ret));
        return ret;
    }
    UDF_LOG_INFO("[cache_id:%ld][Transfer] success, %s, num_tensors=%zu, block_len = %lu, dst_cache_id:%ld.", req_info->cache_id,
                 TransferReqDebugString(*req_info).c_str(), cache_entry.tensors.size(), req_info->block_len, req_info->dst_cache_id);
    return FsmStatus::kFsmSuccess;
}

bool CacheManager::GetCacheEntry(const CacheIndex &cache_index, CacheEntry &cache_entry, bool is_prefix) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto &index_to_id = is_prefix ? prefix_index_to_id_ : cache_index_to_id_;
    const auto it = index_to_id.find(cache_index);
    if (it == index_to_id.cend()) {
        return false;
    }
    // cacheIndex存在, 则cache entry必然存在
    cache_entry = *DoGetCacheEntry(it->second);
    return true;
}

bool CacheManager::GetCacheEntry(int64_t cache_id, CacheEntry &cache_entry) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto entry = DoGetCacheEntry(cache_id);
    bool success = (entry != nullptr);
    if (success) {
        cache_entry = *entry;
    }
    return success;
}

bool CacheManager::GetCacheIndex(const std::pair<int64_t, uint32_t> &cache_id_and_batch_index, CacheIndex &cache_index) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = cache_id_and_batch_id_to_cache_index_.find(cache_id_and_batch_index);
    if (it != cache_id_and_batch_id_to_cache_index_.cend()) {
        cache_index = it->second;
        return true;
    }
    return false;
}

const CacheEntry *CacheManager::DoGetCacheEntry(int64_t cache_id) const {
    const auto it = cache_id_to_entry_.find(cache_id);
    return (it == cache_id_to_entry_.cend()) ? nullptr : &it->second;
}

void CacheManager::Enable() {
    enabled_ = true;
}

bool CacheManager::IsEnabled() const {
    return enabled_;
}

FsmStatus CacheManager::CheckAndCreateCacheIndices(const AllocateCacheReqInfo &req_info,
                                                   std::vector<CacheIndex> &cache_indices) const {
    if (cache_id_to_entry_.find(req_info.cache_id) != cache_id_to_entry_.cend()) {
        UDF_LOG_ERROR("[cache_id:%ld][Allocate] check failed, cache_id already exist", req_info.cache_id);
        return FsmStatus::kFsmCacheIdAlreadyExist;
    }
    cache_indices.reserve(req_info.num_requests);
    auto &index_to_id = req_info.is_prefix ? prefix_index_to_id_ : cache_index_to_id_;
    for (size_t i = 0; i < req_info.num_requests; ++i) {
        cache_indices.emplace_back(req_info.req_ids[i], req_info.model_id);
        const auto it = index_to_id.find(cache_indices.back());
        if (it != index_to_id.cend()) {
            UDF_LOG_ERROR("[cache_id:%ld][Allocate] check failed, cacheKey (%lu, %lu) already mapped to Cache [%ld]",
                          req_info.cache_id, req_info.req_ids[i], req_info.model_id, it->second);
            return FsmStatus::kFsmCacheKeyAlreadyExist;
        }
    }
    return FsmStatus::kFsmSuccess;
}

FsmStatus CacheManager::CheckAddr(uint64_t addr, uint64_t check_size) {
    const auto it = addr_to_size_.find(addr);
    if (it == addr_to_size_.end()) {
        for (auto &addr_size_pair : addr_to_size_) {
            auto exist_addr = addr_size_pair.first;
            if ((addr > exist_addr) && ((addr + check_size) <= (exist_addr + addr_size_pair.second))) {
                return FsmStatus::kFsmSuccess;
            }
        }
        return FsmStatus::kFsmParamInvalid;
    }
    if (it->second < check_size) {
        UDF_LOG_ERROR("Addr size is over limit, real size:%lu, req size:%lu.", it->second, check_size);
        return FsmStatus::kFsmParamInvalid;
    }
    return FsmStatus::kFsmSuccess;
}
}  // namespace FlowFunc

