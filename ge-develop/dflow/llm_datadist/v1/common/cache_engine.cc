/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/llm_log.h"
#include "common/swap_impl.h"
#include "common/cache_engine.h"
#include "graph/utils/type_utils.h"
#include "common/llm_checker.h"
#include "common/llm_scope_guard.h"

namespace llm {
namespace {
constexpr size_t kMaxDimNum = 32U;
constexpr size_t kValidCacheSizePerLayer = 2U;
}  // namespace

ge::Status CacheEngine::Allocate(const CacheDesc &cache_desc,
                                 const std::vector<CacheKey> &cache_keys,
                                 Cache &cache) {
  LLM_CHK_BOOL_RET_STATUS(cache_desc.placement == static_cast<uint32_t>(CachePlacement::DEVICE),
                         ge::LLM_PARAM_INVALID,
                         "[Allocate] failed, only support device cache");
  const auto cache_id = cache_id_gen_.fetch_add(1, std::memory_order::memory_order_relaxed);
  LLM_CHECK_GE(cache_id, 1);  // actually not possible
  LLM_CHK_BOOL_RET_STATUS(!cache_desc.shape.empty() && (cache_desc.shape.size() < kMaxDimNum),
                         ge::LLM_PARAM_INVALID,
                         "Invalid shape: %s, dim_num (%zu) must be in range [0, 33)",
                         llm::ToString(cache_desc.shape).c_str(), cache_desc.shape.size());
  cache.cache_id = cache_id;
  const auto ret = llm_flow_service_->Allocate(cache_desc, cache_keys, cache);
  LLM_CHK_STATUS(ret, "[cache_id:%ld][Allocate] failed, num_tensors = %u, shape = %s, dtype = %s",
                cache_id, cache_desc.num_tensors, llm::ToString(cache_desc.shape).c_str(),
                ge::TypeUtils::DataTypeToSerialString(cache_desc.data_type).c_str());
  LLM_CHK_STATUS_RET_NOLOG(ret);
  {
    std::lock_guard<std::mutex> lk(mu_);
    (void) cache_id_to_desc_.emplace(cache_id, cache_desc);
  }
  LLMLOGI("[cache_id:%ld][Allocate] success, num_tensors = %u, shape = %s",
         cache_id, cache_desc.num_tensors, llm::ToString(cache_desc.shape).c_str());
  return ge::SUCCESS;
}

ge::Status CacheEngine::Deallocate(int64_t cache_id) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto it = cache_id_to_desc_.find(cache_id);
  LLM_CHK_BOOL_RET_STATUS(it != cache_id_to_desc_.cend(),
                         ge::LLM_KV_CACHE_NOT_EXIST,
                         "[cache_id:%ld][Deallocate] failed, cache_id not found", cache_id);
  LLM_CHK_BOOL_RET_STATUS(it->second.placement == static_cast<uint32_t>(CachePlacement::DEVICE),
                         ge::LLM_PARAM_INVALID,
                         "[cache_id:%ld][Deallocate] failed, only support device cache", cache_id);
  const auto ret = llm_flow_service_->Deallocate(cache_id);
  LLM_CHK_STATUS(ret, "[cache_id:%ld][Deallocate] failed", cache_id);
  LLM_CHK_STATUS_RET_NOLOG(ret);
  (void) cache_id_to_desc_.erase(it);
  LLMLOGI("[cache_id:%ld][Deallocate] success", cache_id);
  return ge::SUCCESS;
}

ge::Status CacheEngine::RemoveCacheKey(const CacheKey &cache_key) const {
  const auto ret = llm_flow_service_->RemoveCacheIndex(cache_key);
  LLM_CHK_STATUS(ret, "[RemoveCacheKey] failed, cache_key = (%lu, %lu)",
                cache_key.req_id, cache_key.model_id);
  LLM_CHK_STATUS_RET_NOLOG(ret);
  LLMLOGI("[Remove][Index] success, cache_key = (%lu, %lu)", cache_key.req_id, cache_key.model_id);
  return ge::SUCCESS;
}

ge::Status CacheEngine::PullCache(int64_t cache_id, const CacheKey &cache_key, const PullCacheParam &pull_cache_param) {
  uint32_t num_tensors;
  {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = cache_id_to_desc_.find(cache_id);
    LLM_CHK_BOOL_RET_STATUS(it != cache_id_to_desc_.cend(),
                           ge::LLM_KV_CACHE_NOT_EXIST,
                           "[cache_id:%ld][Pull] failed, cache_id not found", cache_id);
    LLM_CHK_BOOL_RET_STATUS(it->second.placement == static_cast<uint32_t>(CachePlacement::DEVICE),
                           ge::LLM_PARAM_INVALID,
                           "[cache_id:%ld][Pull] failed, only support device cache", cache_id);
    const auto batch_size = static_cast<uint32_t>(it->second.shape.front());
    LLM_CHK_BOOL_RET_STATUS(pull_cache_param.batch_index < batch_size,
                           ge::LLM_PARAM_INVALID,
                           "[cache_id:%ld][Pull] failed, dst_batch_index (%u) out of range [0, %u)",
                           cache_id, pull_cache_param.batch_index, batch_size);
    LLM_CHK_BOOL_RET_STATUS(pull_cache_param.tensor_num_per_layer >= 1,
                           ge::LLM_PARAM_INVALID,
                           "[cache_id:%ld][Pull] failed, tensor_num_per_layer (%lu) out of range [1, %u]", cache_id,
                           pull_cache_param.tensor_num_per_layer, it->second.num_tensors);
    num_tensors = it->second.num_tensors;
  }
  const auto ret = llm_flow_service_->PullCache(cache_id, cache_key, pull_cache_param, num_tensors);
  LLM_CHK_STATUS(ret, "Request[%lu] [Pull] failed, cache_id = %ld, %s",
                cache_key.req_id, cache_id, LLMUtils::DebugString(cache_key).c_str());
  LLM_CHK_STATUS_RET_NOLOG(ret);
  LLMLOGI("Request[%lu] [Pull] success, cache_id = %ld, %s",
         cache_key.req_id, cache_id, LLMUtils::DebugString(cache_key).c_str());
  return ge::SUCCESS;
}

ge::Status CacheEngine::TransferCache(uint64_t task_id, const TransferCacheConfig &transfer_cache_config,
                                      const TransferBlockConfig &transfer_block_config) {
  if (transfer_cache_config.type == 0U) {
    LLM_CHK_BOOL_RET_STATUS(transfer_cache_config.dst_addrs.size() == transfer_cache_config.tensor_num_per_layer,
                           ge::LLM_PARAM_INVALID,
                           "[cache_id:%ld][Transfer] failed, addr size is not valid:%zu, expected:%zu.",
                           transfer_cache_config.src_cache_id, transfer_cache_config.dst_addrs.size(),
                           transfer_cache_config.tensor_num_per_layer);
  }
  LLM_CHK_BOOL_RET_STATUS(transfer_block_config.src_blocks.size() == transfer_block_config.dst_blocks.size(),
                         ge::LLM_PARAM_INVALID, "Param src_blocks and dst_blocks size should be same.");
  {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = cache_id_to_desc_.find(transfer_cache_config.src_cache_id);
    LLM_CHK_BOOL_RET_STATUS(it != cache_id_to_desc_.cend(), ge::LLM_KV_CACHE_NOT_EXIST,
                           "[cache_id:%ld][Transfer] failed, cache_id not found", transfer_cache_config.src_cache_id);
    const auto batch_size = static_cast<uint32_t>(it->second.shape.front());
    LLM_CHK_BOOL_RET_STATUS(transfer_cache_config.batch_index < batch_size, ge::LLM_PARAM_INVALID,
                           "[cache_id:%ld][Transfer] failed, batch_index (%lu) out of range [0, %u)",
                           transfer_cache_config.src_cache_id, transfer_cache_config.batch_index, batch_size);
    LLM_CHK_BOOL_RET_STATUS(transfer_cache_config.tensor_num_per_layer >= 1,
                           ge::LLM_PARAM_INVALID,
                           "[cache_id:%ld][Transfer] failed, tensor_num_per_layer (%lu) out of range [1, %u]",
                           transfer_cache_config.src_cache_id, transfer_cache_config.tensor_num_per_layer, it->second.num_tensors);

    uint64_t max_layer_index = it->second.num_tensors >= transfer_cache_config.tensor_num_per_layer ?
                               it->second.num_tensors / transfer_cache_config.tensor_num_per_layer : 1;
    LLM_CHK_BOOL_RET_STATUS(transfer_cache_config.layer_index < max_layer_index, ge::LLM_PARAM_INVALID,
                           "[cache_id:%ld][Transfer], the layer_index [%zu] must be of range [0, %lu).",
                           transfer_cache_config.src_cache_id, transfer_cache_config.layer_index, max_layer_index);
    LLM_CHK_BOOL_RET_STATUS(transfer_cache_config.dst_layer_index < max_layer_index, ge::LLM_PARAM_INVALID,
                           "[cache_id:%ld][Transfer], the dst_layer_index [%zu] must be of range [0, %lu).",
                           transfer_cache_config.src_cache_id, transfer_cache_config.dst_layer_index, max_layer_index);
  }
  LLMLOGI("[Transfer] task id:%lu", task_id);
  const auto ret = llm_flow_service_->TransferCache(transfer_cache_config, transfer_block_config);
  LLM_CHK_STATUS(ret, "[Transfer] failed, cache_id = %ld, layer_index = %lu, cluster_id = %lu,",
                transfer_cache_config.src_cache_id, transfer_cache_config.layer_index,
                transfer_cache_config.cluster_id);
  LLM_CHK_STATUS_RET_NOLOG(ret);
  return ge::SUCCESS;
}

ge::Status CacheEngine::CopyCache(const CopyCacheParam &copy_cache_param) {
  const auto src_batch_index = copy_cache_param.src_batch_index;
  const auto dst_batch_index = copy_cache_param.dst_batch_index;
  const auto src_id = copy_cache_param.src_cache_id;
  const auto dst_id = copy_cache_param.dst_cache_id;
  {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it_src = cache_id_to_desc_.find(src_id);
    LLM_CHK_BOOL_RET_STATUS(it_src != cache_id_to_desc_.cend(),
                           ge::LLM_KV_CACHE_NOT_EXIST,
                           "[Copy][%ld->%ld] failed, src_cache_id not found", src_id, dst_id);
    const auto it_dst = cache_id_to_desc_.find(dst_id);
    LLM_CHK_BOOL_RET_STATUS(it_dst != cache_id_to_desc_.cend(),
                           ge::LLM_KV_CACHE_NOT_EXIST,
                           "[Copy][%ld->%ld] failed, dst_cache_id not found", src_id, dst_id);
    const auto &src_desc = it_src->second;
    const auto src_batch_size = static_cast<uint32_t>(src_desc.shape.front());
    LLM_CHK_BOOL_RET_STATUS(src_batch_index < src_batch_size,
                           ge::LLM_PARAM_INVALID,
                           "[Copy][%ld->%ld] failed, src_batch_index (%u) out of range [0, %u)",
                           src_id, dst_id, src_batch_index, src_batch_size);
    const auto &dst_desc = it_dst->second;
    const auto dst_batch_size = static_cast<uint32_t>(dst_desc.shape.front());
    LLM_CHK_BOOL_RET_STATUS(dst_batch_index < dst_batch_size,
                           ge::LLM_PARAM_INVALID,
                           "[Copy][%ld->%ld] failed, dst_batch_index (%u) out of range [0, %u)",
                           src_id, dst_id, dst_batch_index, dst_batch_size);
    LLM_CHK_BOOL_RET_STATUS(src_desc.num_tensors == dst_desc.num_tensors,
                           ge::LLM_PARAM_INVALID,
                           "[Copy][%ld->%ld] failed, num_tensors mismatches, "
                           "src_tensor_num = %u, dst_tensor_num = %u",
                           src_id, dst_id, src_desc.num_tensors, dst_desc.num_tensors);
  }
  const auto ret = llm_flow_service_->CopyCache(copy_cache_param);
  LLM_CHK_STATUS(ret, "Request[%s] [Copy][%ld->%ld] failed",
                copy_cache_param.req_id == UINT64_MAX ? "-1" : std::to_string(copy_cache_param.req_id).c_str(),
                copy_cache_param.src_cache_id,
                copy_cache_param.dst_cache_id);
  LLM_CHK_STATUS_RET_NOLOG(ret);
  LLMLOGI("[Copy][%ld->%ld] success, dst_batch_index = %u", src_id, dst_id, dst_batch_index);
  return ge::SUCCESS;
}

ge::Status CacheEngine::GetCacheTensors(int64_t cache_id,
                                        std::vector<ge::Tensor> &outputs,
                                        int32_t tensor_index) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cache_id_to_desc_.find(cache_id);
    LLM_CHK_BOOL_RET_STATUS(it != cache_id_to_desc_.cend(),
                           ge::LLM_KV_CACHE_NOT_EXIST, "cache does not exist, cache_id = %ld", cache_id);
    const auto &cache_desc = it->second;
    LLM_CHK_BOOL_RET_STATUS(tensor_index < static_cast<int32_t>(cache_desc.num_tensors),
                           ge::LLM_PARAM_INVALID,
                           "[cache_id:%ld][GetTensor] failed, tensor_index (%d) out of range [0, %u)",
                           cache_id, tensor_index, cache_desc.num_tensors);
  }
  const auto ret = llm_flow_service_->GetCacheTensors(cache_id, outputs, tensor_index);
  LLM_CHK_STATUS(ret, "[cache_id:%ld][GetTensor] failed", cache_id);
  LLM_CHK_STATUS_RET_NOLOG(ret);
  return ge::SUCCESS;
}

}  // namespace llm
