/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_datadist.h"
#include "common/mem_utils.h"
#include "common/llm_flow_service.h"
#include "common/cache_engine.h"
#include "common/llm_common.h"
#include "common/llm_checker.h"
#include "common/llm_scope_guard.h"

namespace llm {
class LLMDataDist::Impl {
 public:
  explicit Impl(const uint64_t cluster_id) : cluster_id_(cluster_id) {}
  ~Impl() {
    if (is_initialized_) {
      LLMDataDistFinalize();
    }
  }

  ge::Status LLMDataDistInitialize(const std::map<ge::AscendString, ge::AscendString> &options);

  void LLMDataDistFinalize();

  bool IsInitialized() const;

  ge::Status CheckLinkStatus(uint64_t remote_cluster_id);

  ge::Status LinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                          const int32_t timeout);

  ge::Status UnlinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                            const int32_t timeout, bool force_flag = false);

  ge::Status AllocateCache(const CacheDesc &cache_desc,
                           Cache &cache,
                           const std::vector<CacheKey> &cache_keys);

  ge::Status DeallocateCache(int64_t cache_id);

  ge::Status RemoveCacheKey(const CacheKey &cache_key);

  ge::Status PullCache(int64_t cache_id, const CacheKey &cache_key, const PullCacheParam &pull_cache_param);

  ge::Status PullBlocks(int64_t cache_id, const CacheKey &cache_key, const PullCacheParam &pull_cache_param);

  ge::Status TransferCache(uint64_t task_id, const TransferCacheConfig &transfer_cache_config,
                           const TransferBlockConfig &transfer_block_config);

  ge::Status CopyCache(const CacheEntry &src_cache_entry, const CacheEntry &dst_cache_entry,
                       const CopyCacheParam &copy_cache_param, const std::vector<int32_t> &device_ids);

  ge::Status CopyCache(const CopyCacheParam &copy_cache_param);

  ge::Status GetCacheTensors(int64_t cache_id, std::vector<ge::Tensor> &outputs, int32_t tensor_index);

  ge::Status SwitchRole(const std::string &role, const std::map<std::string, std::string> &options);

  ge::Status SwapBlocks(const Cache &src, const Cache &dst, const uint64_t block_size, const uint32_t type,
                        const std::vector<std::pair<int64_t, int64_t>> &block_mapping);

 private:
  ge::Status InitFlowService(const std::map<ge::AscendString, ge::AscendString> &options);

  uint64_t cluster_id_;
  std::string role_;
  bool enable_switch_role_ = false;
  std::atomic<bool> is_initialized_{false};
  std::vector<int32_t> device_ids_;

  uint64_t model_id_ = 0U;
  std::mutex mutex_;
  std::unique_ptr<LlmFlowService> llm_flow_service_;
  std::unique_ptr<CacheEngine> cache_engine_;
};

ge::Status LLMDataDist::Impl::LLMDataDistInitialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  LLMLOGI("LLMDataDist initialize cluster id:%lu", cluster_id_);
  std::lock_guard<std::mutex> lk(mutex_);
  if (!is_initialized_) {
    auto iter = options.find(LLM_OPTION_ROLE);
    LLM_ASSERT_TRUE(iter != options.end(), "[LLMDataDist] option:%s not found.", LLM_OPTION_ROLE);
    role_ = iter->second.GetString();
    LLMLOGI("LLMDataDist role:%s.", role_.c_str());
    LLM_ASSERT_TRUE((role_ == kDecoder) || (role_ == kPrompt) || (role_ == kMix), "Invalid role: %s", role_.c_str());
    const auto ret = InitFlowService(options);
    if (ret != ge::SUCCESS) {
      if (llm_flow_service_ != nullptr) {
        llm_flow_service_->Finalize();
      }
      LLMLOGE(ret, "[Manager] initialize failed.");
      return ret;
    }
    is_initialized_ = true;
    LLMLOGI("[LLMDataDist] role:[%s] cluster:[%lu] init success", role_.c_str(), cluster_id_);
  }
  return ge::SUCCESS;
}

void LLMDataDist::Impl::LLMDataDistFinalize() {
  std::lock_guard<std::mutex> lk(mutex_);
  if (!is_initialized_.load()) {
    LLMLOGW("[LLMDataDist] cluster:%lu is not initialized", cluster_id_);
    return;
  }
  llm_flow_service_->Finalize();
  is_initialized_.store(false);
  LLMLOGI("LLMDataDist Finalize end.");
}


bool LLMDataDist::Impl::IsInitialized() const {
  return is_initialized_.load(std::memory_order::memory_order_relaxed);
}

ge::Status LLMDataDist::Impl::CheckLinkStatus(uint64_t remote_cluster_id){
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(), ge::FAILED, "Llm datadist of cluster:%lu is not initialized.",
                         cluster_id_);
  LLM_CHK_BOOL_RET_STATUS(role_ == kDecoder, ge::LLM_PARAM_INVALID,
                         "[LLMDataDist] CheckLinkStatus can only be used by Decoder, current_role is %s, ",
                         role_.c_str());
  LLM_CHK_STATUS_RET(llm_flow_service_->CheckLinkStatus(remote_cluster_id));
  return ge::SUCCESS;
}

ge::Status LLMDataDist::Impl::LinkClusters(const std::vector<ClusterInfo> &clusters,
                                           std::vector<ge::Status> &rets, const int32_t timeout) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(), ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  LLM_CHK_BOOL_RET_STATUS(role_ == kDecoder,
                         ge::LLM_PARAM_INVALID,
                         "[LLMDataDist] linkClusters can only be used by Decoder, current_role is %s, ", role_.c_str());
  rets.clear();
  LLM_CHK_STATUS_RET(llm_flow_service_->LinkClusters(clusters, rets, timeout));
  return ge::SUCCESS;
}

ge::Status LLMDataDist::Impl::UnlinkClusters(const std::vector<ClusterInfo> &clusters,
                                             std::vector<ge::Status> &rets, const int32_t timeout, bool force_flag) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  rets.clear();
  LLM_CHK_STATUS_RET(llm_flow_service_->UnlinkClusters(clusters, rets, timeout, force_flag), "Unlink clusters failed");
  return ge::SUCCESS;
}

ge::Status LLMDataDist::Impl::AllocateCache(const CacheDesc &cache_desc,
                                            Cache &cache,
                                            const std::vector<CacheKey> &cache_keys) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  return cache_engine_->Allocate(cache_desc, cache_keys, cache);
}

ge::Status LLMDataDist::Impl::DeallocateCache(int64_t cache_id) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  return cache_engine_->Deallocate(cache_id);
}

ge::Status LLMDataDist::Impl::PullCache(int64_t cache_id,
                                        const CacheKey &cache_key,
                                        const PullCacheParam &pull_cache_param) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  LLM_CHK_BOOL_RET_STATUS(role_ == kDecoder, ge::LLM_PARAM_INVALID,
                         "[PullKvCache] is only supported by Decoder, current_role is %s.",
                         role_.c_str());
  return cache_engine_->PullCache(cache_id, cache_key, pull_cache_param);
}

ge::Status LLMDataDist::Impl::PullBlocks(int64_t cache_id,
                                         const CacheKey &cache_key,
                                         const PullCacheParam &pull_cache_param) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  LLM_CHK_BOOL_RET_STATUS(role_ == kDecoder, ge::LLM_PARAM_INVALID,
                         "[PullKvBlocks] is only supported by Decoder, current_role is %s.",
                         role_.c_str());
  LLM_CHK_BOOL_RET_STATUS((!pull_cache_param.prompt_blocks.empty()),
                         ge::LLM_PARAM_INVALID,
                         "src_blocks is empty, pull from non-block cache is not supported yet");
  LLM_CHK_BOOL_RET_STATUS(pull_cache_param.prompt_blocks.size() == pull_cache_param.decoder_blocks.size(),
                         ge::LLM_PARAM_INVALID,
                         "number of src_blocks (%zu) mismatches that of dst_blocks (%zu)",
                         pull_cache_param.prompt_blocks.size(), pull_cache_param.decoder_blocks.size());
  LLM_CHK_BOOL_RET_STATUS(cache_key.prompt_batch_index == 0U,
                         ge::LLM_PARAM_INVALID,
                         "invalid cache_key.prompt_batch_index (%lu), only 0 is supported in pull block",
                         cache_key.prompt_batch_index);
  return cache_engine_->PullCache(cache_id, cache_key, pull_cache_param);
}

ge::Status LLMDataDist::Impl::TransferCache(uint64_t task_id, const TransferCacheConfig &transfer_cache_config,
                                            const TransferBlockConfig &transfer_block_config) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  LLM_CHK_BOOL_RET_STATUS(device_ids_.size() <= 1U, ge::LLM_FEATURE_NOT_ENABLED,
                        "Transfer cache is not supported when device num bigger than one.");
  LLM_CHK_BOOL_RET_STATUS(role_ == kPrompt, ge::LLM_PARAM_INVALID,
                         "Transfer cache is not supported by %s", role_.c_str());
  return cache_engine_->TransferCache(task_id, transfer_cache_config, transfer_block_config);
}

ge::Status LLMDataDist::Impl::CopyCache(const CopyCacheParam &copy_cache_param) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  return cache_engine_->CopyCache(copy_cache_param);
}

ge::Status LLMDataDist::Impl::CopyCache(const CacheEntry &src_cache_entry,
                                        const CacheEntry &dst_cache_entry,
                                        const CopyCacheParam &copy_cache_param,
                                        const std::vector<int32_t> &device_ids) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  LLM_CHK_BOOL_RET_STATUS(src_cache_entry.cache_addrs.size() % device_ids_.size() == 0, ge::LLM_PARAM_INVALID,
                         "Param invalid, tensor_addrs size:%zu of src cache should be a multiple of device num:%zu.",
                         src_cache_entry.cache_addrs.size(), device_ids_.size());
  LLM_CHK_BOOL_RET_STATUS(dst_cache_entry.cache_addrs.size() % device_ids_.size() == 0, ge::LLM_PARAM_INVALID,
                         "Param invalid, tensor_addrs size:%zu of dst cache should be a multiple of device num:%zu.",
                         dst_cache_entry.cache_addrs.size(), device_ids_.size());
  return llm_flow_service_->CopyCache(src_cache_entry, dst_cache_entry, copy_cache_param, device_ids);
}

ge::Status LLMDataDist::Impl::RemoveCacheKey(const CacheKey &cache_key) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  return cache_engine_->RemoveCacheKey(cache_key);
}

ge::Status LLMDataDist::Impl::GetCacheTensors(int64_t cache_id,
                                              std::vector<ge::Tensor> &outputs,
                                              int32_t tensor_index) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed),
                         ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  return cache_engine_->GetCacheTensors(cache_id, outputs, tensor_index);
}

ge::Status LLMDataDist::Impl::InitFlowService(const std::map<ge::AscendString, ge::AscendString> &options) {
  LLM_CHK_STATUS_RET(LLMUtils::ParseFlag(LLM_OPTION_ENABLE_SWITCH_ROLE, options, enable_switch_role_));
  llm_flow_service_ = llm::MakeUnique<LlmFlowService>(role_, enable_switch_role_, cluster_id_);
  LLM_CHECK_NOTNULL(llm_flow_service_);
  LLM_CHK_STATUS_RET(LLMUtils::ParseDeviceId(options, device_ids_, ge::OPTION_EXEC_DEVICE_ID),
                    "Failed to get device id");
  if (device_ids_.empty()) {
    device_ids_.emplace_back(0U);
  }
  LLM_CHK_STATUS_RET(llm_flow_service_->Initialize(options, device_ids_), "Failed to initialize LLM flow service");

  cache_engine_ = llm::MakeUnique<CacheEngine>(llm_flow_service_.get());
  LLM_CHECK_NOTNULL(cache_engine_);

  LLMLOGI("Initialize LLM flow service successfully");
  return ge::SUCCESS;
}

ge::Status LLMDataDist::Impl::SwitchRole(const std::string &role, const std::map<std::string, std::string> &options) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(), ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  LLM_CHK_BOOL_RET_STATUS(enable_switch_role_,
                         ge::LLM_FEATURE_NOT_ENABLED,
                         "Option %s not enabled",
                         LLM_OPTION_ENABLE_SWITCH_ROLE);
  LLM_CHK_BOOL_RET_STATUS(device_ids_.size() <= 1U, ge::LLM_FEATURE_NOT_ENABLED,
                         "Switch role is not supported when device num bigger than one.");
  LLM_CHK_STATUS_RET(llm_flow_service_->SwitchRole(role, options), "Failed to switch role");
  LLMEVENT("role switched from %s to %s", role_.c_str(), role.c_str());
  role_ = role;
  return ge::SUCCESS;
}

ge::Status LLMDataDist::Impl::SwapBlocks(const Cache &src, const Cache &dst, const uint64_t block_size,
                                         const uint32_t type,
                                         const std::vector<std::pair<int64_t, int64_t>> &block_mapping) {
  LLM_CHK_BOOL_RET_STATUS(is_initialized_.load(std::memory_order::memory_order_relaxed), ge::FAILED,
                         "Llm datadist of cluster:%lu is not initialized.", cluster_id_);
  LLM_CHK_BOOL_RET_STATUS(src.per_device_tensor_addrs.size() == device_ids_.size(), ge::LLM_PARAM_INVALID,
                         "Param invalid, per_device_tensor_addrs size:%zu of src cache should be equal to device num:%zu.",
                         src.per_device_tensor_addrs.size(), device_ids_.size());
  LLM_CHK_BOOL_RET_STATUS(dst.per_device_tensor_addrs.size() == device_ids_.size(), ge::LLM_PARAM_INVALID,
                         "Param invalid, per_device_tensor_addrs size:%zu of dst cache should be equal to device num:%zu.",
                         dst.per_device_tensor_addrs.size(), device_ids_.size());
  SwapBlockParam swap_param{block_size, type, block_mapping};
  return llm_flow_service_->SwapBlocks(src, dst, swap_param, device_ids_);
}

// LLMDataDist start
LLMDataDist::LLMDataDist(const uint64_t cluster_id) {
  LLMLOGD("Enter LLMDataDist construction.");
  impl_ = llm::MakeUnique<Impl>(cluster_id);
  if (impl_ == nullptr) {
    LLMLOGW("Create LlmEngineImpl failed");
  }
}

LLMDataDist::~LLMDataDist() = default;

ge::Status LLMDataDist::LLMDataDistInitialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes((impl_->LLMDataDistInitialize(options)));
}

void LLMDataDist::LLMDataDistFinalize() {
  if (impl_ != nullptr) {
    impl_->LLMDataDistFinalize();
  }
}

ge::Status LLMDataDist::CheckLinkStatus(uint64_t remote_cluster_id){
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes((impl_->CheckLinkStatus(remote_cluster_id)));
}

ge::Status LLMDataDist::LinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                                     const int32_t timeout) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes((impl_->LinkClusters(clusters, rets, timeout)));
}

ge::Status LLMDataDist::UnlinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                                       const int32_t timeout, bool force_flag) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes((impl_->UnlinkClusters(clusters, rets, timeout, force_flag)));
}

ge::Status LLMDataDist::AllocateCache(const CacheDesc &cache_desc,
                                      Cache &cache,
                                      const std::vector<CacheKey> &cache_keys) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->AllocateCache(cache_desc, cache, cache_keys));
}

ge::Status LLMDataDist::DeallocateCache(int64_t cache_id) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->DeallocateCache(cache_id));
}

ge::Status LLMDataDist::PullCache(int64_t cache_id, const CacheKey &cache_key, const PullCacheParam &pull_cache_param) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->PullCache(cache_id, cache_key, pull_cache_param));
}

ge::Status LLMDataDist::PullBlocks(int64_t cache_id, const CacheKey &cache_key, const PullCacheParam &pull_cache_param) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->PullBlocks(cache_id, cache_key, pull_cache_param));
}

ge::Status LLMDataDist::TransferCache(uint64_t task_id, const TransferCacheConfig &transfer_cache_config,
                                      const TransferBlockConfig &transfer_block_config) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->TransferCache(task_id, transfer_cache_config, transfer_block_config));
}

ge::Status LLMDataDist::CopyCache(const CacheEntry &src_cache_entry, const CacheEntry &dst_cache_entry,
                                  const CopyCacheParam &copy_cache_param, const std::vector<int32_t> &device_ids) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->CopyCache(src_cache_entry, dst_cache_entry, copy_cache_param, device_ids));
}

ge::Status LLMDataDist::CopyCache(const CopyCacheParam &copy_cache_param) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->CopyCache(copy_cache_param));
}

ge::Status LLMDataDist::RemoveCacheKey(const CacheKey &cache_key) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->RemoveCacheKey(cache_key));
}

ge::Status LLMDataDist::GetCacheTensors(int64_t cache_id,
                                        std::vector<ge::Tensor> &outputs,
                                        int32_t tensor_index) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->GetCacheTensors(cache_id, outputs, tensor_index));
}

ge::Status LLMDataDist::SwitchRole(const std::string &role, const std::map<std::string, std::string> &options) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->SwitchRole(role, options));
}

ge::Status LLMDataDist::SwapBlocks(const Cache &src, const Cache &dst, const uint64_t block_size, const uint32_t type,
                                   const std::vector<std::pair<int64_t, int64_t>> &block_mapping) {
  LLM_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::LLM_PARAM_INVALID, "impl is nullptr, check LLMDataDist construct");
  return TransRetToLlmCodes(impl_->SwapBlocks(src, dst, block_size, type, block_mapping));
}

bool LLMDataDist::IsInitialized() const {
  return impl_ != nullptr ? impl_->IsInitialized() : false;
}
}  // namespace llm
