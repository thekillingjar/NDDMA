/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cache_manager.h"
#include <numeric>
#include "acl/acl.h"
#include "common/llm_utils.h"
#include "common/llm_thread_pool.h"
#include "common/def_types.h"
// for using rtMemcpyEx with mbuf
#include "runtime/rt_external.h"

namespace llm {
namespace {
constexpr uint64_t kMaxBlockSize = 4UL * 1024 * 1024 * 1024; // 4GB

class CopyJob {
 public:
  explicit CopyJob(aclrtStream stream, bool mbuf_involved = false, uint64_t max_block_size = kMaxBlockSize)
      : stream_(stream), mbuf_involved_(mbuf_involved), max_block_size_(max_block_size) {
  }
  ~CopyJob() = default;

  ge::Status AddCopyTask(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) {
    if (!NeedCopyAsync(kind, count)) {
      if (rt_context_ == nullptr) {
        LLM_CHK_BOOL_RET_STATUS(aclrtGetCurrentContext(&rt_context_) == ACL_ERROR_NONE, ge::FAILED, "Failed to get aclrt context");
      }
      auto fut = pool_.commit([this, dst, dest_max, src, count, kind]() -> ge::Status {
        (void) aclrtSetCurrentContext(rt_context_);
        const auto ret = mbuf_involved_ ? rtMemcpyEx(dst, dest_max, src, count, kind) :
                         rtMemcpy(dst, dest_max, src, count, kind);
        LLM_CHK_BOOL_RET_STATUS(ret == RT_ERROR_NONE, ge::FAILED,
                               "failed to copy cache, rt_ret = 0x%X", static_cast<uint32_t>(ret));
        return ge::SUCCESS;
      });
      LLM_CHK_BOOL_RET_STATUS(fut.valid(), ge::FAILED, "Failed to commit copy task");
      copy_futs_.emplace_back(std::move(fut));
      return ge::SUCCESS;
    }
    need_sync_ = true;
    uint64_t offset = 0;
    uint64_t remaining = count;
    while (remaining > 0) {
      uint64_t size_to_copy = remaining <= max_block_size_ ? remaining : max_block_size_;
      auto dst_start = static_cast<uint8_t *>(dst) + offset;
      auto src_start = static_cast<const uint8_t *>(src) + offset;
      LLM_CHK_ACL_RET(aclrtMemcpyAsync(dst_start,
                                      dest_max,
                                      src_start,
                                      size_to_copy,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE,
                                      stream_));
      offset += max_block_size_;
      remaining -= size_to_copy;
    }
    return ge::SUCCESS;
  }

  ge::Status GetResult() {
    if (need_sync_) {
      LLM_CHK_STATUS_RET(aclrtSynchronizeStream(stream_));
    }
    for (size_t i = 0U; i < copy_futs_.size(); ++i) {
      LLM_CHK_STATUS_RET(copy_futs_[i].get(), "Failed to copy cache, index = %zu", i);
    }
    return ge::SUCCESS;
  }

 private:
  bool NeedCopyAsync(rtMemcpyKind_t kind, uint64_t count) const {
    constexpr uint64_t kMinBlockSize = 2UL * 1024 * 1024; // 2MB
    return (kind == RT_MEMCPY_DEVICE_TO_DEVICE) && (count >= kMinBlockSize || mbuf_involved_);
  }

  aclrtStream stream_ = nullptr;
  aclrtContext rt_context_ = nullptr;
  bool need_sync_ = false;
  bool mbuf_involved_ = false;
  uint64_t max_block_size_ = 0;
  llm::LLMThreadPool pool_{"ge_llm_cahm", 4};
  std::vector<std::future<ge::Status>> copy_futs_;
};
}  // namespace

ge::Status CacheManager::CopyCacheForContinuous(const CacheEntry &src_cache_entry,
                                                const CacheEntry &dst_cache_entry,
                                                const CopyCacheParam &copy_cache_param,
                                                size_t per_device_addr_num,
                                                size_t device_index) {
  const auto src_id = copy_cache_param.src_cache_id;
  const auto dst_id = copy_cache_param.dst_cache_id;
  uint64_t copy_size;
  LLM_CHK_STATUS_RET(CheckCopyParams(src_cache_entry, dst_cache_entry, copy_cache_param, copy_size),
                    "[Copy][%ld->%ld] check param failed", src_id, dst_id);
  auto src_offset = src_cache_entry.stride * copy_cache_param.src_batch_index + copy_cache_param.offset;
  auto dst_offset = dst_cache_entry.stride * copy_cache_param.dst_batch_index + copy_cache_param.offset;
  LLMLOGI("[Copy][%ld->%ld] start", src_id, dst_id);
  EnsureCopyStream(device_index);
  const auto copy_kind = ResolveCopyKind(src_cache_entry.placement, dst_cache_entry.placement);
  CopyJob copy_job(copy_streams_[device_index], copy_cache_param.mbuf_involved);
  auto begin = device_index * per_device_addr_num;
  for (size_t i = 0U; i < per_device_addr_num; ++i) {
    auto src_addr = llm::PtrToPtr<void, uint8_t>(src_cache_entry.cache_addrs[begin + i].get()) + src_offset;
    auto dst_max = dst_cache_entry.stride - copy_cache_param.offset;
    auto dst_addr = llm::PtrToPtr<void, uint8_t>(dst_cache_entry.cache_addrs[begin + i].get()) + dst_offset;
    LLM_CHK_STATUS_RET(copy_job.AddCopyTask(dst_addr, dst_max, src_addr, copy_size, copy_kind),
                      "[Copy][%ld->%ld] invoke rtMemcpy failed, index = %zu",
                      src_id,
                      dst_id,
                      i);
  }
  LLM_CHK_STATUS_RET(copy_job.GetResult(), "[Copy][%ld->%ld] invoke aclrtSynchronizeStream failed", src_id, dst_id);
  LLMLOGI("[Copy][%ld->%ld] success, num_tensors = %zu, src_batch_index = %u, "
         "dst_batch_index = %u, offset = %lu, size = %ld",
         src_id, dst_id, per_device_addr_num, copy_cache_param.src_batch_index,
         copy_cache_param.dst_batch_index, copy_cache_param.offset, copy_cache_param.size);
  return ge::SUCCESS;
}

ge::Status CacheManager::CopyCacheForBlocks(const CacheEntry &src_cache_entry,
                                            const CacheEntry &dst_cache_entry,
                                            const CopyCacheParam &copy_cache_param,
                                            size_t per_device_addr_num,
                                            size_t device_index) {
  const auto src_id = copy_cache_param.src_cache_id;
  const auto dst_id = copy_cache_param.dst_cache_id;
  LLM_CHK_BOOL_RET_STATUS(src_cache_entry.stride == dst_cache_entry.stride, ge::FAILED,
                         "[Copy][%ld->%ld] failed, block_size mismatches, src = %lu, dst = %lu",
                         src_id, dst_id, src_cache_entry.stride, dst_cache_entry.stride);
  const auto block_size = src_cache_entry.stride;
  LLMLOGI("[Copy][%ld->%ld] start", src_id, dst_id);
  EnsureCopyStream(device_index);
  CopyJob copy_job(copy_streams_[device_index], copy_cache_param.mbuf_involved);
  const auto copy_kind = ResolveCopyKind(src_cache_entry.placement, dst_cache_entry.placement);
  auto begin = device_index * per_device_addr_num;
  for (size_t i = 0U; i < per_device_addr_num; ++i) {
    auto src_addr_base = llm::PtrToPtr<void, uint8_t>(src_cache_entry.cache_addrs[begin + i].get());
    auto dst_addr_base = llm::PtrToPtr<void, uint8_t>(dst_cache_entry.cache_addrs[begin + i].get());
    for (auto &block_index_pair : copy_cache_param.copy_block_infos) {
      const auto src_block_index = block_index_pair.first;
      const auto dst_block_index = block_index_pair.second;
      LLM_CHK_BOOL_RET_STATUS(src_block_index < src_cache_entry.num_blocks, ge::LLM_PARAM_INVALID,
                             "src_block_index:%lu out of range [0, %lu)", src_block_index, src_cache_entry.num_blocks);
      LLM_CHK_BOOL_RET_STATUS(dst_block_index < dst_cache_entry.num_blocks, ge::LLM_PARAM_INVALID,
                             "dst_block_index:%lu out of range [0, %lu)", dst_block_index, src_cache_entry.num_blocks);
      auto src_addr = src_addr_base + block_size * src_block_index;
      auto dst_addr = dst_addr_base + block_size * dst_block_index;
      LLM_CHK_STATUS_RET(copy_job.AddCopyTask(dst_addr, block_size, src_addr, block_size, copy_kind),
                        "[Copy][%ld->%ld] invoke rtMemcpy failed, index = %zu", src_id, dst_id, i);
    }
  }
  LLM_CHK_STATUS_RET(copy_job.GetResult(), "[Copy][%ld->%ld] invoke aclrtSynchronizeStream failed", src_id, dst_id);
  LLMLOGI("[Copy][%ld->%ld] success, num_tensors = %zu, num_blocks = %zu",
         src_id, dst_id, per_device_addr_num, copy_cache_param.copy_block_infos.size());
  return ge::SUCCESS;
}

ge::Status CacheManager::CheckCopyParams(const CacheEntry &src_cache_entry,
                                         const CacheEntry &dst_cache_entry,
                                         const CopyCacheParam &copy_cache_param,
                                         uint64_t &copy_size) {
  const auto src_batch_size = src_cache_entry.batch_size;
  LLM_CHK_BOOL_RET_STATUS(copy_cache_param.src_batch_index < src_batch_size,
                         ge::LLM_PARAM_INVALID,
                         "src_batch_index (%u) out of range [0, %u)",
                         copy_cache_param.src_batch_index, src_batch_size);
  const auto dst_batch_size = dst_cache_entry.batch_size;
  LLM_CHK_BOOL_RET_STATUS(copy_cache_param.dst_batch_index < dst_batch_size,
                         ge::LLM_PARAM_INVALID,
                         "dst_batch_index (%u) out of range [0, %u)",
                         copy_cache_param.dst_batch_index, dst_batch_size);
  if (copy_cache_param.offset > 0) {
    LLM_CHK_BOOL_RET_STATUS(copy_cache_param.offset < src_cache_entry.stride,
                           ge::LLM_PARAM_INVALID,
                           "offset out of range of src cache tensor, offset = %lu, tensor stride = %lu",
                           copy_cache_param.offset, src_cache_entry.stride);
    LLM_CHK_BOOL_RET_STATUS(copy_cache_param.offset < dst_cache_entry.stride,
                           ge::LLM_PARAM_INVALID,
                           "offset out of range of dst cache tensor, offset = %lu, tensor stride = %lu",
                           copy_cache_param.offset, dst_cache_entry.stride);
  }
  copy_size = copy_cache_param.size != -1 ? static_cast<uint64_t>(copy_cache_param.size)
                                          : src_cache_entry.stride - copy_cache_param.offset;
  LLM_CHK_BOOL_RET_STATUS(copy_size <= (src_cache_entry.stride - copy_cache_param.offset),
                         ge::LLM_PARAM_INVALID,
                         "src_tensor out of range, offset = %lu, copy_size = %ld, tensor stride = %lu",
                         copy_cache_param.offset, copy_cache_param.size, src_cache_entry.stride);
  LLM_CHK_BOOL_RET_STATUS(copy_size <= (dst_cache_entry.stride - copy_cache_param.offset),
                         ge::LLM_PARAM_INVALID,
                         "dst_tensor out of range, offset = %lu, copy_size = %ld, tensor stride = %lu, "
                         "src tensor stride = %lu",
                         copy_cache_param.offset, copy_cache_param.size, dst_cache_entry.stride,
                         src_cache_entry.stride);
  return ge::SUCCESS;
}

rtMemcpyKind_t CacheManager::ResolveCopyKind(CachePlacement src_placement, CachePlacement dst_placement) {
  return (src_placement == CachePlacement::HOST) ?
         ((dst_placement == CachePlacement::HOST) ? RT_MEMCPY_HOST_TO_HOST : RT_MEMCPY_HOST_TO_DEVICE) :
         ((dst_placement == CachePlacement::HOST) ? RT_MEMCPY_DEVICE_TO_HOST : RT_MEMCPY_DEVICE_TO_DEVICE);
}

void CacheManager::DestroyCopyStream(size_t device_index) {
  if (device_index < copy_streams_.size() && (copy_streams_[device_index] != nullptr)) {
    LLM_CHK_ACL(aclrtDestroyStream(copy_streams_[device_index]));
    copy_streams_[device_index] = nullptr;
  }
}

ge::Status CacheManager::EnsureCopyStream(size_t device_index) {
  std::lock_guard<std::mutex> lk(copy_mu_);
  if (copy_streams_[device_index] == nullptr) {
    LLM_CHK_ACL_RET(aclrtCreateStream(&copy_streams_[device_index]));
  }
  return ge::SUCCESS;
}

ge::Status CacheManager::InitCopyStreams(size_t device_num) {
  copy_streams_.resize(device_num);
  return ge::SUCCESS;
}
}  // namespace llm
