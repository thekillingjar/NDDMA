/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "swap_impl.h"
#include "common/llm_thread_pool.h"
#include "common/llm_blocking_queue.h"
#include "common/llm_log.h"
#include "common/llm_checker.h"
#include "common/llm_scope_guard.h"

namespace llm {
namespace {
constexpr int32_t kSwapIn = 0;
constexpr int32_t kSwapOut = 1;
constexpr int32_t kHbmBufferNum = 4;
constexpr int32_t kDefaultTimeout = 1;
constexpr uint32_t kDefaultMaxQueueSize = 10240U;
struct D2dArgs {
  void *src_addr;
  void *dst_addr;
  uint64_t block_size;
};

void D2dMemcpyThread(const aclrtContext &rt_context, const std::atomic_bool &d2d_thread_run_flag,
                     ge::BlockingQueue<D2dArgs> &d2d_args_queue, ge::BlockingQueue<void *> &hbm_buffers,
                     std::pair<std::atomic_int32_t, size_t> &status_info) {
  (void)pthread_setname_np(pthread_self(), "ge_llm_swap");
  auto &ret = status_info.first;
  ret = aclrtSetCurrentContext(rt_context);
  if (ret != ACL_ERROR_NONE) {
    LLMLOGE(ge::FAILED, "aclrtSetCurrentContext failed");
    return;
  }
  aclrtStream swap_in_stream = nullptr;
  ret = aclrtCreateStream(&swap_in_stream);
  if (ret != ACL_ERROR_NONE) {
    LLMLOGE(ge::FAILED, "aclrtCreateStream failed");
    return;
  }
  LLM_MAKE_GUARD(destroy_stream, [&swap_in_stream]() { LLM_CHK_ACL(aclrtDestroyStream(swap_in_stream)); });
  while (true) {
    if (!d2d_thread_run_flag && d2d_args_queue.Size() == 0) {
      break;
    }
    D2dArgs d2d_args;
    if (!d2d_args_queue.Pop(d2d_args)) {
      continue;
    }
    LLMLOGI("Begin aclrtMemcpyAsync, block size:%lu", d2d_args.block_size);
    const auto copy_start = std::chrono::steady_clock::now();
    ret = aclrtMemcpyAsync(d2d_args.dst_addr, d2d_args.block_size, d2d_args.src_addr, d2d_args.block_size,
                           ACL_MEMCPY_DEVICE_TO_DEVICE, swap_in_stream);
    if (ret != ACL_ERROR_NONE) {
      LLMLOGE(ge::FAILED, "aclrtMemcpyAsync failed");
      return;
    }
    ret = aclrtSynchronizeStream(swap_in_stream);
    if (ret != ACL_ERROR_NONE) {
      LLMLOGE(ge::FAILED, "aclrtSynchronizeStream failed");
      return;
    }
    const auto copy_end = std::chrono::steady_clock::now();
    status_info.second += std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count();
    (void)hbm_buffers.Push(d2d_args.src_addr);
  }
}

void StopSwapThread(ge::BlockingQueue<void *> &hbm_buffers, ge::BlockingQueue<D2dArgs> &d2d_args_queue,
                    std::atomic_bool &d2d_thread_run_flag, std::thread &d2d_thread) {
  d2d_thread_run_flag = false;
  if (d2d_thread.joinable()) {
    d2d_thread.join();
  }
  hbm_buffers.Stop();
  d2d_args_queue.Stop();
}

ge::Status AllCachesH2dCopy(const std::vector<uintptr_t> &src_addrs, const std::vector<uintptr_t> &dst_addrs,
                            const uint64_t block_size, const std::vector<std::pair<int64_t, int64_t>> &block_mapping,
                            ge::BlockingQueue<void *> &hbm_buffers) {
  aclrtContext rt_context = nullptr;
  LLM_CHK_ACL_RET(aclrtGetCurrentContext(&rt_context));
  std::pair<std::atomic_int32_t, size_t> status_info{ACL_ERROR_NONE, 0UL};
  std::atomic_bool d2d_thread_run_flag(true);
  ge::BlockingQueue<D2dArgs> d2d_args_queue(kDefaultMaxQueueSize);
  std::thread d2d_thread = std::thread(D2dMemcpyThread, rt_context, std::ref(d2d_thread_run_flag),
                                       std::ref(d2d_args_queue), std::ref(hbm_buffers), std::ref(status_info));
  LLM_MAKE_GUARD(stop_thread, ([&d2d_thread_run_flag, &d2d_thread, &hbm_buffers, &d2d_args_queue]() {
                  StopSwapThread(hbm_buffers, d2d_args_queue, d2d_thread_run_flag, d2d_thread);
                }));
  std::queue<std::pair<int64_t, int64_t>> block_indices;
  for (const auto &src_to_dst : block_mapping) {
    block_indices.push(src_to_dst);
  }
  std::atomic<size_t> h2d_copy_time{0UL};
  for (size_t i = 0U; i < src_addrs.size(); ++i) {
    auto src_addr = src_addrs[i];
    auto dst_addr = dst_addrs[i];
    std::queue<std::pair<int64_t, int64_t>> tmp_block_indices(block_indices);
    while (true) {
      if (tmp_block_indices.empty() || (status_info.first != ACL_ERROR_NONE)) {
        break;
      }
      void *hbm_addr = nullptr;
      if (!hbm_buffers.Pop(hbm_addr, kDefaultTimeout)) {
        continue;
      }
      const auto &block_index = tmp_block_indices.front();
      tmp_block_indices.pop();
      const uintptr_t src = src_addr + block_index.first * block_size;
      LLMLOGI("Begin aclrtMemcpy, src_index:%ld, dst_index:%ld", block_index.first, block_index.second);
      const auto copy_start = std::chrono::steady_clock::now();
      LLM_CHK_ACL_RET(
          aclrtMemcpy(hbm_addr, block_size, reinterpret_cast<void *>(src), block_size, ACL_MEMCPY_HOST_TO_DEVICE));
      const auto copy_end = std::chrono::steady_clock::now();
      const auto cost = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count();;
      h2d_copy_time.fetch_add(cost, std::memory_order_relaxed);
      const uintptr_t d2d_dst_addr = dst_addr + block_index.second * block_size;
      LLM_CHK_BOOL_RET_STATUS(
          d2d_args_queue.Push(D2dArgs{hbm_addr, reinterpret_cast<void *>(d2d_dst_addr), block_size}, false), ge::FAILED,
          "d2d queue push args failed");
    }
  }
  StopSwapThread(hbm_buffers, d2d_args_queue, d2d_thread_run_flag, d2d_thread);
  LLMLOGI("[LlmPerf] h2d aclrtMemcpy cost time:%zu us, d2d aclrtMemcpyAsync cost time:%zu us",
         h2d_copy_time.load(), status_info.second);
  LLM_CHK_BOOL_RET_STATUS(status_info.first == ACL_ERROR_NONE, ge::FAILED, "d2d memcpy failed");
  return ge::SUCCESS;
}
}  // namespace

ge::Status SwapImpl::SwapBlocks(const std::vector<uintptr_t> &src_addrs, const std::vector<uintptr_t> &dst_addrs,
                                const uint64_t block_size,
                                const std::vector<std::pair<int64_t, int64_t>> &block_mapping,
                                const CopyInfo &copy_info) {
  std::vector<std::vector<std::pair<int64_t, int64_t>>> ordered_block_mapping;
  LLM_CHK_STATUS_RET(LLMUtils::FindContiguousBlockIndexPair(block_mapping, ordered_block_mapping));
  const auto start = std::chrono::steady_clock::now();
  llm::LLMThreadPool swap_out_pool("ge_llm_swap", kHbmBufferNum);
  aclrtContext rt_context = nullptr;
  LLM_CHK_ACL_RET(aclrtGetCurrentContext(&rt_context));
  std::vector<std::future<ge::Status>> rets;
  std::atomic<size_t> rt_copy_time{0UL};
  for (size_t i = 0U; i < src_addrs.size(); ++i) {
    auto src_addr = src_addrs[i];
    auto dst_addr = dst_addrs[i];
    std::future<ge::Status> f = swap_out_pool.commit([src_addr, dst_addr, &ordered_block_mapping, &block_size,
                                                      &rt_context, &rt_copy_time, &copy_info]() -> ge::Status {
      LLM_CHK_ACL_RET(aclrtSetCurrentContext(rt_context));
      for (const auto &ordered_block : ordered_block_mapping) {
        const int64_t src_index = ordered_block.front().first;
        const int64_t dst_index = ordered_block.front().second;
        const uint64_t copy_size = block_size * ordered_block.size();
        auto src = src_addr + src_index * block_size;
        auto dst = dst_addr + dst_index * block_size;
        LLMLOGI("Begin mem copy, src index:%ld, dst index:%ld, copy size:%lu, contiguous block num:%lu", src_index,
               dst_index, copy_size, ordered_block.size());
        const auto copy_start = std::chrono::steady_clock::now();
        if (copy_info.copy_type == CopyType::kMemcpyEx) {
          LLM_CHK_ACL_RET(rtMemcpyEx(reinterpret_cast<void *>(dst), copy_size, reinterpret_cast<void *>(src), copy_size,
                                   copy_info.copy_kind));
        } else {
          LLM_CHK_ACL_RET(rtMemcpy(reinterpret_cast<void *>(dst), copy_size, reinterpret_cast<void *>(src), copy_size,
                                 copy_info.copy_kind));
        }
        const auto copy_end = std::chrono::steady_clock::now();
        const auto cost = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count();
        rt_copy_time.fetch_add(cost, std::memory_order_relaxed);
      }
      return ge::SUCCESS;
    });
    LLM_CHK_BOOL_RET_STATUS(f.valid(), ge::FAILED, "commit blocks rtMemcpy task failed");
    rets.emplace_back(std::move(f));
  }
  for (size_t i = 0U; i < rets.size(); ++i) {
    LLM_CHK_BOOL_RET_STATUS(rets[i].get() == ge::SUCCESS, ge::FAILED, "the %zuth blocks mem copy failed", i);
  }
  const auto end = std::chrono::steady_clock::now();
  LLMLOGI("[LlmPerf] mem copy cost time:%zu us, copy kind:%d, swap blocks cost time:%zu us", rt_copy_time.load(),
         copy_info.copy_kind, std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  return ge::SUCCESS;
}

ge::Status SwapImpl::SwapInBlocks(const std::vector<uintptr_t> &src_addrs, const std::vector<uintptr_t> &dst_addrs,
                                  const uint64_t block_size,
                                  const std::vector<std::pair<int64_t, int64_t>> &block_mapping) {
  const auto start = std::chrono::steady_clock::now();
  ge::BlockingQueue<void *> hbm_buffers;
  aclError ret = ACL_ERROR_NONE;
  std::vector<void *> need_freed_buffers;
  for (int32_t i = 0; i < kHbmBufferNum; ++i) {
    void *addr = nullptr;
    ret = aclrtMalloc(&addr, block_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    if (ret != ACL_ERROR_NONE) {
      LLMLOGE(ge::FAILED, "aclrtMalloc failed");
      break;
    }
    need_freed_buffers.emplace_back(addr);
    (void)hbm_buffers.Push(addr);
  }
  const auto malloc_end = std::chrono::steady_clock::now();
  LLMLOGI("Malloc hbm buffer success, num:%d, per buffer size:%lu", need_freed_buffers.size(), block_size);
  LLM_MAKE_GUARD(free_buffers, [&need_freed_buffers]() {
    for (const auto &buffer : need_freed_buffers) {
      (void)aclrtFree(buffer);
    }
  });
  LLM_CHK_BOOL_RET_STATUS(ret == ACL_ERROR_NONE, ge::LLM_DEVICE_OUT_OF_MEMORY, "aclrtMalloc hbm buffer failed");
  LLM_CHK_STATUS_RET(AllCachesH2dCopy(src_addrs, dst_addrs, block_size, block_mapping, hbm_buffers),
                    "h2d memcpy failed");
  const auto end = std::chrono::steady_clock::now();
  LLMLOGI("[LlmPerf] malloc hbm buffer cost time:%zu us, swap in blocks cost time:%zu us",
         std::chrono::duration_cast<std::chrono::microseconds>(malloc_end - start).count(),
         std::chrono::duration_cast<std::chrono::microseconds>(end - malloc_end).count());
  return ge::SUCCESS;
}

ge::Status SwapImpl::SwapOutBlocks(const std::vector<uintptr_t> &src_addrs, const std::vector<uintptr_t> &dst_addrs,
                                   const uint64_t block_size,
                                   const std::vector<std::pair<int64_t, int64_t>> &block_mapping) {
  return SwapBlocks(src_addrs, dst_addrs, block_size, block_mapping,
                    CopyInfo{CopyType::kMemcpyEx, RT_MEMCPY_DEVICE_TO_HOST});
}

ge::Status SwapImpl::SwapBlocks(const Cache &src, const Cache &dst, const uint64_t block_size, const uint32_t type,
                                const std::vector<std::pair<int64_t, int64_t>> &block_mapping,
                                size_t device_index) const {
  // validate per_device_tensor_addrs has device_index size before in llm_datadist.cc.
  const auto &src_addrs = src.per_device_tensor_addrs;
  const auto &dst_addrs = dst.per_device_tensor_addrs;
  LLM_CHK_BOOL_RET_STATUS((src_addrs[device_index].size() == dst_addrs[device_index].size()), ge::LLM_PARAM_INVALID,
                         "src adrrs size:%zu not equal dst addrs size:%zu", src_addrs[device_index].size(),
                         dst_addrs[device_index].size());
  LLMLOGI("Begin swap blocks, cache num:%zu, swap block num:%zu", src_addrs.front().size(), block_mapping.size());
  LLM_CHK_ACL_RET(aclrtSetDevice(device_id_));
  LLM_MAKE_GUARD(reset_device, [this]() { LLM_CHK_ACL(aclrtResetDevice(device_id_)); });
  if (type == kSwapOut) {
    LLM_CHK_STATUS_RET(SwapOutBlocks(src_addrs[device_index], dst_addrs[device_index], block_size, block_mapping),
                      "swap out blocks failed");
  } else {
    LLM_CHK_STATUS_RET(SwapInBlocks(src_addrs[device_index], dst_addrs[device_index], block_size, block_mapping));
  }
  return ge::SUCCESS;
}
}  // namespace llm
