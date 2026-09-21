/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>

#include "graph/utils/math_util.h"
#include "base/err_msg.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/manager/mem_manager.h"
#include "caching_mem_allocator.h"
#include "rts_caching_mem_allocator.h"

#include "runtime/rt.h"

#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "utils/utils.h"

namespace gert {
namespace memory {
std::map<int32_t, std::map<rtMemType_t, std::shared_ptr<RtsCachingMemAllocator>>>
    RtsCachingMemAllocator::device_id_to_allocators_ = {};
std::mutex RtsCachingMemAllocator::mutex_;

RtsCachingMemAllocator::RtsCachingMemAllocator(const uint32_t device_id, const rtMemType_t memory_type)
    : ScalableAllocator(span_allocator_, rt_mem_allocator_), rt_mem_allocator_(*this, device_id, memory_type) {
  allocator_id_with_type_ = "[rts_allocator_" + std::to_string(allocator_id_) + "]";
}

RtsCachingMemAllocator::~RtsCachingMemAllocator() {
  // Free all memory before rt_mem_allocator_ destruction
  ScalableAllocator::Finalize(true);
}

void RtsCachingMemAllocator::Free(ge::MemBlock *block) {
  ScalableAllocator::Free(dynamic_cast<PageSpan *>(block));
}

PageSpan *RtsCachingMemAllocator::FetchNewSpan(ge::Allocator &allocator, const MemSize size, const PageLen page_len) {
  const MemSize huge_page_mem_size = ScalableAllocator::config_.huge_page_mem_size;
  if (size >= huge_page_mem_size) {
    return ScalableAllocator::FetchNewSpan(allocator, size, page_len);
  }
  if (IsThresholdExceeded(huge_page_mem_size)) {
    GELOGI("OccupiedSize:%llu add size:%llu exceed total_thresold:%llu.",
                device_allocator_.GetOccupiedSize(), huge_page_mem_size, config_.page_mem_size_total_threshold);

    // has freed memory, return nullptr and try recycle
    if (GetIdleMemSize() > 0U) {
      return nullptr;
    }
  }
  const auto addr = DevAlloc(huge_page_mem_size);
  if (addr == nullptr) {
    return nullptr;
  }

  const auto span = BlockAlloc(allocator, addr, reinterpret_cast<MemAddr>(addr->GetAddr()),
                               static_cast<size_t>(huge_page_mem_size));
  if (span == nullptr) {
    device_allocator_.Free(addr);
    return nullptr;
  }
  // span is idle, so the usage count needs to be 0
  while (span->GetCount() > 0U) {
    span->SubCount();
  }

  SpanLayerId fit_layer_id = SpanLayerId_GetIdFromSize(huge_page_mem_size, ScalableAllocator::config_.page_idem_num);
  return SplitSpan(allocator, page_len, fit_layer_id, span, size);
}

ge::MemBlock *RtsCachingMemAllocator::AllocateWithTryRecycle(size_t size) {
  auto addr = Alloc(*this, size);
  if (addr != nullptr) {
    return addr;
  }

  GELOGW("%s Failed to apply for memory. We will try to free memory from memory pool, the above warning log can be "
         "ignored. Try to free cached memory...", GetId().c_str());
  PrintDetails(DLOG_INFO);

  Recycle();
  addr = Alloc(*this, size);
  return addr;
}

ge::MemBlock *RtsCachingMemAllocator::Malloc(size_t size) {
  GELOGI("Malloc size:%zu.", size);
  return AllocateWithTryRecycle(size);
}

ge::Status RtsCachingMemAllocator::Initialize(const std::vector<rtMemType_t> &memory_types) {
  (void) memory_types;
  return ge::SUCCESS;
}

bool RtsCachingMemAllocator::IsThresholdExceeded(const MemSize size) const {
  return ScalableAllocator::IsThresholdExceeded(size);
}

std::shared_ptr<RtsCachingMemAllocator> RtsCachingMemAllocator::GetAllocator(const uint32_t device_id,
                                                                             const rtMemType_t memory_type) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it_device = device_id_to_allocators_.find(device_id);
  if (it_device != device_id_to_allocators_.end()) {
    const auto it_type = it_device->second.find(memory_type);
    if (it_type != it_device->second.end()) {
      GELOGI("Get RtsCachingMemAllocator device id:%u memory type:%u.", device_id, memory_type);
      return it_type->second;
    }
  }
  const auto allocator = ge::MakeShared<RtsCachingMemAllocator>(device_id, memory_type);
  if (allocator != nullptr) {
    device_id_to_allocators_[device_id][memory_type] = allocator;
    GELOGI("Create RtsCachingMemAllocator device id:%u memory type:%u.", device_id, memory_type);
  }
  return allocator;
}

RtsAllocatorManager::~RtsAllocatorManager() {
}

ge::Status RtsAllocatorManager::Initialize(const std::vector<rtMemType_t> &memory_types) {
  GELOGI("RtsAllocatorManager Initialize.");
  return RtsCachingMemAllocator::Initialize(memory_types);
}

void RtsAllocatorManager::Finalize() {
  GELOGI("RtsAllocatorManager Finalize.");
  bool delay_finalize = false;
  {
    const std::lock_guard<std::mutex> lock(CachingMemAllocator::mutex_);
    delay_finalize = !CachingMemAllocator::all_caching_mem_allocators_.empty();
  }

  const std::lock_guard<std::mutex> lock(RtsCachingMemAllocator::mutex_);
  if (!delay_finalize) {
    RtsCachingMemAllocator::device_id_to_allocators_.clear();
  }
}

void RtsAllocatorManager::ReleaseResource(const uint32_t device_id) {
  GELOGI("RtsAllocatorManager ReleaseResource device id:%u.", device_id);
  {
    const std::lock_guard<std::mutex> lock(CachingMemAllocator::mutex_);
    for (auto allocator : CachingMemAllocator::all_caching_mem_allocators_) {
      if (allocator != nullptr) {
        allocator->Finalize(true);
      }
    }
  }

  const std::lock_guard<std::mutex> lock(RtsCachingMemAllocator::mutex_);
  for (auto &allocators_it : RtsCachingMemAllocator::device_id_to_allocators_) {
    for (auto &allocator_it : allocators_it.second) {
      if (allocator_it.second != nullptr) {
        allocator_it.second->Finalize(true);
      }
    }
  }
}

ge::Allocator *RtsAllocatorManager::CreateAllocator(const uint32_t device_id, const rtMemType_t memory_type) {
  auto allocator = RtsCachingMemAllocator::GetAllocator(device_id, memory_type);
  return allocator.get();
}

bool RegisterAllocatorManager() {
  static RtsAllocatorManager allocator_manager;
  ge::MemManager::Instance().RegisterAllocatorManager(&allocator_manager);
  return true;
}

const bool register_mng = RegisterAllocatorManager();
}  // namespace memory
}  // namespace gert
