/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RTS_CACHING_MEM_ALLOCATOR_H
#define AIR_RTS_CACHING_MEM_ALLOCATOR_H

#include <cstdint>
#include <array>
#include <memory>
#include <map>
#include "runtime/mem.h"
#include "ge/ge_api_types.h"
#include "kernel/memory/allocator/scalable_allocator.h"
#include "ge/ge_allocator.h"
#include "runtime/base.h"
#include "runtime/mem_allocator.h"

namespace gert {
namespace memory {
class FirstLevelPool : public DeviceMemAllocator {
 public:
  FirstLevelPool(ge::Allocator &allocator, DeviceId device_id, const std::string &purpose)
      : allocator_(allocator), device_id_(device_id), purpose_(purpose) {}
  BlockAddr Alloc(const MemSize size) override {
    auto addr = allocator_.Malloc(size);
    if (addr != nullptr) {
      GE_PRINT_DYNAMIC_MEMORY(rtMalloc, purpose_.c_str(), size);
    }
    GELOGI("Malloc from level_1 allocator addr:%p size:%lu.", addr, size);
    return addr;
  }
  bool Free(ge::MemBlock *const addr) override {
    auto block = addr;
    if (block != nullptr) {
      GELOGI("Free to level_1 allocator dddr:%p size:%lu.", block, block->GetSize());
      block->Free();
    }
    return true;
  }
  DeviceId GetDeviceId() const override {
    return device_id_;
  }

 private:
  ge::Allocator &allocator_;
  DeviceId device_id_;
  std::string purpose_;
};

class RtsCachingMemAllocator : public ge::Allocator, public ScalableAllocator {
 public:
  RtsCachingMemAllocator(const uint32_t device_id, const rtMemType_t memory_type);
  ~RtsCachingMemAllocator() override;
  ge::MemBlock *Malloc(size_t size) override;
  void Free(ge::MemBlock *block) override;
  PageSpan *FetchNewSpan(ge::Allocator &allocator, const MemSize size, const PageLen page_len) override;
  static ge::Status Initialize(const std::vector<rtMemType_t> &memory_types);

  DeviceId GetDeviceId() const {
    return rt_mem_allocator_.GetDeviceId();
  }

  static std::shared_ptr<RtsCachingMemAllocator> GetAllocator(const uint32_t device_id, const rtMemType_t memory_type);
 protected:
  bool IsThresholdExceeded(const MemSize size) const override;

 public:
  static std::map<int32_t, std::map<rtMemType_t, std::shared_ptr<RtsCachingMemAllocator>>> device_id_to_allocators_;
  static std::mutex mutex_;
 private:
  ge::MemBlock *AllocateWithTryRecycle(size_t size);

 private:
  SpanAllocatorImp span_allocator_;
  RtMemAllocator rt_mem_allocator_{*this, 0U};
};

class RtsAllocatorManager : public AllocatorManager {
 public:
  RtsAllocatorManager() = default;
  ~RtsAllocatorManager() override;
  ge::Status Initialize(const std::vector<rtMemType_t> &memory_types) override;
  void Finalize() override;
  void ReleaseResource(const uint32_t device_id = 0U) override;
  ge::Allocator *CreateAllocator(const uint32_t device_id, const rtMemType_t memory_type) override;
};

class RtsFirstLevelPool : public FirstLevelPool {
 public:
  RtsFirstLevelPool(ge::Allocator &allocator, DeviceId device_id, const std::string &purpose)
      : FirstLevelPool(allocator, device_id, purpose) {}
  BlockAddr Alloc(const MemSize size) override {
    const std::lock_guard<std::mutex> lock(RtsCachingMemAllocator::mutex_);
    return FirstLevelPool::Alloc(size);
  }
  bool Free(ge::MemBlock *const addr) override {
    const std::lock_guard<std::mutex> lock(RtsCachingMemAllocator::mutex_);
    return FirstLevelPool::Free(addr);
  }
};
}
}  // namespace gert

#endif
