/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H983DE866_DB91_4944_A25A_6BB0322FDB37
#define H983DE866_DB91_4944_A25A_6BB0322FDB37

#include "kernel/memory/type/mem_size.h"
#include "kernel/memory/type/mem_addr.h"
#include "kernel/memory/util/object_allocator.h"
#include "kernel/memory/allocator/scalable_config.h"
#include "runtime/mem.h"
#include "graph/manager/active_memory_allocator.h"

namespace gert {
using DeviceId = uint32_t;

class DeviceMemAllocator {
 public:
  virtual ~DeviceMemAllocator() = default;
  virtual BlockAddr Alloc(const MemSize size) = 0;
  virtual bool Free(ge::MemBlock *const addr) = 0;
  virtual DeviceId GetDeviceId() const = 0;
};

class RtMemAllocator : public DeviceMemAllocator {
 public:
  explicit RtMemAllocator(ge::Allocator &allocator, const DeviceId device_id, const uint32_t mem_type = RT_MEMORY_HBM);
  ~RtMemAllocator() override = default;
  BlockAddr Alloc(const MemSize size) override;
  bool Free(ge::MemBlock *const addr) override;
  DeviceId GetDeviceId() const override {
    return device_id_;
  }

  uint32_t GetMemType() const {
    return mem_type_;
  }

 private:
  ge::Allocator &allocator_;
  const DeviceId device_id_;
  const uint32_t mem_type_;
  ObjectAllocator<ge::MemBlock> block_allocator_{ScalableConfig().span_prepared_count};
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceMemAllocator &mem_allocator);

  BlockAddr Alloc(const MemSize size);
  void Free(ge::MemBlock *const addr);

  MemSize GetOccupiedSize() const {
    return occupied_size_ + active_memory_allocator_.ActiveMemorySize();
  }

  size_t GetAllocCount() const {
    return alloc_count_;
  }

  size_t GetFreeCount() const {
    return free_count_;
  }

  DeviceId GetDeviceId() const {
    return mem_allocator_.GetDeviceId();
  }

  ge::ExpandableActiveMemoryAllocatorImp &GetExpandableAllocator() {
    return active_memory_allocator_;
  }

  void SetLogErrorIfAllocFailed(bool log_error_if_alloc_failed);

 private:
  MemSize occupied_size_{0};
  size_t alloc_count_{0};
  size_t free_count_{0};
  DeviceMemAllocator &mem_allocator_;
  ge::ExpandableActiveMemoryAllocatorImp active_memory_allocator_{ge::kLargePageSizeBits};
  bool log_error_if_alloc_failed_ = true;
};
}

#endif
