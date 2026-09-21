/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_V2_KERNEL_MEMORY_UTIL_DEVICE_MEMORY_STATS_H
#define RUNTIME_V2_KERNEL_MEMORY_UTIL_DEVICE_MEMORY_STATS_H
#include <atomic>
#include <queue>
#include <mutex>
#include "aprof_pub.h"

namespace gert {
  struct MemoryRecorder {
    int64_t size;
    uint64_t addr;
    uint64_t total_allocate_memory;
    uint64_t total_reserve_memory;
    uint64_t time_stamp;
  };

  class DeviceMemoryRecorder {
   public:
    static uint64_t GetTotalAllocateMemory() { return total_allocate_memory_.load(); }
    static uint64_t GetTotalReserveMemory() { return total_reserve_memory_.load(); }
    static void AddTotalReserveMemory(const uint64_t &num) { total_reserve_memory_ += num; }
    static void ReduceTotalReserveMemory(const uint64_t &num) { total_reserve_memory_ -= num; }
    static void ClearReserveMemory() { total_reserve_memory_.store(0UL); }
    static const MemoryRecorder GetRecorder();
    static bool IsRecorderEmpty();
    static void SetRecorder(const void *const addr, const int64_t size);
    static void AddTotalAllocateMemory(const uint64_t &num) { total_allocate_memory_ += num; }
    static void ReduceTotalAllocateMemory(const uint64_t &num) { total_allocate_memory_ -= num; }
   private:
    static std::atomic<uint64_t> total_allocate_memory_;
    static std::atomic<uint64_t> total_reserve_memory_;
    static std::queue<MemoryRecorder> memory_record_queue_;
    static std::mutex mtx_;
  };
}
#endif
