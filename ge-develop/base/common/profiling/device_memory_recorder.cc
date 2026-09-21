/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/device_memory_recorder.h"
#include "runtime/subscriber/global_profiler.h"
#include "graph/def_types.h"

namespace gert {
std::atomic<uint64_t> DeviceMemoryRecorder::total_allocate_memory_{0UL};
std::atomic<uint64_t> DeviceMemoryRecorder::total_reserve_memory_{0UL};
std::queue<MemoryRecorder> DeviceMemoryRecorder::memory_record_queue_;
std::mutex DeviceMemoryRecorder::mtx_;

void DeviceMemoryRecorder::SetRecorder(const void *const addr, const int64_t size) {
  if (GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kMemory)) {
    MemoryRecorder memory_info_data;
    memory_info_data.size = size;
    memory_info_data.addr = ge::PtrToValue(addr);
    memory_info_data.total_allocate_memory = GetTotalAllocateMemory();
    memory_info_data.total_reserve_memory = GetTotalReserveMemory();
    memory_info_data.time_stamp = MsprofSysCycleTime();
    GELOGI("[CannMemoryProfiler][RecordMemory] Record memory info: "
        "addr: %llu, size: %lld, total allocate size: %llu, total reserve size: %lld"
        "time stamp: %llu",
        memory_info_data.addr,
        memory_info_data.size,
        memory_info_data.total_allocate_memory,
        memory_info_data.total_reserve_memory,
        memory_info_data.time_stamp);
    const std::lock_guard<std::mutex> lck(mtx_);
    memory_record_queue_.push(memory_info_data);
  }
}

bool DeviceMemoryRecorder::IsRecorderEmpty() {
  const std::lock_guard<std::mutex> lck(mtx_);
  return memory_record_queue_.empty();
}

const MemoryRecorder DeviceMemoryRecorder::GetRecorder() {
  MemoryRecorder memory_info;
  const std::lock_guard<std::mutex> lck(mtx_);
  if (!memory_record_queue_.empty()) {
    memory_info = memory_record_queue_.front();
    memory_record_queue_.pop();
  }
  return memory_info;
}
} // namespace gert