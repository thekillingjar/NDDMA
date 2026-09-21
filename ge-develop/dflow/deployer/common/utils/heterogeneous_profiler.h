/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_COMMON_HETEROGENEOUS_PROFILER_H_
#define AIR_RUNTIME_COMMON_HETEROGENEOUS_PROFILER_H_
#include <map>
#include <atomic>
#include <mutex>
#include <vector>
namespace ge {
enum class ProfilerEvent : int32_t {
    kMbufAlloc,
    kMemCopyToMbuf,
    kMbufEnqueue,
    kMbufDequeue,
    kMbufCopyToMem,
    kPrepareInputs,
    kPrepareOutputs,
    kDynamicExecute,
    kUpdateOutputs,
};
enum class ProfilerType : int32_t {
    kStartPoint,
    kEndPoint,
};

class HeterogeneousProfiler {
 public:
  struct HeterogeneousProfilerRecordKey {
    int64_t thread_id;
    int32_t device_id;
    uint32_t queue_id;
    ProfilerEvent profiler_event;
    bool operator < (const HeterogeneousProfilerRecordKey &other) const {
      if (thread_id != other.thread_id) {
        return thread_id < other.thread_id;
      } else if (queue_id != other.queue_id) {
        return queue_id < other.queue_id;
      } else if (device_id != other.device_id) {
        return device_id < other.device_id;
      } else {
        return profiler_event < other.profiler_event;
      }
    }
  };
  static HeterogeneousProfiler &Instance();
  void InitHeterogeneousPoriler();
  void RecordHeterogeneousProfilerEvent(const ProfilerType type, const ProfilerEvent event_type,
                                        const int32_t device_id = std::numeric_limits<uint32_t>::max(),
                                        const uint32_t queue_id = std::numeric_limits<uint32_t>::max());
  void PrintHeterogeneousProfilerData();
 private:
  void ProcessDetailTimeStamp();
  void PrintAvgHeterogeneousProfilerData(const HeterogeneousProfilerRecordKey &key,
                                         const uint64_t &totalDuration,
                                         const uint32_t &recordNum,
                                         const uint32_t &maxDuration) const;
  HeterogeneousProfiler() = default;
  ~HeterogeneousProfiler() = default;
  bool enable_flag_;
  std::mutex mutex_;
  // value is iterator id
  std::map<HeterogeneousProfilerRecordKey, uint64_t> queueId_and_event_to_iter_;
  // key:thread_id+queue_id+event , value(key:iterator id, value:0th-starttime 1st-endtime)
  std::map<HeterogeneousProfilerRecordKey, std::map<uint64_t, std::vector<uint64_t>>> profiler_record_;
  // queueid:timestamps list
  std::map<uint32_t, std::vector<uint64_t>> mbuf_alloc_start_total_record_;
  std::map<uint32_t, std::vector<uint64_t>> enqueue_end_total_record_;
};
}  // namespace ge
#endif  // AIR_RUNTIME_COMMON_HETEROGENEOUS_PROFILER_H_
