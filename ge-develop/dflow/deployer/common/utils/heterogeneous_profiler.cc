/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "heterogeneous_profiler.h"
#include "mmpa/mmpa_api.h"
#include "graph/types.h"
#include "common/util.h"

namespace ge {
namespace {
const size_t kStartimestampIndex = 0UL;
const size_t kEndTimestampIndex = 1UL;
const std::map<ge::ProfilerEvent, std::string> kEvnetTypeToStr = {{ProfilerEvent::kMbufAlloc, "[AllocMbuff]"},
  {ProfilerEvent::kMemCopyToMbuf, "[MemoryCopyToMbuf]"}, {ProfilerEvent::kMbufEnqueue, "[MbufEnqueue]"},
  {ProfilerEvent::kMbufDequeue, "[MbufDequeue]"}, {ProfilerEvent::kMbufCopyToMem, "[MbufCopyToMemory]"},
  {ProfilerEvent::kPrepareInputs, "[PrepareInputs]"}, {ProfilerEvent::kPrepareOutputs, "[PrepareOutputs]"},
  {ProfilerEvent::kDynamicExecute, "[DynamicExec]"}, {ProfilerEvent::kUpdateOutputs, "[UpdateOutputs]"}};
int64_t GetThread() {
  thread_local static int64_t tid = static_cast<int64_t>(mmGetTid());
  return tid;
}

bool CheckDetailRecordInvalid(const std::map<uint32_t, std::vector<uint64_t>> &check_data, const size_t expect_size) {
  for (const auto &data : check_data) {
    if (data.second.size() != expect_size) {
      GEEVENT("Current senario not support calculate total duration");
      return false;
    }
  }
  return true;
}

uint64_t GetMinMaxStartTimestampByIndex(const std::map<uint32_t, std::vector<uint64_t>> &input_data,
                                        const size_t index,
                                        const bool min_value) {
  uint64_t result = min_value ? std::numeric_limits<uint64_t>::max() : 0UL;
  for (const auto &data : input_data) {
    if (data.second.size() <= index) {
      return result;
    }
    if (min_value) {
      result = (data.second.at(index) < result) ? data.second.at(index) : result;
    } else {
      result = (data.second.at(index) > result) ? data.second.at(index) : result;
    }
  }
  return result;
}
}

HeterogeneousProfiler &HeterogeneousProfiler::Instance() {
  static HeterogeneousProfiler heterogeneous_profiler;
  return heterogeneous_profiler;
}

void HeterogeneousProfiler::InitHeterogeneousPoriler() {
  const char_t *profiling_to_std_out = nullptr;
  MM_SYS_GET_ENV(MM_ENV_GE_PROFILING_TO_STD_OUT, profiling_to_std_out);
  if ((profiling_to_std_out != nullptr) && strcmp(profiling_to_std_out, "2") == 0) {
    enable_flag_ = true;
  } else {
    enable_flag_ = false;
  }
  GEEVENT("Init profiler flag:%d", static_cast<int32_t>(enable_flag_));
}

void HeterogeneousProfiler::RecordHeterogeneousProfilerEvent(const ProfilerType type, const ProfilerEvent event_type,
                                                             const int32_t device_id, const uint32_t queue_id) {
  if (!enable_flag_) {
    return;
  }
  const HeterogeneousProfilerRecordKey record_key = {.thread_id = GetThread(),
                                              .device_id = device_id,
                                              .queue_id = queue_id,
                                              .profiler_event = event_type};
  const std::lock_guard<std::mutex> lock(mutex_);
  const uint64_t current_timestamp = ge::GetCurrentTimestamp();

  if (type == ProfilerType::kStartPoint) {
    // set first iterator value
    if (queueId_and_event_to_iter_.count(record_key) == 0UL) {
      queueId_and_event_to_iter_[record_key] = 0UL;
    }
    const uint64_t current_iterator = queueId_and_event_to_iter_[record_key];
    // record starttimestamp
    auto &profiler_data = profiler_record_[record_key];
    if (!profiler_data[current_iterator].empty()) {
      profiler_data[current_iterator][kStartimestampIndex] = current_timestamp;
    } else {
      profiler_data[current_iterator].emplace_back(current_timestamp);
    }

    if ((event_type == ProfilerEvent::kMbufAlloc) && (queue_id != std::numeric_limits<uint32_t>::max())) {
      mbuf_alloc_start_total_record_[queue_id].emplace_back(current_timestamp);
    }
  } else if (type == ProfilerType::kEndPoint) {
    const auto iterator_iter = queueId_and_event_to_iter_.find(record_key);
    if (iterator_iter == queueId_and_event_to_iter_.cend()) {
      GELOGW("QueueId:%u event:%d get end profiler before start profiler is illegal,",
             queue_id, static_cast<int32_t>(event_type));
      return;
    }
    const uint64_t current_iterator = iterator_iter->second;
    auto &profiler_data = profiler_record_[record_key];
    profiler_data[current_iterator].emplace_back(current_timestamp);
    if ((event_type == ProfilerEvent::kMbufEnqueue) && (queue_id != std::numeric_limits<uint32_t>::max())) {
      enqueue_end_total_record_[queue_id].emplace_back(current_timestamp);
    }
    // increase iterator id when get end
    ++queueId_and_event_to_iter_[record_key];
  } else {
    GELOGW("Invaid profiler type:%d from queue_id:%u", static_cast<int32_t>(type), queue_id);
  }
}

void HeterogeneousProfiler::ProcessDetailTimeStamp() {
  if (mbuf_alloc_start_total_record_.empty() || enqueue_end_total_record_.empty()) {
    GEEVENT("There are no detail record to be processed");
    return;
  }
  const size_t exp_iter_num = mbuf_alloc_start_total_record_.cbegin()->second.size();
  if (!CheckDetailRecordInvalid(mbuf_alloc_start_total_record_, exp_iter_num) ||
      !CheckDetailRecordInvalid(enqueue_end_total_record_, exp_iter_num)) {
    return;
  }

  for (size_t i = 0UL; i < exp_iter_num; ++i) {
    uint64_t input_start_timestamp = GetMinMaxStartTimestampByIndex(mbuf_alloc_start_total_record_, i, true);
    uint64_t input_end_timestamp = GetMinMaxStartTimestampByIndex(enqueue_end_total_record_, i, false);
    if ((input_start_timestamp != std::numeric_limits<uint64_t>::max()) &&
        (input_end_timestamp != 0UL)&& (input_start_timestamp <= input_end_timestamp)) {
      GEEVENT("[HeterogeneousProfiler] [Iterator]:%zu [Input prepare duration]:%lu",
              i, (input_end_timestamp - input_start_timestamp));
    } else {
      GEEVENT("[HeterogeneousProfiler] [Iterator]:%zu Invalid timestamp: input start:%lu end:%lu",
              i, input_start_timestamp, input_end_timestamp);
    }
  }
}

void HeterogeneousProfiler::PrintAvgHeterogeneousProfilerData(const HeterogeneousProfilerRecordKey &key,
                                                              const uint64_t &totalDuration,
                                                              const uint32_t &recordNum,
                                                              const uint32_t &maxDuration) const {
  std::stringstream ss;
  ss << "[Record thread id]:" << key.thread_id << ", ";
  ss << "[Device id]:" << key.device_id << ", ";
  ss << "[Queue id]:" << key.queue_id << ", ";
  const auto type_iter = kEvnetTypeToStr.find(key.profiler_event);
  if (type_iter != kEvnetTypeToStr.cend()) {
    ss << "[Event type]:" << type_iter->second << ", ";
  } else {
    ss << "[Event type]:Invalid" << static_cast<int32_t>(key.profiler_event) << ", ";
  }
  if (recordNum != 0U) {
    ss << "[PerDuration]:" << totalDuration / recordNum << ", [Times]:" << recordNum << ", ";
    ss << "[MaxDuration]:" << maxDuration;
  }
  GEEVENT("[PerHeterogeneousProfiler] %s", ss.str().c_str());
}

void HeterogeneousProfiler::PrintHeterogeneousProfilerData() {
  if (!enable_flag_) {
    return;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto profiler_data : profiler_record_) {
    auto totalDuration = 0ULL;
    auto recordNum = 0U;
    auto maxDuration = 0U;
    for (auto record : profiler_data.second) {
      std::stringstream ss;
      ss << "[Record thread id]:" << profiler_data.first.thread_id << ", ";
      ss << "[Queue id]:" << profiler_data.first.queue_id << ", ";
      const auto type_iter = kEvnetTypeToStr.find(profiler_data.first.profiler_event);
      if (type_iter != kEvnetTypeToStr.cend()) {
        ss << "[Event type]:" << type_iter->second << ", ";
      } else {
        ss << "[Event type]:Invalid" << static_cast<int32_t>(profiler_data.first.profiler_event) << ", ";
      }
      ss << "[Iterator]:" << record.first << ", ";
      if (record.second.size() != 2UL) {
        ss << "[Invalid record size]:" << record.second.size() << ",";
      } else {
        if (record.second[kStartimestampIndex] <= record.second[kEndTimestampIndex]) {
          auto duration = record.second[kEndTimestampIndex] - record.second[kStartimestampIndex];
          totalDuration += duration;
          recordNum++;
          maxDuration = duration > maxDuration ? duration : maxDuration;
          ss << "[Duration]:" << duration << "us";
        } else {
          ss << "[Invalid start timestamp]:" << record.second[kStartimestampIndex] <<
                ", [Invalid end timestamp]:" << record.second[kEndTimestampIndex];
        }
      }
      GEEVENT("[HeterogeneousProfiler] %s", ss.str().c_str());
    }
    PrintAvgHeterogeneousProfilerData(profiler_data.first, totalDuration, recordNum, maxDuration);
  }
  ProcessDetailTimeStamp();
  enable_flag_ = false;
  profiler_record_.clear();
  queueId_and_event_to_iter_.clear();
  mbuf_alloc_start_total_record_.clear();
  enqueue_end_total_record_.clear();
}
}  // namespace ge
