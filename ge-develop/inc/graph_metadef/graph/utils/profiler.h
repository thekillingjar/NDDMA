/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_PROFILER_H
#define METADEF_CXX_PROFILER_H
#include <memory>
#include <array>
#include <ostream>
#include <chrono>
#include <atomic>
#include "graph/types.h"

namespace ge {
namespace profiling {
constexpr size_t kMaxStrLen = 256UL;
constexpr int64_t kMaxStrIndex = 1024 * 1024;
constexpr size_t kMaxRecordNum = 10UL * 1024UL * 1024UL;
enum class EventType {
  kEventStart,
  kEventEnd,
  kEventTimestamp,
  kEventTypeEnd
};
struct ProfilingRecord {
  int64_t element;
  int64_t thread;
  int64_t event;
  EventType et;
  std::chrono::time_point<std::chrono::system_clock> timestamp;
};

struct StrHash {
    char_t str[kMaxStrLen];
    uint64_t hash;
};

class Profiler {
 public:
  static std::unique_ptr<Profiler> Create();
  void UpdateHashByIndex(const int64_t index, const uint64_t hash);
  void RegisterString(const int64_t index, const std::string &str);
  void RegisterStringHash(const int64_t index, const uint64_t hash, const std::string &str);
  void Record(const int64_t element, const int64_t thread, const int64_t event, const EventType et,
              const std::chrono::time_point<std::chrono::system_clock> time_point);
  void RecordCurrentThread(const int64_t element, const int64_t event, const EventType et);
  void RecordCurrentThread(const int64_t element, const int64_t event, const EventType et,
                           const std::chrono::time_point<std::chrono::system_clock> time_point);

  void Reset();
  void Dump(std::ostream &out_stream) const;

  size_t GetRecordNum() const noexcept;
  const ProfilingRecord *GetRecords() const;

  using ConstStringHashesPointer = StrHash const(*);
  using StringHashesPointer = StrHash (*);
  ConstStringHashesPointer GetStringHashes() const;
  StringHashesPointer GetStringHashes() ;

  ~Profiler();
  Profiler();

 private:
  void DumpByIndex(const int64_t index, std::ostream &out_stream) const;

 private:
  std::atomic<size_t> record_size_;
  std::array<ProfilingRecord, kMaxRecordNum> records_;
  StrHash indexes_to_str_hashes_[kMaxStrIndex];
};
}
}
#endif  // METADEF_CXX_PROFILER_H
