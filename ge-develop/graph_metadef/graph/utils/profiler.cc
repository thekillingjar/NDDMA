/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/profiler.h"
#include <cstring>
#include "mmpa/mmpa_api.h"
#include "securec.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/def_types.h"
#include "graph/types.h"

namespace ge {
namespace profiling {
namespace {
constexpr char_t kVersion[] = "1.0";
int64_t GetThread() {
  thread_local static int64_t tid = static_cast<int64_t>(mmGetTid());
  return tid;
}
void DumpEventType(const EventType et, std::ostream &out_stream) {
  switch (et) {
    case EventType::kEventStart:
      out_stream << "Start";
      break;
    case EventType::kEventEnd:
      out_stream << "End";
      break;
    case EventType::kEventTimestamp:
      break;
    default:
      out_stream << "UNKNOWN(" << static_cast<int64_t>(et) << ")";
      break;
  }
}
}

void Profiler::RecordCurrentThread(const int64_t element, const int64_t event, const EventType et) {
  Record(element, GetThread(), event, et, std::chrono::system_clock::now());
}

void Profiler::RecordCurrentThread(const int64_t element, const int64_t event, const EventType et,
                                   const std::chrono::time_point<std::chrono::system_clock> time_point) {
  Record(element, GetThread(), event, et, time_point);
}

void Profiler::UpdateHashByIndex(const int64_t index, const uint64_t hash) {
  if (index >= kMaxStrIndex) {
    return;
  }
  PtrAdd<StrHash>(GetStringHashes(), static_cast<size_t>(kMaxStrIndex), static_cast<size_t>(index))->hash = hash;
}

void Profiler::RegisterString(const int64_t index, const std::string &str) {
  if (index >= kMaxStrIndex) {
    return;
  }

  // can not use strcpy_s, which will copy nothing when the length of str beyond kMaxStrLen
  const auto ret = strncpy_s(PtrAdd<StrHash>(GetStringHashes(),
                                             static_cast<size_t>(kMaxStrIndex), static_cast<size_t>(index))->str,
                             kMaxStrLen, str.c_str(), kMaxStrLen - 1UL);
  if (ret != EN_OK) {
    GELOGW("Register string failed, index %ld, str %s", index, str.c_str());
  }
}

void Profiler::RegisterStringHash(const int64_t index, const uint64_t hash, const std::string &str) {
  if (index >= kMaxStrIndex) {
    return;
  }

  // can not use strcpy_s, which will copy nothing when the length of str beyond kMaxStrLen
  const auto ret = strncpy_s(PtrAdd<StrHash>(GetStringHashes(),
                                             static_cast<size_t>(kMaxStrIndex), static_cast<size_t>(index))->str,
                             kMaxStrLen, str.c_str(), kMaxStrLen - 1UL);
  if (ret != EN_OK) {
    GELOGW("Register string failed, index %ld, str %s", index, str.c_str());
  }
  PtrAdd<StrHash>(GetStringHashes(), static_cast<size_t>(kMaxStrIndex), static_cast<size_t>(index))->hash = hash;
}

void Profiler::Record(const int64_t element, const int64_t thread, const int64_t event, const EventType et,
                      const std::chrono::time_point<std::chrono::system_clock> time_point) {
  auto current_index = record_size_++;
  if (current_index >= kMaxRecordNum) {
    return;
  }
  records_[current_index] = ProfilingRecord({element, thread, event, et, time_point});
}
void Profiler::Dump(std::ostream &out_stream) const {
  if (record_size_ == 0UL) {
    return;
  }
  size_t print_size = record_size_;
  out_stream << "Profiler version: " << &kVersion[0]
             << ", dump start, records num: " << print_size << std::endl;
  if (print_size > records_.size()) {
    out_stream << "Too many records(" << print_size << "), the records after "
               << records_.size() << " will be dropped" << std::endl;
    print_size = records_.size();
  }
  for (size_t i = 0UL; i < print_size; ++i) {
    auto &rec = records_[i];
    // in format: <timestamp> <thread-id> <module-id> <record-type> <event-type>
    out_stream << std::chrono::duration_cast<std::chrono::nanoseconds>(rec.timestamp.time_since_epoch()).count() << ' ';
    out_stream << rec.thread << ' ';
    DumpByIndex(rec.element, out_stream);
    out_stream << ' ';
    DumpByIndex(rec.event, out_stream);
    out_stream << ' ';
    DumpEventType(rec.et, out_stream);
    out_stream << std::endl;
  }
  out_stream << "Profiling dump end" << std::endl;
}
void Profiler::DumpByIndex(const int64_t index, std::ostream &out_stream) const {
  if ((index < 0) || (index >= kMaxStrIndex) ||
      (strnlen(PtrAdd<const StrHash>(GetStringHashes(),
          static_cast<size_t>(kMaxStrIndex),
          static_cast<size_t>(index))->str, kMaxStrLen) == 0UL)) {
    out_stream << "UNKNOWN(" << index << ")";
  } else {
    out_stream << '[' << PtrAdd<const StrHash>(GetStringHashes(),
        static_cast<size_t>(kMaxStrIndex), static_cast<size_t>(index))->str << "]";
  }
}
Profiler::Profiler() : record_size_(0UL), records_(), indexes_to_str_hashes_() {}
void Profiler::Reset() {
  // 不完全reset，indexes_to_str_hashes_还是有值的
  record_size_ = 0UL;
}
std::unique_ptr<Profiler> Profiler::Create() {
  return ComGraphMakeUnique<Profiler>();
}
size_t Profiler::GetRecordNum() const noexcept {
  return record_size_;
}
const ProfilingRecord *Profiler::GetRecords() const {
  return &(records_[0UL]);
}
Profiler::ConstStringHashesPointer Profiler::GetStringHashes() const {
  return indexes_to_str_hashes_;
}
Profiler::StringHashesPointer Profiler::GetStringHashes() {
  return indexes_to_str_hashes_;
}
Profiler::~Profiler() = default;
}
}
