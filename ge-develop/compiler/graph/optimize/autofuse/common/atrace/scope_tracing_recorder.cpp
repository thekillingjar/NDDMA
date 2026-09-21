/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include "common/tracing_type.h"
#include "atrace/tracing_recorder_manager.h"
#include "common/scope_tracing_recorder.h"
namespace ge {
namespace {
std::string ToString(const std::vector<std::string> &msgs) {
  std::string out;
  int32_t id = 0;
  for (const auto &msg : msgs) {
    if (id != 0) {
      out.append(",");
    }
    out.append(msg);
    id++;
  }
  return out;
}
}
ScopeTracingRecorder::ScopeTracingRecorder(const TracingModule stage, std::string msg)
    : stage_(stage),
      msg_(std::move(msg)),
      start_(std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now())
                 .time_since_epoch()
                 .count()) {}
ScopeTracingRecorder::ScopeTracingRecorder(const TracingModule stage, const std::vector<std::string> &msgs)
    : stage_(stage),
      msg_(ToString(msgs)),
      start_(std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now())
                 .time_since_epoch()
                 .count()) {}
ScopeTracingRecorder::~ScopeTracingRecorder() {
  auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now())
                 .time_since_epoch()
                 .count();
  TracingRecorderManager::Instance().RecordDuration(stage_, {msg_}, start_, end - start_);
}
}  // namespace ge
extern "C" {
void TracingRecordDuration(const ge::TracingModule stage, const std::vector<std::string> &msgs,
                           const uint64_t start, const uint64_t duration) {
  ge::TracingRecorderManager::Instance().RecordDuration(stage, msgs, start, duration);
}
void ReportTracingRecordDuration(const ge::TracingModule stage) {
  ge::TracingRecorderManager::Instance().Report(stage);
}
}