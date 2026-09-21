/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TRACING_TYPE_H
#define AUTOFUSE_TRACING_TYPE_H
#include "common/checker.h"
namespace ge {
enum class TracingModule : uint8_t {
  kCANNInitialize,
  kModelCompile,
  kAutoFuseBackend,
  kModelLoad,
  kTracingModuleEnd,
};
enum class TracingEvent : uint8_t {
  kEventDuration,
  kEventTypeEnd
};
struct TracingRecord {
  std::vector<std::string> tracing_msgs;
  uint64_t start;
  uint64_t duration;
  int32_t pid;
  int32_t thread;
  uint8_t event;
  [[nodiscard]] std::string Debug() const {
    std::stringstream ss;
    ss << "TracingRecord[msg:";
    for (const auto &msg : tracing_msgs) {
      ss << msg << ",";
    }
    ss << "start=" << start << ",duration=" << duration << "ns,pid=" << pid << ",tid=" << thread
       << ",event=" << std::to_string(event) << "]";
    return ss.str();
  }
};
// kMaxAtracingProfilingRecordNum * kMaxAtracingProfilingMsgSize 必须小于128K
constexpr int32_t kMaxAtracingProfilingRecordNum = 512;
constexpr int32_t kMaxAtracingProfilingMsgSize = 240 - 1;
constexpr int32_t kMaxAtracingProfilingHandleSzie = 100;
constexpr int32_t kMaxEventBindNum = 5;
}  // namespace ge
#endif  // AUTOFUSE_TRACING_TYPE_H
