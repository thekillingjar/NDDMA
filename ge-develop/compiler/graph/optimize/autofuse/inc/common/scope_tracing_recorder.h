/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_SCOPE_TRACING_RECORDER_H
#define AUTOFUSE_SCOPE_TRACING_RECORDER_H

#include <chrono>
#include "tracing_type.h"
namespace ge {
class ScopeTracingRecorder {
 public:
  ScopeTracingRecorder(const TracingModule stage, std::string msg);
  ScopeTracingRecorder(const TracingModule stage, const std::vector<std::string> &msgs);
  ~ScopeTracingRecorder();

 private:
  TracingModule stage_;
  std::string msg_;
  uint64_t start_;
};
}  // namespace ge
inline uint64_t CurrentTimeNanos() {
  return std::chrono::time_point_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now()
                 ).time_since_epoch().count();
}
extern "C" {
void TracingRecordDuration(const ge::TracingModule stage, const std::vector<std::string> &msgs,
                           const uint64_t start, const uint64_t duration);
void ReportTracingRecordDuration(const ge::TracingModule stage);
}
#define CONCAT_(x, y) x##y
// 记录函数级打点
#define TRACING_PERF_SCOPE(module, ...) \
  ge::ScopeTracingRecorder scope##__COUNTER__(module, std::vector<std::string>{__VA_ARGS__})

// 记录代码片段耗时,START和END需要成对使用
#define TRACING_DURATION_START(tag) const uint64_t CONCAT_(startUsec, tag) = CurrentTimeNanos()
#define TRACING_DURATION_END(stage, tag, ...)                                                    \
  do {                                                                                           \
    const uint64_t CONCAT_(endUsec, tag) = CurrentTimeNanos();                                   \
    const auto CONCAT_(duration, tag) = (CONCAT_(endUsec, tag) - CONCAT_(startUsec, tag));       \
    TracingRecordDuration(stage, std::vector<std::string>{__VA_ARGS__}, CONCAT_(startUsec, tag), \
                          CONCAT_(duration, tag));                                               \
  } while (false)
#define TRACING_COMPILE_DURATION_END(tag, ...) TRACING_DURATION_END(ge::TracingModule::kModelCompile, tag, __VA_ARGS__)
#define TRACING_INIT_DURATION_END(tag, ...) TRACING_DURATION_END(ge::TracingModule::kCANNInitialize, tag, __VA_ARGS__)
#define TRACING_LOAD_DURATION_END(tag, ...) TRACING_DURATION_END(ge::TracingModule::kModelLoad, tag, __VA_ARGS__)
#endif  // AUTOFUSE_SCOPE_TRACING_RECORDER_H
