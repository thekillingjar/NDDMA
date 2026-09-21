/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_HYBRID_PROFILER_H_
#define GE_HYBRID_EXECUTOR_HYBRID_PROFILER_H_

#include <chrono>
#include <ostream>

#include "graph/types.h"

namespace ge {
namespace hybrid {
class HybridProfiler {
 public:
  enum class EventType {
    GENERAL,
    SHAPE_INFERENCE,
    COMPILE,
    EXECUTION,
    CALLBACKS
  };

  struct Event {
    std::chrono::system_clock::time_point timestamp;
    EventType event_type;
    std::string desc;
  };

  HybridProfiler();
  ~HybridProfiler() = default;

  void RecordEvent(const EventType event_type, const char_t *const fmt, ...);

  void Reset();

  void Dump(std::ostream &output_stream);

 private:
  std::vector<Event> events_;
  std::atomic_int counter_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_HYBRID_PROFILER_H_
