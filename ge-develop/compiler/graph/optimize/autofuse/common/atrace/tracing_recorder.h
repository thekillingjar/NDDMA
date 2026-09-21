/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TRACING_RECORDER_H
#define AUTOFUSE_TRACING_RECORDER_H

#include <mutex>
#include "common/tracing_type.h"
#include "atracing_reporter.h"

namespace ge {
constexpr int64_t kInvalidHandle = -1;
class TracingRecorder {
 public:
  explicit TracingRecorder(const TracingModule module);
  ~TracingRecorder();
  // 记录message的经历的时间
  void RecordDuration(const std::vector<std::string> &tracing_msg, uint64_t start, uint64_t duration);
  // 上报tracing日志到tracing模块
  void Report();

 private:
  TracingRecord *RecordMsgs(const std::vector<std::string> &tracing_msg, const TracingEvent ev);
  void SubmitTraceMsgs(const TracingRecord *tracing_record);
  void Initialize();
  void Finalize();
  void Report(const TracingRecord *tracing_record, int64_t record_num);
  std::string GetHandleName() const;
  std::vector<TracingRecord> records_;
  std::vector<TraHandle> handles_;
  std::vector<TraEventHandle> finalize_event_handles_;
  TracingModule module_{TracingModule::kModelCompile};
  bool is_ready_{false};
  std::atomic<int32_t> event_bind_num_{0};
  std::mutex mu_;
};
}  // namespace ge

#endif  // AUTOFUSE_TRACING_RECORDER_H
