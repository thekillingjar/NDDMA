/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TRACING_RECORDER_MANAGER_H
#define AUTOFUSE_TRACING_RECORDER_MANAGER_H
#include "common/tracing_type.h"
#include "tracing_recorder.h"
namespace ge {
class TracingRecorderManager {
 public:
  static TracingRecorderManager &Instance();
  // 记录一个外部事件，包括:
  // module:模块
  // msgs:事件名(主事件名/子事件名)
  // start:开始时间
  // duration:经历时间
  void RecordDuration(const TracingModule module, const std::string &msg, uint64_t start, uint64_t duration) const;
  void RecordDuration(TracingModule module, const std::vector<std::string> &msgs, uint64_t start, uint64_t duration) const;
  // 上报Tracing日志(落盘)
  Status Report(const TracingModule module) const;
  TracingRecorderManager(const TracingRecorderManager &) = delete;
  TracingRecorderManager(TracingRecorderManager &&) = delete;
  TracingRecorderManager &operator=(const TracingRecorderManager &) = delete;
  TracingRecorderManager &operator=(TracingRecorderManager &&) = delete;

 private:
  TracingRecorderManager();
  ~TracingRecorderManager() = default;
  void InitTracingRecorder();
  [[nodiscard]] TracingRecorder *GetTracingRecorder(TracingModule module) const;
  std::vector<std::unique_ptr<TracingRecorder>> tracing_recorders_;
};
}  // namespace ge
#endif  // AUTOFUSE_TRACING_RECORDER_MANAGER_H
