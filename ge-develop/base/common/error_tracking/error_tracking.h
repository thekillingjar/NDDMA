/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_ERROR_TRACKING_H_
#define GE_COMMON_ERROR_TRACKING_H_

#include <mutex>
#include "graph/op_desc.h"
#include "runtime/base.h"

namespace ge {
struct ErrorTrackingOpInfo {
  std::string op_name;
  std::string op_origin_name;
};

class  TaskKey {
public:
  TaskKey(uint32_t task_id, uint32_t stream_id, uint32_t  context_id, uint32_t  thread_id) :
    task_id_(task_id), stream_id_(stream_id), context_id_(context_id), thread_id_(thread_id) {
  }

  TaskKey(uint32_t task_id, uint32_t stream_id) : TaskKey(task_id, stream_id, UINT32_MAX, UINT32_MAX) {
  }
  bool operator <(const TaskKey &other) const {
      if (this->task_id_ < other.GetTaskId()) {
          return true;
      } else if (this->task_id_ == other.GetTaskId()) {
          if (this->stream_id_ < other.GetStreamId()) {
              return true;
          } else if (this->stream_id_ == other.GetStreamId()) {
              if (this->thread_id_ < other.GetThreadId()) {
                  return true;
              } else if (this->thread_id_ == other.GetThreadId()) {
                  return this->context_id_ < other.GetContextId();
              }
          }
      }

      return false;
  }
  uint32_t GetTaskId() const{
    return task_id_;
  }
  uint32_t GetStreamId() const{
    return stream_id_;
  }
  uint32_t GetThreadId() const{
    return thread_id_;
  }
  uint32_t GetContextId() const{
    return context_id_;
  }
private:
    uint32_t  task_id_;
    uint32_t  stream_id_;
    uint32_t  context_id_;
    uint32_t  thread_id_;
};

class ErrorTracking {
public:
  ErrorTracking(const ErrorTracking &) = delete;
  ErrorTracking(ErrorTracking &&) = delete;
  ErrorTracking &operator=(const ErrorTracking &) = delete;
  ErrorTracking &operator=(ErrorTracking &&) = delete;

  static ErrorTracking &GetInstance();

  void SaveGraphTaskOpdescInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id, const uint32_t model);
  void SaveGraphTaskOpdescInfo(const OpDescPtr &op, const TaskKey &key, const uint32_t model);
  void SaveSingleOpTaskOpdescInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id);
  void UpdateTaskId(const uint32_t old_task_id, const uint32_t new_task_id, const uint32_t stream_id, const uint32_t model);

  bool GetGraphTaskOpdescInfo(const uint32_t task_id, const uint32_t stream_id, ErrorTrackingOpInfo &op_info);

  bool GetGraphTaskOpdescInfo(const TaskKey &key, ErrorTrackingOpInfo &op_info);

  bool GetSingleOpTaskOpdescInfo(const uint32_t task_id, const uint32_t stream_id, ErrorTrackingOpInfo &op_info);

  void ClearUnloadedModelOpdescInfo(const uint32_t model);

private:
ErrorTracking();
void AddTaskOpdescInfo(const OpDescPtr &op, const TaskKey &key,
  std::map<TaskKey, ErrorTrackingOpInfo> &map, uint32_t max_count) const;
bool GetTaskOpdescInfo(const TaskKey &key, const std::map<TaskKey, ErrorTrackingOpInfo> &map,
                       ErrorTrackingOpInfo &op_info);
bool GetTaskOpdescInfo(const TaskKey &key, const std::map<uint32_t, std::map<TaskKey, ErrorTrackingOpInfo>> &map,
                       ErrorTrackingOpInfo &op_info);
std::mutex mutex_;
uint32_t single_op_max_count_{4096U};
std::map<uint32_t, std::map<TaskKey, ErrorTrackingOpInfo>> graph_task_to_op_info_;
std::map<TaskKey, ErrorTrackingOpInfo> single_op_task_to_op_info_;
};

  void ErrorTrackingCallback(rtExceptionInfo *const exception_data);

  uint32_t RegErrorTrackingCallBack();
}
#endif
