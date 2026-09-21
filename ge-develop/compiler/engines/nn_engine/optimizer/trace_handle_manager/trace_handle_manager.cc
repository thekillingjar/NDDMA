/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trace_handle_manager/trace_handle_manager.h"
#include <vector>
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "atrace_pub.h"
#include "adapter/common/op_store_adapter_manager.h"

namespace fe {
namespace {
constexpr char const *kGlobalTraceHandleName = "FE_Global_Trace";
constexpr char const *kStatisticsTraceHandleName = "FE_Statistics_Trace";
constexpr char const *kCompileTraceHandlePrefix = "FE_CompileTd_";
constexpr char const *kFinalizeEventName = "FE_Finalize_Event";
const uint64_t kTreadIdReminder = 10;
}
TraceHandleManager::TraceHandleManager() : is_init_(false), global_handle_(-1), statistics_handle_(-1),
                                           finalize_event_handle_() {}

TraceHandleManager::~TraceHandleManager() {}

TraceHandleManager& TraceHandleManager::Instance() {
  static TraceHandleManager trace_handle_manager;
  return trace_handle_manager;
}

Status TraceHandleManager::Initialize() {
  if (is_init_) {
    FE_LOGD("Trace handle manager has already been initialized.");
    return SUCCESS;
  }

  global_handle_ = AtraceCreate(TracerType::TRACER_TYPE_SCHEDULE, kGlobalTraceHandleName);
  if (global_handle_ < 0) {
    FE_LOGW("Trace handle [%s] does not been created.", kGlobalTraceHandleName);
  }
  statistics_handle_ = AtraceCreate(TracerType::TRACER_TYPE_SCHEDULE, kStatisticsTraceHandleName);
  if (statistics_handle_ < 0) {
    FE_LOGW("Trace handle [%s] does not been created.", kStatisticsTraceHandleName);
  }

  finalize_event_handle_ = AtraceEventCreate(kFinalizeEventName);
  if (finalize_event_handle_ < 0) {
    FE_LOGW("Failed to create event handle for [%s].", kFinalizeEventName);
  }

  // bind event with handle
  if (finalize_event_handle_ >= 0) {
    if (global_handle_ >= 0) {
      (void)AtraceEventBindTrace(finalize_event_handle_, global_handle_);
    }
    if (statistics_handle_ >= 0) {
      (void)AtraceEventBindTrace(finalize_event_handle_, statistics_handle_);
    }
  }

  is_init_ = true;
  FE_LOGD("Trace handle manager has been initialized successfully.");
  return SUCCESS;
}

void TraceHandleManager::Finalize() {
  if (!is_init_) {
    FE_LOGD("Trace handle manager has not been initialized yet.");
    return;
  }
  SaveAndDestroyTraceHandle();
  is_init_ = false;
  FE_LOGD("Trace handle manager has been finalized successfully.");
}

void TraceHandleManager::SaveAndDestroyTraceHandle() {
  // report event, to save trace bind with this event handle
  (void)AtraceEventReportSync(finalize_event_handle_);
  AtraceEventDestroy(finalize_event_handle_);
  FE_LOGD("Event handle of [%s] has been reported and destroyed.", kFinalizeEventName);
  AtraceDestroy(global_handle_);
  global_handle_ = -1;
  FE_LOGD("Trace handle of [%s] has been destroyed.", kGlobalTraceHandleName);
  AtraceDestroy(statistics_handle_);
  statistics_handle_ = -1;
  FE_LOGD("Trace handle of [%s] has been destroyed.", kStatisticsTraceHandleName);

  std::lock_guard<std::mutex> lock_guard(subgraph_mutex_);
  for (auto iter = subgraph_event_map_.begin(); iter != subgraph_event_map_.end(); iter++) {
    (void)AtraceEventReportSync(iter->second);
    AtraceEventDestroy(iter->second);
    FE_LOGD("Event handle of thread[%lu] has been reported and destroyed.", iter->first);
  }
  subgraph_event_map_.clear();
  for (auto iter = subgraph_handle_map_.begin(); iter != subgraph_handle_map_.end(); iter++) {
    AtraceDestroy(iter->second);
    FE_LOGD("Trace handle of thread[%lu] has been destroyed.", iter->first);
  }
  subgraph_handle_map_.clear();
}

void TraceHandleManager::AddSubGraphTraceHandle() {
  uint64_t thread_id = GetCurThreadId();
  FE_LOGD("Begin to add trace handle for sub graph[%lu].", thread_id);
  uint64_t remainder = thread_id % kTreadIdReminder;
  std::lock_guard<std::mutex> lock_guard(subgraph_mutex_);
  auto iter = subgraph_handle_map_.find(remainder);
  if (iter != subgraph_handle_map_.end()) {
    return;
  }
  std::string trace_name = kCompileTraceHandlePrefix + std::to_string(remainder);
  TraHandle subgraph_trace_handle = AtraceCreate(TracerType::TRACER_TYPE_SCHEDULE, trace_name.c_str());
  if (subgraph_trace_handle < 0) {
    FE_LOGW("Trace handle [%s] does not been created.", trace_name.c_str());
    return;
  }
  TraEventHandle subgraph_event_handle = AtraceEventCreate(trace_name.c_str());
  if (subgraph_event_handle < 0) {
    FE_LOGW("Event handle [%s] does not been created.", trace_name.c_str());
    return;
  }
  (void)AtraceEventBindTrace(subgraph_event_handle, subgraph_trace_handle);
  subgraph_event_map_.emplace(remainder, subgraph_event_handle);
  subgraph_handle_map_.emplace(remainder, subgraph_trace_handle);
  FE_LOGD("Trace handle[%s] has been created successfully.", trace_name.c_str());
}

void TraceHandleManager::SubmitGlobalTrace(const std::string &trace_msg) const {
  (void)SubmitTrace(global_handle_, trace_msg);
}

void TraceHandleManager::SubmitGlobalTrace(const TraceMsgBasePtr &trace_msg) const {
  if (trace_msg == nullptr) {
    return;
  }
  (void)SubmitTrace(global_handle_, trace_msg->GenerateTraceMsg());
}

void TraceHandleManager::SubmitStatisticsTrace(const std::string &trace_msg) const {
  (void)SubmitTrace(statistics_handle_, trace_msg);
}

bool TraceHandleManager::SubmitTrace(const TraHandle &trace_handle, const std::string &trace_msg) {
  if (trace_handle < 0 || trace_msg.empty()) {
    return true;
  }
  uint32_t msg_size = static_cast<uint32_t>(trace_msg.size());
  const void *buffer = reinterpret_cast<const void *>(trace_msg.data());
  uint32_t buf_size = msg_size > DEFAULT_ATRACE_MSG_SIZE ? DEFAULT_ATRACE_MSG_SIZE : msg_size;
  TraStatus trace_status = AtraceSubmit(trace_handle, buffer, buf_size);
  return trace_status == TRACE_SUCCESS;
}
}
