/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_context.h"
#include <stdexcept>
#include "graph/ge_global_options.h"
#include "graph/ge_local_context.h"
#include "graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "utils/extern_math_util.h"
#include "external/ge_common/ge_api_types.h"

namespace ge {
namespace {
const int32_t kDecimal = 10;
const char_t *kHostExecPlacement = "HOST";
const char_t *kEnabled = "1";

template<class T>
ge::Status GetOptionValue(const std::string &option_name, T &var) {
  std::string option;
  if (ge::GetContext().GetOption(option_name, option) != GRAPH_SUCCESS) {
    return ge::FAILED;
  }

  int64_t value = 0;
  try {
    value = static_cast<int64_t>(std::stoi(option.c_str()));
  } catch (std::invalid_argument &) {
    GELOGW("[Init] Transform option %s %s to int failed, as catching invalid_argument exception", option_name.c_str(),
           option.c_str());
    return ge::FAILED;
  } catch (std::out_of_range &) {
    GELOGW("[Init] Transform option %s %s to int failed, as catching out_of_range exception", option_name.c_str(),
           option.c_str());
    return ge::FAILED;
  }
  if (!IntegerChecker<T>::Compat(value)) {
    GELOGW("[Init] Transform option %s %s to int failed, value is invalid_argument", option_name.c_str(),
           option.c_str());
    return ge::FAILED;
  }
  var = value;
  return ge::SUCCESS;
}
}  // namespace

GEContext &GetContext() {
  static GEContext ge_context {};
  return ge_context;
}

thread_local uint64_t GEContext::session_id_ = 0UL;
thread_local uint64_t GEContext::context_id_ = 0UL;

graphStatus GEContext::GetOption(const std::string &key, std::string &option) {
  return GetThreadLocalContext().GetOption(key, option);
}

const std::string &GEContext::GetReadableName(const std::string &key) {
  return GetThreadLocalContext().GetReadableName(key);
}

bool GEContext::IsOverflowDetectionOpen() const {
  std::string enable_overflow_detection;
  if (GetThreadLocalContext().GetOption("ge.exec.overflow", enable_overflow_detection) != GRAPH_SUCCESS) {
    return false;
  }
  GELOGD("Option ge.exec.overflow is %s.", enable_overflow_detection.c_str());
  return (enable_overflow_detection == kEnabled);
}

bool GEContext::IsGraphLevelSat() const {
  std::string graph_level_sat;
  if (GetThreadLocalContext().GetOption("ge.graphLevelSat", graph_level_sat) != GRAPH_SUCCESS) {
    return false;
  }
  GELOGD("Option ge.graphLevelSat is %s.", graph_level_sat.c_str());
  return (graph_level_sat == kEnabled);
}

bool GEContext::GetHostExecFlag() const {
  std::string exec_placement;
  if (GetThreadLocalContext().GetOption("ge.exec.placement", exec_placement) != GRAPH_SUCCESS) {
    return false;
  }
  GELOGD("Option ge.exec.placement is %s.", exec_placement.c_str());
  return exec_placement == kHostExecPlacement;
}

bool GEContext::GetTrainGraphFlag() const {
  std::string run_mode;
  if ((GetThreadLocalContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == ge::GRAPH_SUCCESS) &&
      (!run_mode.empty())) {
    if (static_cast<ge::GraphRunMode>(std::strtol(run_mode.c_str(), nullptr, kDecimal)) >= ge::TRAIN) {
      return true;
    }
  }
  return false;
}

uint64_t GEContext::GetInputFusionSize() const {
  const uint64_t default_fusion_size = 128 * 1024U;    // 128KB
  const uint64_t max_fusion_size = 32 * 1024 * 1024U;  // 32MB

  std::string fusion_size;
  if (GetThreadLocalContext().GetOption(OPTION_EXEC_INPUT_FUSION_SIZE, fusion_size) != GRAPH_SUCCESS) {
    return default_fusion_size;
  }

  long value = std::strtol(fusion_size.c_str(), nullptr, kDecimal);
  if (value < 0) {
    GELOGI("%s is %s which is less than 0, return 0", OPTION_EXEC_INPUT_FUSION_SIZE, fusion_size.c_str());
    return 0U;
  }

  uint64_t result = static_cast<uint64_t>(value);
  if (result > max_fusion_size) {
    GELOGW("option [%s] is %s which is bigger than max(%" PRIu64 "), return max", OPTION_EXEC_INPUT_FUSION_SIZE,
           fusion_size.c_str(), max_fusion_size);
    return max_fusion_size;
  }
  return result;
}

std::mutex &GetGlobalOptionsMutex() {
  static std::mutex global_options_mutex;
  return global_options_mutex;
}

std::map<std::string, std::string> &GetMutableGlobalOptions() {
  static std::map<std::string, std::string> context_global_options{};
  return context_global_options;
}

void GEContext::Init() {
  (void) GetOptionValue("ge.exec.sessionId", session_id_);
  (void) GetOptionValue("ge.exec.deviceId", device_id_);

  int32_t stream_sync_timeout = -1;
  (void) GetOptionValue("stream_sync_timeout", stream_sync_timeout);
  SetStreamSyncTimeout(stream_sync_timeout);

  int32_t event_sync_timeout = -1;
  (void) GetOptionValue("event_sync_timeout", event_sync_timeout);
  SetEventSyncTimeout(event_sync_timeout);
}

uint64_t GEContext::SessionId() const { return session_id_; }

uint32_t GEContext::DeviceId() const {
  uint32_t device_id = 0U;
  // session device id has priority
  auto status = GetOptionValue("ge.session_device_id", device_id);
  return (status == ge::SUCCESS) ? device_id : device_id_;
}

int32_t GEContext::StreamSyncTimeout() const { return GetThreadLocalContext().StreamSyncTimeout(); }

int32_t GEContext::EventSyncTimeout() const { return GetThreadLocalContext().EventSyncTimeout(); }

void GEContext::SetSessionId(const uint64_t session_id) { session_id_ = session_id; }

void GEContext::SetContextId(const uint64_t context_id) { context_id_ = context_id; }

void GEContext::SetCtxDeviceId(const uint32_t device_id) { device_id_ = device_id; }

void GEContext::SetStreamSyncTimeout(const int32_t timeout) { GetThreadLocalContext().SetStreamSyncTimeout(timeout); }

void GEContext::SetEventSyncTimeout(const int32_t timeout) { GetThreadLocalContext().SetEventSyncTimeout(timeout); }

graphStatus GEContext::SetOptionNameMap(const std::string &option_name_map_json) {
  return GetThreadLocalContext().SetOptionNameMap(option_name_map_json);
}

OptimizationOption &GEContext::GetOo() const {
  return GetThreadLocalContext().GetOo();
}
}  // namespace ge
