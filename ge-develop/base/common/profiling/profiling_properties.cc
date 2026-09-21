/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_properties.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/ge_context.h"
#include "mmpa/mmpa_api.h"
#include "common/ge_inner_attrs.h"
#include "nlohmann/json.hpp"
#include "common/global_variables/diagnose_switch.h"

namespace ge {

namespace {
const std::string kFpPoint = "fp_point";
const std::string kBpPoint = "bp_point";
const std::string kProfilingExecuteOn = "1";
const std::string kProfilingExecuteOff = "0";
const std::string kProfStart = "prof_start";
const std::string kProfStop = "prof_stop";
const std::string kProfModelSubscribe = "prof_model_subscribe";
const std::string kProfModelUnsubscribe = "prof_model_cancel_subscribe";
}  // namespace
ProfilingProperties& ProfilingProperties::Instance() {
  static ProfilingProperties profiling_properties;
  return profiling_properties;
}

void ProfilingProperties::SetLoadProfiling(const bool is_load_profiling) {
  const std::lock_guard<std::mutex> lock(mutex_);
  is_load_profiling_ = is_load_profiling;
}
bool ProfilingProperties::IsLoadProfiling() {
  const std::lock_guard<std::mutex> lock(mutex_);
  return is_load_profiling_;
}

void ProfilingProperties::SetExecuteProfiling(const bool is_exec_profiling) {
  const std::lock_guard<std::mutex> lock(mutex_);
  is_execute_profiling_ = is_exec_profiling;
}

void ProfilingProperties::SetExecuteProfiling(const std::map<std::string, std::string> &options) {
  const std::map<std::string, std::string>::const_iterator &iter = options.find(kProfilingIsExecuteOn);
  if (iter != options.end()) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (iter->second == kProfilingExecuteOn) {
      is_execute_profiling_ = true;
    } else if (iter->second == kProfilingExecuteOff) {
      is_execute_profiling_ = false;
    } else {
      GELOGW("Set execute profiling failed, set value[%s]", iter->second.c_str());
    }
  }
}

bool ProfilingProperties::IsExecuteProfiling() {
  const std::lock_guard<std::mutex> lock(mutex_);
  return is_execute_profiling_;
}

void ProfilingProperties::SetTrainingTrace(const bool is_train_trace) {
  const std::lock_guard<std::mutex> lock(mutex_);
  is_training_trace_ = is_train_trace;
}

void ProfilingProperties::SetOpDetailProfiling(const bool is_op_detail_profiling) {
  is_op_detail_profiling_.store(is_op_detail_profiling);
}

bool ProfilingProperties::IsOpDetailProfiling() {
  return is_op_detail_profiling_.load();
}
bool ProfilingProperties::IsDynamicShapeProfiling() const {
  // current use the same switch with op detail.
  return is_op_detail_profiling_.load();
}
void ProfilingProperties::GetFpBpPoint(std::string &fp_point, std::string &bp_point) {
  // Env or options mode, fp_point_/bp_point_ have initiliazed on profiling init
  const std::lock_guard<std::mutex> lock(mutex_);
  if ((!fp_point_.empty()) && (!bp_point_.empty())) {
    fp_point = fp_point_;
    bp_point = bp_point_;
    GELOGI("Bp Fp have been initialized in env or options. bp_point: %s, fp_point: %s", bp_point.c_str(),
           fp_point.c_str());
    return;
  }
  // ProfApi mode and training trace is set
  // Parse options first
  bool is_profiling_valid = false;
  std::string profiling_options;
  if ((ge::GetContext().GetOption(OPTION_EXEC_PROFILING_OPTIONS, profiling_options) == SUCCESS) &&
      (!profiling_options.empty())) {
    is_profiling_valid = true;
  } else {
    const char_t *env_profiling_options = nullptr;
    MM_SYS_GET_ENV(MM_ENV_PROFILING_OPTIONS, env_profiling_options);
    if (env_profiling_options == nullptr) {
      GELOGI("PROFILING_OPTIONS env does not exist.");
      return;
    }
    GELOGI("Parse env PROFILING_OPTIONS:%s.", env_profiling_options);
    profiling_options = env_profiling_options;
    is_profiling_valid = true;
  }
  if (is_profiling_valid) {
    try {
      const nlohmann::json prof_options = nlohmann::json::parse(profiling_options);
      if (prof_options.contains(kFpPoint)) {
        fp_point_ = prof_options[kFpPoint];
      }
      if (prof_options.contains(kBpPoint)) {
        bp_point_ = prof_options[kBpPoint];
      }
      fp_point = fp_point_;
      bp_point = bp_point_;
      if ((!fp_point_.empty()) && (!bp_point_.empty())) {
        GELOGI("Training trace bp fp is set, bp_point:%s, fp_point:%s.", bp_point_.c_str(), fp_point_.c_str());
      }
    } catch (nlohmann::json::exception &e) {
      GELOGE(ge::FAILED, "Nlohmann json prof options is invalid, catch exception:%s", e.what());
      return;
    }
  }

  return;
}

void ProfilingProperties::SetFpBpPoint(const std::string &fp_point, const std::string &bp_point) {
  const std::lock_guard<std::mutex> lock(mutex_);
  fp_point_ = fp_point;
  bp_point_ = bp_point;
}

void ProfilingProperties::UpdateDeviceIdCommandParams(const std::string &config_data) {
  device_command_params_ = config_data;
}

const std::string &ProfilingProperties::GetDeviceConfigData() const {
  return device_command_params_;
}

void ProfilingProperties::ClearProperties() {
  const std::lock_guard<std::mutex> lock(mutex_);
  diagnoseSwitch::DisableProfiling();
  is_load_profiling_ = false;
  is_op_detail_profiling_.store(false);
  is_execute_profiling_ = false;
  is_training_trace_ = false;
  fp_point_.clear();
  bp_point_.clear();
}

bool ProfilingProperties::IsTrainingModeProfiling() const {
  if (is_load_offline_flag_) {
    return false;
  }
  return domi::GetContext().train_flag;
}

void ProfilingProperties::SetProfilingLoadOfflineFlag(const bool is_load_offline) {
  is_load_offline_flag_ = is_load_offline;
}

void ProfilingProperties::UpdateSubscribeDeviceModuleMap(const std::string &prof_type, const uint32_t device_id,
                                                         const uint64_t module) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (prof_type == kProfModelSubscribe) {
    if (subs_dev_module_.find(device_id) != subs_dev_module_.end()) {
      subs_dev_module_[device_id].subscribe_count++;
    } else {
      DeviceSubsInfo dev_info{};
      dev_info.module = module;
      dev_info.subscribe_count = 1U;
      subs_dev_module_[device_id] = dev_info;
    }
  } else if (prof_type == kProfModelUnsubscribe) {
    const auto iter = subs_dev_module_.find(device_id);
    if (iter != subs_dev_module_.end()) {
      if (iter->second.subscribe_count > 0U) {
        iter->second.subscribe_count--;
      }
      if (iter->second.subscribe_count == 0U) {
        (void)subs_dev_module_.erase(iter);
      }
    }
  } else {
    GELOGI("No need to update device_id module map.");
  }
}

void ProfilingProperties::UpdateDeviceIdModuleMap(const std::string &prof_type, const uint64_t module,
                                                  const std::vector<int32_t> &device_list) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (prof_type == kProfStart) {
    for (size_t i = 0U; i < device_list.size(); i++) {
      const std::map<int32_t, uint64_t>::iterator iter = device_id_module_map_.find(device_list[i]);
      if (iter != device_id_module_map_.end()) {
        const uint64_t prof_on_module = device_id_module_map_[device_list[i]];
        // save all profiling on module of device
        device_id_module_map_[device_list[i]] = prof_on_module | module;
      } else {
        device_id_module_map_[device_list[i]] = module;
      }
    }
  } else if (prof_type == kProfStop) {
    for (size_t i = 0U; i < device_list.size(); i++) {
      const std::map<int32_t, uint64_t>::iterator iter = device_id_module_map_.find(device_list[i]);
      if (iter != device_id_module_map_.end()) {
        const uint64_t prof_on_module = device_id_module_map_[device_list[i]];
        const uint64_t prof_off_module = prof_on_module & module;
        const uint64_t prof_on_left_module = prof_on_module & (~prof_off_module);
        // stop profiling on module of device
        device_id_module_map_[device_list[i]] = prof_on_left_module;
      }
    }
  } else {
    GELOGI("No need to update device_id module map.");
  }
}
bool ProfilingProperties::ProfilingSubscribeOn() const {
  return (subscribe_count_.load() > 0U);
}

void ProfilingProperties::InsertSubscribeGraphId(const uint32_t graph_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  (void)subscribe_graph_id_.insert(graph_id);
}

void ProfilingProperties::RemoveSubscribeGraphId(const uint32_t graph_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  (void)subscribe_graph_id_.erase(graph_id);
}

void ProfilingProperties::ClearDeviceIdMap() {
  const std::lock_guard<std::mutex> lock(mutex_);
  device_id_module_map_.clear();
}

void ProfilingProperties::SetSubscribeInfo(const uint64_t prof_switch, const uint32_t model_id,
                                           const bool is_subscribe) {
  const std::lock_guard<std::mutex> lock(mutex_);
  subscribe_info_.is_subscribe = is_subscribe;
  subscribe_info_.prof_switch = prof_switch;
  subscribe_info_.graph_id = model_id;
}

void ProfilingProperties::CleanSubscribeInfo() {
  const std::lock_guard<std::mutex> lock(mutex_);
  subscribe_info_.is_subscribe = false;
  subscribe_info_.prof_switch = 0U;
  subscribe_info_.graph_id = 0U;
}
}  // namespace ge
