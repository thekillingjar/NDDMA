/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_debug_config.h"
#include <unordered_set>
#include "mmpa/mmpa_api.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/profiling/profiling_properties.h"
#include "graph/ge_context.h"
#include "common/ge_inner_attrs.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr size_t kInteval = 2UL;
constexpr const char_t *kExceptionDumpPath = "NPU_COLLECT_PATH_EXE";
// param type
constexpr char_t const *kEnvType = "env";
constexpr char_t const *kArgsType = "args";
// config type name
constexpr char_t const *kLogConfName = "log";
constexpr char_t const *kDumpConfName = "dump";
constexpr char_t const *kProfilingConfName = "profiling";
const std::vector<std::string> kConfigNames = {kLogConfName, kDumpConfName, kProfilingConfName};
// option type
constexpr char_t const *kTypeName = "type";
constexpr char_t const *kParamListName = "param_list";
// log name
const std::string kLogLevelEnvName = "ASCEND_GLOBAL_LOG_LEVEL";
const std::string kLogModuleLogLevelEnvName = "ASCEND_MODULE_LOG_LEVEL";
const std::string kLogEventEnableEnvName = "ASCEND_GLOBAL_EVENT_ENABLE";
const std::string kLogHostFileNumEnvName = "ASCEND_HOST_LOG_FILE_NUM";
const std::unordered_set<std::string> kLogEnvNames = {kLogLevelEnvName, kLogModuleLogLevelEnvName, kLogEventEnableEnvName,
                                                 kLogHostFileNumEnvName};
// array keys
constexpr char_t const *kKeyName = "key";
constexpr char_t const *kValueName = "value";
constexpr char_t const *kParamTypeName = "param_type";
}
// init static config
nlohmann::json DeviceDebugConfig::global_configs_;

void DeviceMaintenanceMasterCfg::InitProfilingConfig() {
  GELOGI("Generate profiling config json.");
  if (configs_.find(ConfigType::kProfilingConfigType) != configs_.end()) {
    GELOGI("Profiling config has already been inited.");
    return;
  }
  // profiling transfer data/data_len to device
  std::string config_data = ProfilingProperties::Instance().GetDeviceConfigData();
  std::string is_profiling_on = std::to_string(static_cast<int32_t>(
      ProfilingProperties::Instance().IsExecuteProfiling()));
  nlohmann::json js;
  js[kTypeName] = kProfilingConfName;
  auto js_array = nlohmann::json::array();
  std::map<std::string, std::string> profiling_param_list = {
      {kProfilingDeviceConfigData, config_data},
      {kProfilingIsExecuteOn, is_profiling_on}};
  for (const auto &profiling_map : profiling_param_list) {
    nlohmann::json js_param;
    const auto &key = profiling_map.first;
    const auto &value = profiling_map.second;
    js_param[kKeyName] = key;
    js_param[kValueName] = value;
    js_param[kParamTypeName] = std::string(kArgsType);
    js_array.emplace_back(js_param);
    GELOGI("Init profiling_config, key=%s, value=%s successfully", key.c_str(), value.c_str());
  }
  js[kParamListName] = js_array;
  configs_[ConfigType::kProfilingConfigType] = js;
  GELOGI("InitProfilingConfig successfully.");
}

void DeviceMaintenanceMasterCfg::InitDumpConfig() {
  GELOGI("Generate dump config json.");
  if (configs_.find(ConfigType::kDumpConfigType) != configs_.end()) {
    GELOGI("Dump config has already been inited.");
    return;
  }
  const DumpProperties &dump_properties = DumpManager::GetInstance().GetDumpProperties(GetContext().SessionId());
  nlohmann::json js;
  js[kTypeName] = kDumpConfName;
  const char_t *record_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH_EXE, record_path);
  if (record_path == nullptr) {
    MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH, record_path);
  }
  std::map<std::string, std::string> dump_param_list = {
      {OPTION_EXEC_ENABLE_DUMP, dump_properties.GetEnableDump()},
      {OPTION_EXEC_ENABLE_DUMP_DEBUG, dump_properties.GetEnableDumpDebug()},
      {OPTION_EXEC_DUMP_STEP, dump_properties.GetDumpStep()},
      {kHostMasterPidName, std::to_string(mmGetPid())},
      {OPTION_EXEC_DUMP_MODE, dump_properties.GetDumpMode()},
      {OPTION_EXEC_DUMP_PATH, dump_properties.GetDumpPath()}};
  if (record_path != nullptr) {
    dump_param_list.emplace(kExceptionDumpPath, record_path);
  }
  auto js_array = nlohmann::json::array();
  for (const auto &dump_map : dump_param_list) {
    nlohmann::json js_param;
    const auto &key = dump_map.first;
    const auto &value = dump_map.second;
    js_param[kKeyName] = key;
    js_param[kValueName] = value;
    js_param[kParamTypeName] = (key == kExceptionDumpPath) ? std::string(kEnvType) : std::string(kArgsType);
    js_array.emplace_back(js_param);
    GELOGI("Init dump_config key=%s, value=%s successfully.", key.c_str(), value.c_str());
  }
  js[kParamListName] = js_array;
  configs_[ConfigType::kDumpConfigType] = js;
  GELOGI("InitDumpConfig successfully.");
}

Status DeviceMaintenanceMasterCfg::InitLogConfig() {
  GELOGI("Generate log config json.");
  nlohmann::json js;
  js[kTypeName] = kLogConfName;
  auto js_array = nlohmann::json::array();
  for (const auto &log_name : kLogEnvNames) {
    nlohmann::json js_param;
    constexpr size_t kMaxClusterEnvStrLen = 1024UL;
    char_t env_value[kMaxClusterEnvStrLen] = {};
    const int32_t ret = mmGetEnv(log_name.c_str(), env_value, kMaxClusterEnvStrLen);
    if (ret != EN_OK) {
      GELOGW("[Check][Env]Get env[%s] failed.", log_name.c_str());
      continue;
    }
    js_param[kKeyName] = log_name;
    js_param[kValueName] = std::string(env_value);
    js_param[kParamTypeName] = std::string(kEnvType);
    js_array.emplace_back(js_param);
    GELOGI("Init log key=%s, value=%s successfully.", log_name.c_str(), env_value);
  }
  js[kParamListName] = js_array;

  DeviceDebugConfig::global_configs_ = js;
  return SUCCESS;
}

Status DeviceMaintenanceMasterCfg::GetJsonDataByType(ConfigType config_type, std::string &data) const {
  GELOGD("Get device maintenance config json data, config_type=%d.", config_type);
  nlohmann::json js;
  // check global config
  if (config_type == ConfigType::kLogConfigType) {
    if (DeviceDebugConfig::global_configs_.is_null()) {
      GELOGW("Can not find global config");
      return FAILED;
    }
    js = DeviceDebugConfig::global_configs_;
  } else {
    // check device config
    if (configs_.find(config_type) == configs_.end()) {
      GELOGW("Can not find config of type = %d", static_cast<int32_t>(config_type));
      return FAILED;
    }
    js = configs_[config_type];
  }
  try {
    data = js.dump(kInteval, ' ', false, nlohmann::json::error_handler_t::ignore);
  } catch (std::exception &e) {
    GELOGE(FAILED, "[Convert][config]Failed to convert JSON to string, config type=%d.",
           static_cast<int32_t>(config_type));
    return FAILED;
  }
  return SUCCESS;
}

Status DeviceMaintenanceMasterCfg::InitGlobalMaintenanceConfigs() {
  if (!DeviceDebugConfig::global_configs_.is_null()) {
    GELOGI("Global config has been initialized.");
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(InitLogConfig(), "Failed to init log config.");
  return SUCCESS;
}

void DeviceMaintenanceMasterCfg::InitDevMaintenanceConfigs() {
  InitDumpConfig();
  InitProfilingConfig();
}

Status DeviceMaintenanceClientCfg::LoadJsonData(const std::string &data) {
  nlohmann::json config;
  GELOGI("Parse device maintenance config to json, data size=%zu.", data.size());
  // get json
  try {
    config = nlohmann::json::parse(data);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "[Check][data]Invalid json file, data:%s,exception:%s", data.c_str(), e.what());
    REPORT_PREDEFINED_ERR_MSG("E10032", std::vector<const char *>({"file_name", "reason"}),
                              std::vector<const char *>({data.c_str(), e.what()}));
    return FAILED;
  }
  if (config.is_null()) {
    GELOGW("[Check][Config]The message of config is not found in %s", data.c_str());
    return FAILED;
  }
  std::string type;
  nlohmann::json param_list;
  try {
    type = config[kTypeName];
    param_list = config[kParamListName];
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "[Config][Json]Invalid json config, data:%s,exception:%s", data.c_str(), e.what());
    return FAILED;
  }
  const auto &type_it = std::find(kConfigNames.begin(), kConfigNames.end(), type);
  if (type_it == kConfigNames.end()) {
    GELOGW("[Check][Config]Can not find the type:%s.", type.c_str());
    return FAILED;
  }
  if (!param_list.is_array()) {
    GELOGW("[Check][Config]The message of config is not array in %s", data.c_str());
    return FAILED;
  }
  GELOGI("Parse device maintenance config to json, type=%s.", type.c_str());
  if (type == std::string(kLogConfName)) {
    DeviceDebugConfig::global_configs_[kTypeName] = kLogConfName;
    DeviceDebugConfig::global_configs_[kParamListName] = param_list;
  } else if (type == std::string(kDumpConfName)) {
    nlohmann::json js;
    js[kTypeName] = kDumpConfName;
    js[kParamListName] = param_list;
    configs_[ConfigType::kDumpConfigType] = js;
  } else {
    nlohmann::json js;
    js[kTypeName] = kProfilingConfName;
    js[kParamListName] = param_list;
    configs_[ConfigType::kProfilingConfigType] = js;
  }
  return SUCCESS;
}

Status DevMaintenanceCfgUtils::DecodeJsonConfig(const nlohmann::json &param_list_jsons,
                                                std::map<std::string, std::string> &env_option,
                                                std::map<std::string, std::string> &args_option) {
  if (!param_list_jsons.is_array()) {
    GELOGW("[Check][Config]The message of log_jsons is not array.");
    return FAILED;
  }
  for (size_t i = 0UL; i < param_list_jsons.size(); i++) {
    const std::string key_name = param_list_jsons[i][kKeyName];
    const std::string value_name = param_list_jsons[i][kValueName];
    const std::string param_type = param_list_jsons[i][kParamTypeName];
    if (param_type == kEnvType) {
      env_option.emplace(key_name, value_name);
    } else if (param_type == kArgsType) {
      args_option.emplace(key_name, value_name);
    } else {
      GELOGW("param_type[%s] is invalid.", param_type.c_str());
    }
  }
  GELOGI("DecodeJsonConfig successfully, env option size=%zu, args option size=%zu.", env_option.size(),
         args_option.size());
  return SUCCESS;
}

Status DeviceMaintenanceClientCfg::DecodeLogConfig(std::map<std::string, std::string> &env_option,
                                                   std::map<std::string, std::string> &args_option) {
  GELOGI("DecodeLogConfig enter.");
  if (DeviceDebugConfig::global_configs_.is_null()) {
    GELOGW("[Check][Config]The message of global config is null.");
    return SUCCESS;
  }
  // check type name
  const auto &it = DeviceDebugConfig::global_configs_.find(kTypeName);
  if (it == DeviceDebugConfig::global_configs_.end()) {
    GELOGW("[Check][Config]The message of global config is invalid.");
    // has no config, return SUCCESS
    return SUCCESS;
  }
  const std::string &type_name = it.value();
  if (type_name != std::string(kLogConfName)) {
    GELOGW("[Check][Config]Load log config name failed.");
    return FAILED;
  }
  // get param list array
  Status ret = FAILED;
  try {
    const nlohmann::json &param_list_jsons = DeviceDebugConfig::global_configs_[kParamListName];
    ret = DevMaintenanceCfgUtils::DecodeJsonConfig(param_list_jsons, env_option, args_option);
  } catch (const nlohmann::json::exception &e) {
    GELOGW("[Check][Config]Decode json config failed, exception = %s.", e.what());
  }
  return ret;
}

Status DeviceMaintenanceClientCfg::DecodeDumpConfig(std::map<std::string, std::string> &env_option,
                                                    std::map<std::string, std::string> &args_option) const {
  GELOGI("DecodeDumpConfig enter.");
  if (configs_.empty()) {
    GELOGW("[Check][Config]Device config is empty.");
    return SUCCESS;
  }
  const auto &config = configs_.find(ConfigType::kDumpConfigType);
  if (config == configs_.end()) {
    GELOGW("[Check][Config]The message of dump config is not found in device config");
    // has no config, return SUCCESS
    return SUCCESS;
  }
  // check type name
  const nlohmann::json &dump_config = configs_[ConfigType::kDumpConfigType];
  const auto &it = dump_config.find(kTypeName);
  if (it == dump_config.end()) {
    GELOGW("[Check][Config]The message of config is invalid.");
    return FAILED;
  }
  const std::string &type_name = dump_config[kTypeName];
  if (type_name != std::string(kDumpConfName)) {
    GELOGW("[Check][Config]Load dump config name failed.");
    return FAILED;
  }
  // get param list array
  Status ret = FAILED;
  try {
    const nlohmann::json &param_list_jsons = dump_config[kParamListName];
    ret = DevMaintenanceCfgUtils::DecodeJsonConfig(param_list_jsons, env_option, args_option);
  } catch (const nlohmann::json::exception &e) {
    GELOGW("[Check][Config]Decode json config failed, exception = %s.", e.what());
  }
  return ret;
}

Status DeviceMaintenanceClientCfg::DecodeProfilingConfig(std::map<std::string, std::string> &env_option,
                                                         std::map<std::string, std::string> &args_option) const {
  GELOGI("DecodeProfilingConfig enter.");
  if (configs_.empty()) {
    GELOGW("[Check][Config]Device config is empty.");
    return SUCCESS;
  }
  const auto &config = configs_.find(ConfigType::kProfilingConfigType);
  if (config == configs_.end()) {
    GELOGW("[Check][Config]The message of profiling config is not found in device config");
    // has no config, return SUCCESS
    return SUCCESS;
  }
  // check type name
  const nlohmann::json &profiling_config = configs_[ConfigType::kProfilingConfigType];
  const auto &it = profiling_config.find(kTypeName);
  if (it == profiling_config.end()) {
    GELOGW("[Check][Config]The message of config is invalid.");
    return FAILED;
  }
  const std::string &type_name = profiling_config[kTypeName];
  if (type_name != std::string(kProfilingConfName)) {
    GELOGW("[Check][Config]Load profiling config name failed.");
    return FAILED;
  }
  // get param list array
  Status ret = FAILED;
  try {
    const nlohmann::json &param_list_jsons = profiling_config[kParamListName];
    ret = DevMaintenanceCfgUtils::DecodeJsonConfig(param_list_jsons, env_option, args_option);
  } catch (const nlohmann::json::exception &e) {
    GELOGW("[Check][Config]Decode json config failed, exception = %s.", e.what());
  }
  return ret;
}


Status DeviceMaintenanceClientCfg::DecodeConfig(std::map<std::string, std::string> &env_option,
                                                std::map<std::string, std::string> &args_option) const {
  GE_CHK_STATUS_RET(DecodeLogConfig(env_option, args_option), "Decode log failed.");
  GE_CHK_STATUS_RET(DecodeDumpConfig(env_option, args_option), "Decode dump failed.");
  GE_CHK_STATUS_RET(DecodeProfilingConfig(env_option, args_option), "Decode profiling failed.");
  return SUCCESS;
}

std::vector<std::string> DevMaintenanceCfgUtils::TransArgsOption2Array(
    std::map<std::string, std::string> &args_option) {
  std::vector<std::string> out_args;
  GELOGI("TransArgsOption2Array enter, args option size=%zu.", args_option.size());
  for (const auto &option_it : args_option) {
    const auto &key = option_it.first;
    const auto &value = option_it.second;
    out_args.emplace_back(key + "=" + value);
    GELOGI("arg option=%s.", out_args.back().c_str());
  }
  return out_args;
}

void DevMaintenanceCfgUtils::GetLogEnvs(std::map<std::string, std::string> &log_envs) {
  for (const auto &log_env_key : kLogEnvNames) {
    constexpr size_t kMaxEnvStrLen = 1024UL;
    char_t log_env_value[kMaxEnvStrLen] = {};
    const int32_t ret = mmGetEnv(log_env_key.c_str(), log_env_value, kMaxEnvStrLen);
    if (ret == EN_OK) {
      log_envs[log_env_key] = log_env_value;
      GELOGI("Get log env[%s] success, value = %s.", log_env_key.c_str(), log_env_value);
    }
  }
}

Status DeviceMaintenanceCfgManager::CreateDevMaintenanceConfig(int32_t device_id) {
  std::lock_guard<std::mutex> lk(mu_);
  std::unique_ptr<DeviceMaintenanceMasterCfg> new_conf = MakeUnique<DeviceMaintenanceMasterCfg>();
  GE_CHECK_NOTNULL(new_conf);
  new_conf->InitDevMaintenanceConfigs();
  device_configs_.emplace(device_id, std::move(new_conf));
  return SUCCESS;
}

const DeviceMaintenanceMasterCfg *DeviceMaintenanceCfgManager::GetDevMaintenanceConfig(int32_t dev_id) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto &it = device_configs_.find(dev_id);
  if (it == device_configs_.end()) {
    GELOGW("[Get][Config]Get device config, device id[%d] failed.", dev_id);
    return nullptr;
  }
  return it->second.get();
}

void DeviceMaintenanceCfgManager::CloseDevMaintenanceConfig(int32_t device_id) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto &iter = device_configs_.find(device_id);
  if (iter == device_configs_.end()) {
    GELOGW("Can not find device config, device id[%d].", device_id);
    return;
  }
  
  device_configs_.erase(device_id);
  GELOGI("Successfully close device maintenance config, device id[%d]", device_id);
}
} // namespace ge
