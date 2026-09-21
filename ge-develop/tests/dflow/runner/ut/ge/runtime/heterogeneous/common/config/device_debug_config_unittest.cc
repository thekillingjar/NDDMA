/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <map>
#include <stdlib.h>

#include <unistd.h>
#include "stdlib.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "common/profiling/profiling_properties.h"
#include "common/ge_inner_attrs.h"

#include "macro_utils/dt_public_scope.h"
#include "common/config/device_debug_config.h"
#include "executor/engine_daemon.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
class UtDeviceDebugConfig : public testing::Test {
 public:
  UtDeviceDebugConfig() {}
 protected:
  void SetUp() override {
    GEContext().Init();
  }
  void TearDown() override {
    DeviceDebugConfig::global_configs_.clear();
  }
};
namespace {
// log name
const std::string kLogLevelEnvName = "ASCEND_GLOBAL_LOG_LEVEL";
const std::string kLogEventEnableEnvName = "ASCEND_GLOBAL_EVENT_ENABLE";
const std::string kLogHostFileNumEnvName = "ASCEND_HOST_LOG_FILE_NUM";
const std::vector<std::string> kLogEnvNames = {kLogLevelEnvName, kLogEventEnableEnvName, kLogHostFileNumEnvName};
const char_t *const kDumpoff = "off";
const char_t *const kDumpOn = "on";
constexpr const char_t *kExceptionDumpPath = "NPU_COLLECT_PATH_EXE";
}

TEST_F(UtDeviceDebugConfig, DeviceDebugConfigMasterLogEncodeDecode) {
  // init log env
  int32_t env_val = 1;
  int32_t expect_env_val = 1;
  for (const auto &env_name : kLogEnvNames) {
    setenv(env_name.c_str(), std::to_string(env_val).c_str(), 1);
    env_val++;
  }
  DeviceMaintenanceMasterCfg dev_cfg;
  Status ret = dev_cfg.InitGlobalMaintenanceConfigs();
  EXPECT_EQ(ret, SUCCESS);
  // get json data
  std::string data;
  ret = dev_cfg.GetJsonDataByType(DeviceDebugConfig::ConfigType::kLogConfigType, data);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(data.size() > 0, true);
  // load json data
  std::unique_ptr<DeviceMaintenanceClientCfg> dev_client_cfg = MakeUnique<DeviceMaintenanceClientCfg>();
  ret = dev_client_cfg->LoadJsonData(data);
  EXPECT_EQ(ret, SUCCESS);
  // decode
  std::map<std::string, std::string> env_option;
  std::map<std::string, std::string> args_option;
  ret = dev_client_cfg->DecodeConfig(env_option, args_option);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(args_option.empty(), true);
  EXPECT_EQ(env_option.size(), kLogEnvNames.size());
  for (size_t id = 0; id < kLogEnvNames.size(); id++) {
    EXPECT_EQ(env_option[kLogEnvNames[id]] ==  std::to_string(expect_env_val++), true);
  }

  for (const auto &env_name : kLogEnvNames) {
    unsetenv(env_name.c_str());
    env_val++;
  }
}

TEST_F(UtDeviceDebugConfig, DeviceDebugConfigMasterDumpEncodeDecode) {
  // init dump config
  DeviceMaintenanceMasterCfg dev_cfg;
  DumpConfig conf;
  conf.dump_status = kDumpOn;
  conf.dump_debug = kDumpoff;
  conf.dump_step = "0|2-5|8";
  conf.dump_path = "/var/log/npu/dump/";
  conf.dump_mode = "2";
  auto ret = DumpManager::GetInstance().SetDumpConf(conf);
  EXPECT_EQ(ret, SUCCESS);
  dev_cfg.InitDevMaintenanceConfigs();
  // get json data
  std::string data;
  ret = dev_cfg.GetJsonDataByType(DeviceDebugConfig::ConfigType::kDumpConfigType, data);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(data.size() > 0, true);
  // load json data
  DeviceMaintenanceClientCfg dev_client_cfg;
  ret = dev_client_cfg.LoadJsonData(data);
  EXPECT_EQ(ret, SUCCESS);
  // decode
  std::map<std::string, std::string> env_option;
  std::map<std::string, std::string> args_option;
  ret = dev_client_cfg.DecodeConfig(env_option, args_option);
  EXPECT_EQ(ret, SUCCESS);
  constexpr uint32_t kExpectDumpArgsNum = 6U;
  const char_t *const kHostMasterPidName = "HOST_MASTER_PID";
  ASSERT_EQ(args_option.size(), kExpectDumpArgsNum);
  EXPECT_EQ(args_option[OPTION_EXEC_ENABLE_DUMP].empty(), true);
  EXPECT_EQ(args_option[OPTION_EXEC_ENABLE_DUMP_DEBUG].empty(), true);
  EXPECT_EQ(args_option[OPTION_EXEC_DUMP_STEP] == conf.dump_step, true);
  EXPECT_EQ(args_option[kHostMasterPidName] == std::to_string(getpid()), true);
  EXPECT_EQ(args_option[OPTION_EXEC_DUMP_MODE] ==  conf.dump_mode, true);
}

TEST_F(UtDeviceDebugConfig, DeviceDebugConfigMasterDumpEncodeDecodeExceptionDump) {
  // init dump config
  DeviceMaintenanceMasterCfg dev_cfg;
  DumpConfig conf;
  setenv(kExceptionDumpPath, "/var/log/npu/dump/", 0);
  conf.dump_status = kDumpoff;
  conf.dump_debug = kDumpoff;
  auto ret = DumpManager::GetInstance().SetDumpConf(conf);
  EXPECT_EQ(ret, SUCCESS);
  dev_cfg.InitDevMaintenanceConfigs();
  // get json data
  std::string data;
  ret = dev_cfg.GetJsonDataByType(DeviceDebugConfig::ConfigType::kDumpConfigType, data);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(data.size() > 0, true);
  // load json data
  DeviceMaintenanceClientCfg dev_client_cfg;
  ret = dev_client_cfg.LoadJsonData(data);
  EXPECT_EQ(ret, SUCCESS);
  // decode
  std::map<std::string, std::string> env_option;
  std::map<std::string, std::string> args_option;
  ret = dev_client_cfg.DecodeConfig(env_option, args_option);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(env_option.size(), 1);
  EXPECT_EQ(env_option[kExceptionDumpPath], "/var/log/npu/dump/");
  EXPECT_EQ(args_option[OPTION_EXEC_ENABLE_DUMP].empty(), true);
  EXPECT_EQ(args_option[OPTION_EXEC_ENABLE_DUMP_DEBUG].empty(), true);
  unsetenv(kExceptionDumpPath);
}

TEST_F(UtDeviceDebugConfig, DeviceDebugConfigProfilingEncodeDecode) {
  // init profiling config
  DeviceMaintenanceMasterCfg dev_cfg;
  std::vector<int32_t> device_list = {0U};
  std::string config_data = "test_config_data";
  ProfilingProperties::Instance().UpdateDeviceIdCommandParams(config_data);
  dev_cfg.InitDevMaintenanceConfigs();
  // get json data
  std::string data;
  auto ret = dev_cfg.GetJsonDataByType(DeviceDebugConfig::ConfigType::kProfilingConfigType, data);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(data.size() > 0, true);
  // load json data
  DeviceMaintenanceClientCfg dev_client_cfg;
  ret = dev_client_cfg.LoadJsonData(data);
  EXPECT_EQ(ret, SUCCESS);
  // decode
  std::map<std::string, std::string> env_option;
  std::map<std::string, std::string> args_option;
  ret = dev_client_cfg.DecodeConfig(env_option, args_option);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(env_option.empty(), true);
  constexpr uint32_t kExpectDumpArgsNum = 2U;
  ASSERT_EQ(args_option.size(), kExpectDumpArgsNum);
  EXPECT_EQ(args_option[kProfilingDeviceConfigData] == config_data, true);
}

TEST_F(UtDeviceDebugConfig, DeviceDebugConfigUtilsEncodeDecode) {
  std::map<std::string, std::string> env_option = {
      {kLogLevelEnvName, "1"},
      {kLogEventEnableEnvName, "2"},
      {kLogHostFileNumEnvName, "3"},
      {"InvalidEnv", "4"},
  };
  std::map<std::string, std::string> args_option = {
      {OPTION_EXEC_ENABLE_DUMP, "0"},
      {OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"},
      {kHostMasterPidName, "2"},
      {OPTION_EXEC_DUMP_STEP, "3"},
      {OPTION_EXEC_DUMP_MODE, "4"},
  };
  setenv(kLogLevelEnvName.c_str(), env_option[kLogLevelEnvName].c_str(), 1);
  setenv(kLogEventEnableEnvName.c_str(), env_option[kLogEventEnableEnvName].c_str(), 1);
  setenv(kLogHostFileNumEnvName.c_str(), env_option[kLogHostFileNumEnvName].c_str(), 1);

  char_t * level_val = std::getenv(kLogLevelEnvName.c_str());
  EXPECT_EQ(atoi(level_val), 1);
  char_t * log_enable = std::getenv(kLogEventEnableEnvName.c_str());
  EXPECT_EQ(atoi(log_enable), 2);
  char_t * log_host_file_num = std::getenv(kLogHostFileNumEnvName.c_str());
  EXPECT_EQ(atoi(log_host_file_num), 3);

  // set args option
  const auto &args_array = DevMaintenanceCfgUtils::TransArgsOption2Array(args_option);
  // variable args

  constexpr size_t kMaxArgsSize = 128UL;
  const char_t *argv_var[kMaxArgsSize] = {nullptr};
  for (size_t var_id = 0UL; var_id < args_array.size(); var_id++) {
    argv_var[var_id] = args_array[var_id].c_str();
  }
  EngineDaemon daemon;
  daemon.TransArray2ArgsOption(0, args_array.size(), const_cast<char_t **>(argv_var));
  std::map<std::string, std::string> options;
  EXPECT_EQ(daemon.args_option_.size(), args_option.size());
  EXPECT_EQ(daemon.args_option_[OPTION_EXEC_ENABLE_DUMP] == args_option[OPTION_EXEC_ENABLE_DUMP], true);
  EXPECT_EQ(daemon.args_option_[OPTION_EXEC_ENABLE_DUMP_DEBUG] == args_option[OPTION_EXEC_ENABLE_DUMP_DEBUG], true);
  EXPECT_EQ(daemon.args_option_[kHostMasterPidName] == args_option[kHostMasterPidName], true);
  EXPECT_EQ(daemon.args_option_[OPTION_EXEC_DUMP_STEP] == args_option[OPTION_EXEC_DUMP_STEP], true);
  EXPECT_EQ(daemon.args_option_[OPTION_EXEC_DUMP_MODE] == args_option[OPTION_EXEC_DUMP_MODE], true);
}

}

