/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dirent.h>
#include <gtest/gtest.h>
#include <fstream>
#include <map>
#include <string>

#include "macro_utils/dt_public_scope.h"
#include "common/profiling/profiling_init.h"
#include "graph/ge_local_context.h"
#include "graph/manager/graph_manager.h"
#include "common/profiling/profiling_properties.h"
#include "mmpa/mmpa_api.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;
using namespace std;

namespace ge {
class UtestGeProfilingInit : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestGeProfilingInit, Init) {
  std::map<std::string, std::string> options;
  auto &profiling_init = ge::ProfilingInit::Instance();
  EXPECT_EQ(profiling_init.Init(options), ge::SUCCESS);  // default init

  const char_t *const kEnvValue = "MS_PROF_INIT_FAIL";
  mmSetEnv(kEnvValue, "mock_fail", 1);                   // MsprofInit failed
  EXPECT_NE(profiling_init.Init(options), ge::SUCCESS);  // default init fail
  unsetenv(kEnvValue);

  options[ge::OPTION_EXEC_PROFILING_MODE] = "1";
  options[ge::OPTION_EXEC_PROFILING_OPTIONS] = "profiling";
  EXPECT_EQ(profiling_init.Init(options), PARAM_INVALID);
}

TEST_F(UtestGeProfilingInit, test_init) {
  std::map<std::string, std::string> options;
  options[ge::OPTION_EXEC_PROFILING_MODE] = "1";
  options[ge::OPTION_EXEC_PROFILING_OPTIONS] = R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})";
  options[ge::OPTION_EXEC_JOB_ID] = "0";

  auto &profiling_init = ge::ProfilingInit::Instance();
  auto ret = profiling_init.Init(options);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilingInit, test_init_failed) {
  std::map<std::string, std::string> options;
  options[ge::OPTION_EXEC_PROFILING_MODE] = "1";
  options[ge::OPTION_EXEC_PROFILING_OPTIONS] = R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})";
  options[ge::OPTION_EXEC_JOB_ID] = "0";

  const char_t * const kEnvValue = "MS_PROF_INIT_FAIL";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  auto &profiling_init = ge::ProfilingInit::Instance();
  auto ret = profiling_init.Init(options);
  EXPECT_NE(ret, ge::SUCCESS);
  unsetenv(kEnvValue);
}

TEST_F(UtestGeProfilingInit, test_shut) {
  auto &prof_properties = ge::ProfilingProperties::Instance();
  prof_properties.SetExecuteProfiling(true);
  prof_properties.SetLoadProfiling(true);
  auto profiling_on = prof_properties.ProfilingOn();
  EXPECT_TRUE(profiling_on);
  auto &profiling_init = ge::ProfilingInit::Instance();
  profiling_init.ShutDownProfiling();

  const char_t * const kEnvValue = "MS_PROF_FINALIZE_FAIL";
  mmSetEnv(kEnvValue, "mock_fail", 1);
  profiling_init.ShutDownProfiling();
  unsetenv(kEnvValue);
}

TEST_F(UtestGeProfilingInit, test_set_deviceId) {
  uint32_t model_id = 0;
  uint32_t device_id = 0;
  auto &profiling_init = ge::ProfilingInit::Instance();
  auto ret = profiling_init.SetDeviceIdByModelId(model_id, device_id);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGeProfilingInit, test_unset_deviceId) {
  uint32_t model_id = 0;
  uint32_t device_id = 0;
  auto &profiling_init = ge::ProfilingInit::Instance();
  auto ret = profiling_init.UnsetDeviceIdByModelId(model_id, device_id);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGeProfilingInit, InitProfOptions) {
  Status ret;
  struct MsprofOptions prof_conf = {{ 0 }};
  bool is_execute_profiling = false;
  unsetenv("PROFILING_MODE");
  unsetenv("PROFILING_OPTIONS");

  std::map<std::string, std::string> options;
  ret = ProfilingInit::Instance().InitProfOptions(options, prof_conf, is_execute_profiling);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(is_execute_profiling == false);

  options[ge::OPTION_EXEC_PROFILING_MODE] = "1";
  ret = ProfilingInit::Instance().InitProfOptions(options, prof_conf, is_execute_profiling);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(is_execute_profiling == false);

  options[ge::OPTION_EXEC_PROFILING_OPTIONS] = "";
  ret = ProfilingInit::Instance().InitProfOptions(options, prof_conf, is_execute_profiling);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(is_execute_profiling == false);

  options[ge::OPTION_EXEC_PROFILING_OPTIONS] = "invalid options";
  ret = ProfilingInit::Instance().InitProfOptions(options, prof_conf, is_execute_profiling);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  EXPECT_TRUE(is_execute_profiling == false);

  const char* defaultProfOptions = R"({"output":"./","training_trace":"on","task_trace":"on",
  "hccl":"on","aicpu":"on","aic_metrics":"PipeUtilization","msproftx":"off"})";

  options[ge::OPTION_EXEC_PROFILING_OPTIONS] = defaultProfOptions;
  ret = ProfilingInit::Instance().InitProfOptions(options, prof_conf, is_execute_profiling);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(strcmp(prof_conf.options, defaultProfOptions), 0);
  EXPECT_EQ(is_execute_profiling, true);

  // enable prof from env
  is_execute_profiling = false;
  struct MsprofOptions prof_conf2 = {{ 0 }};
  options[ge::OPTION_EXEC_PROFILING_MODE] = "";
  char_t npu_collect_path[MMPA_MAX_PATH] = "true";
  mmSetEnv("PROFILING_MODE", &npu_collect_path[0U], MMPA_MAX_PATH);
  ret = ProfilingInit::Instance().InitProfOptions(options, prof_conf2, is_execute_profiling);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(is_execute_profiling, true);

  unsetenv("PROFILING_MODE");
}
}  // namespace ge
