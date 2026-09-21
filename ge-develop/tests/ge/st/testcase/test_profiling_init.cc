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

#include "macro_utils/dt_public_scope.h"
#include "common/profiling/profiling_init.h"
#include "common/profiling/profiling_manager.h"
#include "graph/ge_local_context.h"
#include "graph/manager/graph_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/davinci_model.h"
#include "common/profiling/profiling_properties.h"
#include "common/profiling/command_handle.h"
#include "mmpa/mmpa_api.h"
#include "macro_utils/dt_public_unscope.h"

#include "depends/runtime/src/runtime_stub.h"

namespace ge {
class ProfilingInitTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(ProfilingInitTest, test_init_default) {
  auto &profiling_init = ge::ProfilingInit::Instance();
  std::map<std::string, std::string> options;
  EXPECT_EQ(profiling_init.Init(options), ge::SUCCESS);  // prof not enabled

  const char_t *const kEnvValue = "MS_PROF_INIT_FAIL";
  mmSetEnv(kEnvValue, "mock_fail", 1);
  EXPECT_NE(profiling_init.Init(options), ge::SUCCESS);  // MsprofInit failed
  unsetenv(kEnvValue);
}

TEST_F(ProfilingInitTest, test_shut) {
  auto &prof_properties = ge::ProfilingProperties::Instance();
  prof_properties.SetExecuteProfiling(true);
  prof_properties.SetLoadProfiling(true);
  prof_properties.SetIfInited(true);
  auto profiling_on = prof_properties.ProfilingOn();
  EXPECT_TRUE(profiling_on);
  char_t npu_collect_path[MMPA_MAX_PATH] = "true";
  mmSetEnv("PROFILING_MODE", &npu_collect_path[0U], MMPA_MAX_PATH);

  std::map<std::string, std::string> options;
  options[ge::OPTION_EXEC_PROFILING_MODE] = "1";
  options[ge::OPTION_EXEC_PROFILING_OPTIONS] = R"({"result_path":"/data/profiling","training_trace":"on",
  "task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})";
  options[ge::OPTION_EXEC_JOB_ID] = "0";

  auto &profiling_init = ge::ProfilingInit::Instance();
  auto ret = profiling_init.Init(options);
  EXPECT_EQ(ret, ge::SUCCESS);
  profiling_init.ShutDownProfiling();

  // MsprofFinalize fail
  const char_t * const kEnvValue = "MS_PROF_FINALIZE_FAIL";
  mmSetEnv(kEnvValue, "mock_fail", 1);
  profiling_init.ShutDownProfiling();

  unsetenv(kEnvValue);
  unsetenv("PROFILING_MODE");
}

TEST_F(ProfilingInitTest, test_init_failed) {
  std::map<std::string, std::string> options;
  options[ge::OPTION_EXEC_PROFILING_MODE] = "1";
  options[ge::OPTION_EXEC_PROFILING_OPTIONS] = R"({"result_path":"/data/profiling","training_trace":"on",
  "task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})";
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

TEST_F(ProfilingInitTest, HandleCtrlSetStepInfo_ReportOk) {
  ProfStepInfoCmd_t info{1, 1, (void *)0x01};
  uint32_t prof_type = PROF_CTRL_STEPINFO;
  Status ret = ProfCtrlHandle(prof_type, &info, sizeof(info));
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(ProfilingInitTest, ProfOpDetailProfiling_Ok) {
  const uint64_t module = 0x0000080000000ULL;
  std::map<std::string, std::string> config_para;
  config_para["devNums"] = "4";
  config_para["devIdList"] = "0,1,2,8";
  auto shared_model = MakeShared<DavinciModel>(0, nullptr);
  uint32_t davinci_model_id = 1U;
  ModelManager::GetInstance().InsertModel(davinci_model_id, shared_model);
  ProfilingManager::Instance().RecordLoadedModelId(davinci_model_id);
  Status ret = ProfilingManager::Instance().ProfStartProfiling(module, config_para, 1);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(davinci_model_id), SUCCESS);
}
}  // namespace ge
