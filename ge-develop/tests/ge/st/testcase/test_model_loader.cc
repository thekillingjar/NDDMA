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
#include "graph/load/model_manager/model_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include "ge/ge_api.h"
#include "graph/compute_graph.h"
#include "framework/common/types.h"
#include "graph/ge_local_context.h"
#include "graph/execute/model_executor.h"
#include "graph/load/graph_loader.h"
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"

#include "ge_graph_dsl/graph_dsl.h"

namespace ge {
namespace {
const int kAippModeDynamic = 2;
}
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICORE, AiCoreNodeExecutor);
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::GE_LOCAL, GeLocalNodeExecutor);
}

class STEST_opt_info : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(STEST_opt_info, record_ts_snapshot) {
  const std::string kTriggerFile = "exec_record_trigger";
  const char_t * const kEnvRecordPath = "NPU_COLLECT_PATH";
  const std::string kRecordFilePrefix = "exec_record_";

  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  mmSetEnv(kEnvRecordPath, &npu_collect_path[0U], 1);

  // 创建trigger文件
  const std::string trigger_file = std::string(&npu_collect_path[0U]) + "/" + kTriggerFile;
  auto trigger_fd = mmOpen(trigger_file.c_str(), M_WRONLY | M_CREAT);
  mmClose(trigger_fd);

  ModelManager *mm = new ModelManager();
  mmSleep(1000U);

  std::string record_file = std::string(&npu_collect_path[0U]) + "/" + kRecordFilePrefix + std::to_string(mmGetPid());
  const auto record_fd = mmOpen(record_file.c_str(), M_RDONLY);
  EXPECT_TRUE(record_fd >= 0);
  mmClose(record_fd);

  delete mm;
  trigger_fd = mmOpen(trigger_file.c_str(), M_RDONLY);
  EXPECT_TRUE(trigger_fd < 0);

  // 清理环境变量
  mmSetEnv(kEnvRecordPath, "", 1);
  // 清理record file
  mmUnlink(record_file.c_str());
}

}  // namespace ge
