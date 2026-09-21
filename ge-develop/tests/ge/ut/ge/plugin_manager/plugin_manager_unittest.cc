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
#include "common/plugin/ge_make_unique_util.h"
#include "common/plugin/op_tiling_manager.h"
#include "common/plugin/tbe_plugin_manager.h"
#include "common/plugin/runtime_plugin_loader.h"

#include <iostream>

using namespace testing;
using namespace std;

namespace ge {
namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
}
class UtestPluginManager : public testing::Test {
 public:
  OpTilingManager tiling_manager;

 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestPluginManager, op_tiling_manager_clear_handles) {
  auto test_so_name = "test_fail.so";
  tiling_manager.handles_[test_so_name] = nullptr;
  tiling_manager.ClearHandles();
  ASSERT_EQ(tiling_manager.handles_.size(), 0);
  string env_value = "test";
  setenv(kEnvName, env_value.c_str(), 0);
  const char *env = getenv(kEnvName);
  ASSERT_NE(env, nullptr);
}

TEST_F(UtestPluginManager, test_tbe_plugin_manager) {
  TBEPluginManager manager;
  manager.handles_vec_ = {nullptr};
  manager.ClearHandles_();
  string env_value = "test";
  setenv(kEnvName, env_value.c_str(), 0);
  const char *env = getenv(kEnvName);
  ASSERT_NE(env, nullptr);
}

TEST_F(UtestPluginManager, test_load_rt_plugin) {
  const std::string path = "./";
  ge::graphStatus ret = RuntimePluginLoader::GetInstance().Initialize(path);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}
}  // namespace ge
