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
#include <gmock/gmock.h>
#include <vector>
#include "graph/ge_context.h"
#include "nlohmann/json.hpp"

#include "macro_utils/dt_public_scope.h"
#include "api/gelib/gelib.h"
#include "framework/omg/ge_init.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/ge_global_options.h"

using namespace std;
using namespace testing;

namespace ge {
using Json = nlohmann::json;

class UtestGeLib : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestGeLib, Normal) {
    EXPECT_NO_THROW(auto p1 = std::make_shared<GELib>());
}

TEST_F(UtestGeLib, InnerInitialize) {
    auto p1 = std::make_shared<GELib>();
    std::map<std::string, std::string> options;
    p1->init_flag_ = true;
    EXPECT_EQ(p1->InnerInitialize(options), SUCCESS);
}

TEST_F(UtestGeLib, InnerInitialize_aicore_num) {
  auto p1 = std::make_shared<GELib>();
  std::map<std::string, std::string> options;
  options[AICORE_NUM] = "2|2";
  options[SOC_VERSION] = "Ascend910";
  p1->init_flag_ = false;
  EXPECT_EQ(p1->InnerInitialize(options), SUCCESS);

  auto &global_options = GetMutableGlobalOptions();
  auto it = global_options.find(ge::AICORE_NUM);
  EXPECT_NE(it, global_options.end());
  if (it != global_options.end()) {
    EXPECT_EQ(it->second == "2", true);
  }
  string aicore_num_option_str;
  (void)GetThreadLocalContext().GetOption(ge::AICORE_NUM, aicore_num_option_str);
  EXPECT_EQ(aicore_num_option_str == "2", true);
}

TEST_F(UtestGeLib, InnerInitialize_aicore_num_invalid) {
  dlog_setlevel(0, 0, 0);
  auto p1 = std::make_shared<GELib>();
  std::map<std::string, std::string> options;

  // parse failed
  options[AICORE_NUM] = "2|i";
  options[SOC_VERSION] = "Ascend910";
  p1->init_flag_ = false;
  EXPECT_NE(p1->InnerInitialize(options), SUCCESS);

  // wrong format
  options[AICORE_NUM] = "20";
  options[SOC_VERSION] = "Ascend910";
  p1->init_flag_ = false;
  EXPECT_NE(p1->InnerInitialize(options), SUCCESS);

  options[AICORE_NUM] = "2|-2";
  options[SOC_VERSION] = "Ascend910";
  p1->init_flag_ = false;
  EXPECT_NE(p1->InnerInitialize(options), SUCCESS);

  options[AICORE_NUM] = "100|100";
  options[SOC_VERSION] = "Ascend910";
  p1->init_flag_ = false;
  EXPECT_NE(p1->InnerInitialize(options), SUCCESS);

  // not greater than 0
  options[AICORE_NUM] = "0|20";
  options[SOC_VERSION] = "Ascend910";
  p1->init_flag_ = false;
  EXPECT_EQ(p1->InnerInitialize(options), SUCCESS);

  options[AICORE_NUM] = "2a|20";
  options[SOC_VERSION] = "Ascend910";
  p1->init_flag_ = false;
  EXPECT_NE(p1->InnerInitialize(options), SUCCESS);

  // not greater than 0
  options[AICORE_NUM] = "01|20";
  options[SOC_VERSION] = "Ascend910";
  p1->init_flag_ = false;
  EXPECT_NE(p1->InnerInitialize(options), SUCCESS);

  options[AICORE_NUM] = "1|99999999999999999999999";
  options[SOC_VERSION] = "Ascend910";
  p1->init_flag_ = false;
  EXPECT_NE(p1->InnerInitialize(options), SUCCESS);

  dlog_setlevel(0, 3, 0);
}

TEST_F(UtestGeLib, SystemInitialize) {
    auto p1 = std::make_shared<GELib>();
    std::map<std::string, std::string> options;
    options[OPTION_GRAPH_RUN_MODE] = "1";
    p1->is_train_mode_ = true;
    EXPECT_EQ(p1->SystemInitialize(options), SUCCESS);
}

TEST_F(UtestGeLib, SetRTSocVersion) {
    auto p1 = std::make_shared<GELib>();
    std::map<std::string, std::string> new_options;
    EXPECT_EQ(p1->SetRTSocVersion(new_options), SUCCESS);
}

TEST_F(UtestGeLib, GeInitInitialize) {
    std::map<std::string, std::string> options;
    EXPECT_EQ(GEInit::Initialize(options), SUCCESS);
}

TEST_F(UtestGeLib, GeInitFinalize) {
    EXPECT_EQ(GEInit::Finalize(), SUCCESS);
}

TEST_F(UtestGeLib, GeInitGetPath) {
    EXPECT_NE(GEInit::GetPath().size(), 0);
}

TEST_F(UtestGeLib, Finalize) {
	auto p1 = std::make_shared<GELib>();
	p1->init_flag_ = false;
  EXPECT_EQ(p1->Finalize(), SUCCESS);
  p1->init_flag_ = true;
  p1->is_train_mode_ = true;
  EXPECT_EQ(p1->Finalize(), SUCCESS);
}

TEST_F(UtestGeLib, set_valid_SyncTimeout_success) {
  std::map<std::string, std::string> options;
  options.insert({ge::OPTION_EXEC_STREAM_SYNC_TIMEOUT, "100"});
  options.insert({ge::OPTION_EXEC_EVENT_SYNC_TIMEOUT, "200"});
  EXPECT_EQ(GEInit::Initialize(options), SUCCESS);
  EXPECT_EQ(ge::GetContext().StreamSyncTimeout(), 100);
  EXPECT_EQ(ge::GetContext().EventSyncTimeout(), 200);
  EXPECT_EQ(GEInit::Finalize(), SUCCESS);
}

TEST_F(UtestGeLib, set_OutOfRange_SyncTimeout) {
  std::map<std::string, std::string> options;
  options.insert({ge::OPTION_EXEC_STREAM_SYNC_TIMEOUT, "123456789987654321"});
  options.insert({ge::OPTION_EXEC_EVENT_SYNC_TIMEOUT, "123456789987654321"});
  EXPECT_EQ(GEInit::Initialize(options), SUCCESS);
  EXPECT_EQ(ge::GetContext().StreamSyncTimeout(), -1);
  EXPECT_EQ(ge::GetContext().EventSyncTimeout(), -1);
  EXPECT_EQ(GEInit::Finalize(), SUCCESS);
}

TEST_F(UtestGeLib, set_Invalid_SyncTimeout) {
  std::map<std::string, std::string> options;
  options.insert({ge::OPTION_EXEC_STREAM_SYNC_TIMEOUT, "abc"});
  options.insert({ge::OPTION_EXEC_EVENT_SYNC_TIMEOUT, "abc"});
  EXPECT_EQ(GEInit::Initialize(options), SUCCESS);
  EXPECT_EQ(ge::GetContext().StreamSyncTimeout(), -1);
  EXPECT_EQ(ge::GetContext().EventSyncTimeout(), -1);
  EXPECT_EQ(GEInit::Finalize(), SUCCESS);
}

TEST_F(UtestGeLib, set_invalid_OptionNameMap) {
  Json option_name_map;
  option_name_map.emplace("ge.enableSmallChannel", "");
  option_name_map.emplace("", "enable_exception_dump");
  std::map<std::string, std::string> options;
  options.insert({ge::OPTION_NAME_MAP, option_name_map.dump()});
  EXPECT_EQ(GEInit::Initialize(options), SUCCESS);
  EXPECT_EQ(ge::GetContext().GetReadableName("ge.enableSmallChannel"), "ge.enableSmallChannel");
  EXPECT_EQ(ge::GetContext().GetReadableName("ge.exec.enable_exception_dump"), "ge.exec.enable_exception_dump");
  EXPECT_EQ(ge::GetContext().GetReadableName("ge.exec.opWaitTimeout"), "ge.exec.opWaitTimeout");
  EXPECT_EQ(GEInit::Finalize(), SUCCESS);
}

TEST_F(UtestGeLib, set_OptionNameMap) {
  Json option_name_map;
  option_name_map.emplace("ge.enableSmallChannel", "enable_small_channel");
  option_name_map.emplace("ge.exec.enable_exception_dump", "enable_exception_dump");
  option_name_map.emplace("ge.exec.opWaitTimeout", "op_wait_timeout");
  std::map<std::string, std::string> options;
  options.insert({ge::OPTION_NAME_MAP, option_name_map.dump()});
  EXPECT_EQ(GEInit::Initialize(options), SUCCESS);
  EXPECT_EQ(ge::GetContext().GetReadableName("ge.enableSmallChannel"), "enable_small_channel");
  EXPECT_EQ(ge::GetContext().GetReadableName("ge.exec.enable_exception_dump"), "enable_exception_dump");
  EXPECT_EQ(ge::GetContext().GetReadableName("ge.exec.opWaitTimeout"), "op_wait_timeout");
  EXPECT_EQ(GEInit::Finalize(), SUCCESS);
}

} // namespace ge
