/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "mmpa/mmpa_api.h"
#include "utils/udf_utils.h"
#define private public
#include "config/global_config.h"
#include "flow_func/flow_func_config_manager.h"
#undef private
#include "dlog_pub.h"

extern int32_t FlowFuncTestMain(int32_t argc, char *argv[]);

namespace FlowFunc {

class MainSTest : public testing::Test {
 protected:
  static void TearDownTestCase() {
    // reset device info
    GlobalConfig::Instance().SetPhyDeviceId(-1);
    GlobalConfig::Instance().SetDeviceId(0);
    GlobalConfig::Instance().SetRunOnAiCpu(false);
  }
  virtual void SetUp() {}

  virtual void TearDown() {
    GlobalMockObject::verify();
  }
};

TEST_F(MainSTest, miss_device_id) {
  char process_name[] = "flow_func_executor";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";
  char param_base_dir[] = "--base_dir=";
  char param_phy_device_id[] = "--phy_device_id=-1";

  char *argv[] = {process_name, param_load_path_ok, param_grp_name_ok, param_base_dir, param_phy_device_id};
  int32_t argc = 5;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainSTest, device_id_invalid) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1d";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainSTest, device_id_over_maX) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=65535";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainSTest, miss_load_path) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_grp_name_ok[] = "--group_name=Grp1";

  char *argv[] = {process_name, param_device_id_ok, param_grp_name_ok};
  int32_t argc = 3;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainSTest, miss_group_name) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok};
  int32_t argc = 3;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainSTest, load_model_failed) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_load_path_ok[] = "--load_path=./model/not_exit_batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainSTest, phy_device_id_invalid) {
  char process_name[] = "flow_func_executor";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";
  char param_base_dir[] = "--base_dir=";
  char param_phy_device_id[] = "--phy_device_id=-2";
  char param_device_id_ok[] = "--device_id=1";

  char *argv[] = {process_name,   param_load_path_ok,  param_grp_name_ok,
                  param_base_dir, param_phy_device_id, param_device_id_ok};
  int32_t argc = 6;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainSTest, req_queue_id_invalid) {
  ::setenv("ASCEND_HOSTPID", "123", 1);
  ::setenv("ASCEND_LOG_SAVE_MODE", "0", 1);
  char process_name[] = "flow_func_executor";
  char para_req_queue_id_invalid[] = "--req_queue_id=a";
  char param_load_path_ok[] = "--load_path=./model/not_exit_batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  bool back_is_on_device = GlobalConfig::Instance().IsOnDevice();
  GlobalConfig::on_device_ = true;
  char *argv[] = {process_name, para_req_queue_id_invalid, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
  ::unsetenv("ASCEND_HOSTPID");
  ::unsetenv("ASCEND_LOG_SAVE_MODE");

  GlobalConfig::on_device_ = back_is_on_device;
}

TEST_F(MainSTest, rsp_queue_id_invalid) {
  ::setenv("ASCEND_HOSTPID", "123", 1);
  ::setenv("ASCEND_LOG_SAVE_MODE", "0", 1);
  char process_name[] = "flow_func_executor";
  char para_req_queue_id_invalid[] = "--rsp_queue_id=a";
  char param_load_path_ok[] = "--load_path=./model/not_exit_batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  bool back_is_on_device = GlobalConfig::Instance().IsOnDevice();
  GlobalConfig::on_device_ = true;
  char *argv[] = {process_name, para_req_queue_id_invalid, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
  ::unsetenv("ASCEND_HOSTPID");
  ::unsetenv("ASCEND_LOG_SAVE_MODE");

  GlobalConfig::on_device_ = back_is_on_device;
}

TEST_F(MainSTest, init_log_success) {
  ::setenv("ASCEND_HOSTPID", "123", 1);
  ::setenv("ASCEND_LOG_SAVE_MODE", "0", 1);
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_load_path_ok[] = "--load_path=./model/not_exit_batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  bool back_is_on_device = GlobalConfig::Instance().IsOnDevice();
  GlobalConfig::on_device_ = true;
  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
  ::unsetenv("ASCEND_HOSTPID");
  ::unsetenv("ASCEND_LOG_SAVE_MODE");

  GlobalConfig::on_device_ = back_is_on_device;
}

TEST_F(MainSTest, set_flow_func_config_success) {
  FlowFuncConfigManager::config_ = nullptr;
  EXPECT_FALSE(FlowFuncConfigManager::GetConfig()->GetAbnormalStatus());
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetCurrentSchedThreadIdx(), 0);
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetCurrentSchedGroupId(), 0);
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetFlowMsgQueueSchedGroupId(), 0);
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
  GlobalConfig::Instance().SetAbnormalStatus(true);
  GlobalConfig::Instance().SetCurrentSchedThreadIdx(1);
  GlobalConfig::Instance().SetCurrentSchedGroupId(2);
  GlobalConfig::Instance().SetFlowMsgQueueSchedGroupId(3);
  EXPECT_TRUE(FlowFuncConfigManager::GetConfig()->GetAbnormalStatus());
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetCurrentSchedThreadIdx(), 1);
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetCurrentSchedGroupId(), 2);
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetFlowMsgQueueSchedGroupId(), 3);
}
}  // namespace FlowFunc