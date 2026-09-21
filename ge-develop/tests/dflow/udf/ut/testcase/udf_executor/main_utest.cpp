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
#define private public
#include "config/global_config.h"
#undef private
#include "dlog_pub.h"

extern int32_t FlowFuncTestMain(int32_t argc, char *argv[]);

namespace FlowFunc {

class MainUTest : public testing::Test {
 protected:
  static void TearDownTestCase() {
    // reset device info
    GlobalConfig::Instance().SetPhyDeviceId(-1);
    GlobalConfig::Instance().SetDeviceId(0);
  }
  virtual void SetUp() {
    back_is_on_device_ = GlobalConfig::Instance().IsOnDevice();
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
    GlobalConfig::on_device_ = back_is_on_device_;
  }

 private:
  bool back_is_on_device_ = false;
};

int DlogSetAttrStub(LogAttr log_attr) {
  return -1;
}

TEST_F(MainUTest, miss_device_id) {
  char process_name[] = "flow_func_executor";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";
  char param_base_dir[] = "--base_dir=";
  char paramPhyDeviceId[] = "--phy_device_id=-1";

  char *argv[] = {process_name, param_load_path_ok, param_grp_name_ok, param_base_dir, paramPhyDeviceId};
  int32_t argc = 5;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainUTest, device_id_invalid) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1d";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainUTest, device_id_over_maX) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=65535";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainUTest, miss_load_path) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_grp_name_ok[] = "--group_name=Grp1";

  char *argv[] = {process_name, param_device_id_ok, param_grp_name_ok};
  int32_t argc = 3;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainUTest, miss_group_name) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok};
  int32_t argc = 3;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainUTest, load_model_failed) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_load_path_ok[] = "--load_path=./model/not_exit_batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
}

TEST_F(MainUTest, phy_device_id_invalid) {
  char process_name[] = "flow_func_executor";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";
  char param_base_dir[] = "--base_dir=";
  char param_phy_device_id_less_than_min[] = "--phy_device_id=-2";
  char param_device_id_ok[] = "--device_id=1";

  char *argv[] = {
      process_name,      param_load_path_ok, param_grp_name_ok, param_base_dir, param_phy_device_id_less_than_min,
      param_device_id_ok};
  int32_t argc = 6;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
  char param_phy_device_id_invalid[] = "--phy_device_id=1d";
  char *argv1[] = {process_name,   param_load_path_ok,          param_grp_name_ok,
                   param_base_dir, param_phy_device_id_invalid, param_device_id_ok};
  ret = FlowFuncTestMain(argc, argv1);
  EXPECT_NE(ret, 0);
}

TEST_F(MainUTest, npu_sched_invalid) {
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char paramGrpNameInvalid0[] = "--npu_sched=Grp1";
  char paramGrpNameInvalid1[] = "--npu_sched=2";

  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, paramGrpNameInvalid0};
  int32_t argc = 4;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);

  char *argv1[] = {process_name, param_device_id_ok, param_load_path_ok, paramGrpNameInvalid1};
  int32_t argc1 = 4;
  int32_t ret1 = FlowFuncTestMain(argc1, argv1);
  EXPECT_NE(ret1, 0);
}
}  // namespace FlowFunc