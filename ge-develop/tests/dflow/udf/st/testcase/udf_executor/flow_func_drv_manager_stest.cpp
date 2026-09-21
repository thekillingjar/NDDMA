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
#include "common/common_define.h"
#include "execute/flow_func_drv_manager.h"
#define private public

#include "config/global_config.h"

#undef private
#include "ascend_hal.h"

namespace FlowFunc {
class FlowFuncDrvManagerSTest : public testing::Test {
 protected:
  void SetUp() override {
    // reset device info
    GlobalConfig::Instance().SetPhyDeviceId(-1);
    GlobalConfig::Instance().SetDeviceId(0);
    back_is_on_device_ = GlobalConfig::Instance().IsOnDevice();
  }

  void TearDown() override {
    GlobalMockObject::verify();

    // reset device info
    GlobalConfig::Instance().SetPhyDeviceId(-1);
    GlobalConfig::Instance().SetDeviceId(0);
    GlobalConfig::on_device_ = back_is_on_device_;
  }

 private:
  bool back_is_on_device_ = false;
};

TEST_F(FlowFuncDrvManagerSTest, init_test_with_buf_cfg) {
  GlobalConfig::Instance().SetMessageQueueIds(100, 101);
  BufferConfigItem cfg0 = {2 * 1024 * 1024, 256, 8 * 1024, "normal"};
  BufferConfigItem cfg1 = {32 * 1024 * 1024, 8 * 1024, 8 * 1024 * 1024, "normal"};
  BufferConfigItem cfg2 = {2 * 1024 * 1024, 256, 8 * 1024, "huge"};
  BufferConfigItem cfg3 = {52 * 1024 * 1024, 8 * 1024, 50 * 1024 * 1024, "huge"};
  BufferConfigItem cfg4 = {66 * 1024 * 1024, 8 * 1024, 64 * 1024 * 1024, "huge"};
  GlobalConfig::Instance().SetBufCfg({cfg0, cfg1, cfg2, cfg3, cfg4});

  auto ret = FlowFuncDrvManager::Instance().Init();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  FlowFuncDrvManager::Instance().Finalize();
  GlobalConfig::Instance().SetMessageQueueIds(UINT32_MAX, UINT32_MAX);
  GlobalConfig::Instance().SetBufCfg({});
}

TEST_F(FlowFuncDrvManagerSTest, init_drv_test) {
  MOCKER(halGrpAttach)
      .stubs()
      .will(returnValue(static_cast<int>(DRV_ERROR_INNER_ERR)))
      .then(returnValue(static_cast<int>(DRV_ERROR_NONE)));
  MOCKER(halEschedAttachDevice).stubs().will(returnValue(DRV_ERROR_INNER_ERR)).then(returnValue(DRV_ERROR_NONE));
  MOCKER(halBuffInit)
      .stubs()
      .will(returnValue(static_cast<int>(DRV_ERROR_INNER_ERR)))
      .then(returnValue(static_cast<int>(DRV_ERROR_NONE)));
  MOCKER(halEschedCreateGrpEx).stubs().will(returnValue(DRV_ERROR_INNER_ERR)).then(returnValue(DRV_ERROR_NONE));
  auto ret = FlowFuncDrvManager::Instance().Init();
  // halGrpAttach error
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  ret = FlowFuncDrvManager::Instance().Init();
  // halEschedAttachDevice error
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  ret = FlowFuncDrvManager::Instance().Init();
  // halBuffInit error
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  ret = FlowFuncDrvManager::Instance().Init();
  // halEschedCreateGrp mainSchedGroup error
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);

  ret = FlowFuncDrvManager::Instance().Init();
  // halEschedCreateGrp workerSchedGroup error
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  FlowFuncDrvManager::Instance().Finalize();
}

TEST_F(FlowFuncDrvManagerSTest, WaitBindHostPid_timeout) {
  MOCKER(drvQueryProcessHostPid).stubs().will(returnValue(DRV_ERROR_NO_PROCESS));
  auto ret = FlowFuncDrvManager::Instance().WaitBindHostPid(200);
  // timeout error
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncDrvManagerSTest, WaitBindHostPid_failed) {
  MOCKER(drvQueryProcessHostPid).stubs().will(returnValue(DRV_ERROR_NO_PROCESS)).then(returnValue(DRV_ERROR_INNER_ERR));
  auto ret = FlowFuncDrvManager::Instance().WaitBindHostPid(200);
  // driver error
  EXPECT_EQ(ret, FLOW_FUNC_ERR_DRV_ERROR);
}

TEST_F(FlowFuncDrvManagerSTest, WaitBindHostPid_success) {
  MOCKER(drvQueryProcessHostPid).stubs().will(returnValue(DRV_ERROR_NO_PROCESS)).then(returnValue(DRV_ERROR_NONE));
  auto ret = FlowFuncDrvManager::Instance().WaitBindHostPid(200);
  // SUCCESS
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}
}  // namespace FlowFunc