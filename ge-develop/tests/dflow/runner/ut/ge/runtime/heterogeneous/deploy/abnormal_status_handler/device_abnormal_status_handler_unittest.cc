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
#include "depends/runtime/src/runtime_stub.h"
#include "deploy/abnormal_status_handler/device_abnormal_status_handler.h"
#include "acl/acl.h"

namespace ge {
namespace {
class CallbackManager {
 public:
  static CallbackManager &GetInstance() {
    static CallbackManager callbackManager;
    return callbackManager;
  }

  rtError_t Register(const char *moduleName, rtTaskFailCallback callback) {
    if (return_failed) {
      return ACL_ERROR_RT_PARAM_INVALID;
    }
    if (callback == nullptr) {
      callbacks_.erase(moduleName);
    } else {
      callbacks_.emplace(moduleName, callback);
    }
    return RT_ERROR_NONE;
  }

  void Call(const char *moduleName, rtExceptionInfo *excpt_info) {
    const std::string name = moduleName;
    auto iter = callbacks_.find(name);
    if (iter != callbacks_.cend()) {
      rtTaskFailCallback callback = iter->second;
      callback(excpt_info);
    }
  }

  void SetReturnFailed() {
    return_failed = true;
  }

  void Reset() {
    callbacks_.clear();
    return_failed = false;
  }

  std::map<std::string, rtTaskFailCallback> callbacks_;
  bool return_failed = false;
};

class MockRuntimeCallbck : public RuntimeStub {
 public:
  rtError_t rtRegTaskFailCallbackByModule(const char *moduleName, rtTaskFailCallback callback) override {
    return CallbackManager::GetInstance().Register(moduleName, callback);
  }
};
}  // namespace
class DeviceAbnormalStatusHandlerTest : public testing::Test {
 protected:
  void SetUp() override {
    auto mock_runtime = std::make_shared<MockRuntimeCallbck>();
    RuntimeStub::SetInstance(mock_runtime);
  }
  void TearDown() override {
    CallbackManager::GetInstance().Reset();
  }
};

TEST_F(DeviceAbnormalStatusHandlerTest, TestRegFailed) {
  CallbackManager::GetInstance().SetReturnFailed();
  DeviceAbnormalStatusHandler::Instance().Initialize();
  EXPECT_TRUE(CallbackManager::GetInstance().callbacks_.empty());
  DeviceAbnormalStatusHandler::Instance().Finalize();
}

TEST_F(DeviceAbnormalStatusHandlerTest, TestRegSuccess) {
  DeviceAbnormalStatusHandler::Instance().Initialize();
  EXPECT_EQ(CallbackManager::GetInstance().callbacks_.count("deployer_dev_abnormal"), 1);
  DeviceAbnormalStatusHandler::Instance().Finalize();
  EXPECT_TRUE(CallbackManager::GetInstance().callbacks_.empty());
}

TEST_F(DeviceAbnormalStatusHandlerTest, TestRegCallbackNull) {
  DeviceAbnormalStatusHandler::Instance().Initialize();
  CallbackManager::GetInstance().Call("deployer_dev_abnormal", nullptr);
  std::map<uint32_t, std::vector<uint32_t>> abnormal_device_info;
  DeviceAbnormalStatusHandler::Instance().HandleDeviceAbnormal(abnormal_device_info);
  EXPECT_TRUE(abnormal_device_info.empty());
  DeviceAbnormalStatusHandler::Instance().Finalize();
}

TEST_F(DeviceAbnormalStatusHandlerTest, TestRegCallbackOom) {
  DeviceAbnormalStatusHandler::Instance().Initialize();
  rtExceptionInfo excpt_info{};
  excpt_info.deviceid = 1;
  excpt_info.retcode = ACL_ERROR_RT_DEVICE_OOM;
  CallbackManager::GetInstance().Call("deployer_dev_abnormal", &excpt_info);
  std::map<uint32_t, std::vector<uint32_t>> abnormal_device_info;
  DeviceAbnormalStatusHandler::Instance().HandleDeviceAbnormal(abnormal_device_info);
  EXPECT_FALSE(abnormal_device_info.empty());
  EXPECT_EQ(abnormal_device_info.count(1), 1);

  // after handle will clear
  std::map<uint32_t, std::vector<uint32_t>> abnormal_device_info_new;
  DeviceAbnormalStatusHandler::Instance().HandleDeviceAbnormal(abnormal_device_info_new);
  EXPECT_TRUE(abnormal_device_info_new.empty());
  DeviceAbnormalStatusHandler::Instance().Finalize();
}

TEST_F(DeviceAbnormalStatusHandlerTest, TestRegCallbackIgnoreOtherFailed) {
  DeviceAbnormalStatusHandler::Instance().Initialize();
  rtExceptionInfo excpt_info{};
  excpt_info.deviceid = 1;
  excpt_info.retcode = 111;
  CallbackManager::GetInstance().Call("deployer_dev_abnormal", &excpt_info);
  std::map<uint32_t, std::vector<uint32_t>> abnormal_device_info;
  DeviceAbnormalStatusHandler::Instance().HandleDeviceAbnormal(abnormal_device_info);
  EXPECT_TRUE(abnormal_device_info.empty());
  DeviceAbnormalStatusHandler::Instance().Finalize();
}
}  // namespace ge
