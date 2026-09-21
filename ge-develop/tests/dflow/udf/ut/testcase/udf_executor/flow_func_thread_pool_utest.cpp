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
#include <atomic>
#include "flow_func/flow_func_defines.h"
#define private public
#include "execute/flow_func_thread_pool.h"
#include "config/global_config.h"
#undef private

namespace FlowFunc {
class ThreadPoolUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    back_is_run_on_aicpu_ = GlobalConfig::Instance().IsRunOnAiCpu();
    GlobalConfig::Instance().SetRunOnAiCpu(false);
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
    GlobalConfig::Instance().SetRunOnAiCpu(back_is_run_on_aicpu_);
  }

 private:
  bool back_is_run_on_aicpu_ = false;
};

TEST_F(ThreadPoolUTest, init_success) {
  FlowFuncThreadPool thread_pool("udf_thread_");
  std::atomic<uint32_t> count(0);
  auto func = [&count](uint32_t thead_idx) { ++count; };
  int32_t ret = thread_pool.Init(func, 2);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  thread_pool.WaitForStop();
  EXPECT_EQ(count.load(), 2);
}

TEST_F(ThreadPoolUTest, device_init_success) {
  GlobalConfig::Instance().SetRunOnAiCpu(true);
  MOCKER(system).stubs().will(returnValue(0));
  FlowFuncThreadPool thread_pool("udf_thread_");
  std::atomic<uint32_t> count(0);
  auto func = [&count](uint32_t thead_idx) { ++count; };
  int32_t ret = thread_pool.Init(func, 2);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  thread_pool.WaitForStop();
  EXPECT_EQ(count.load(), 6);
}

TEST_F(ThreadPoolUTest, device_init_failed) {
  GlobalConfig::Instance().SetRunOnAiCpu(true);
  MOCKER(halGetDeviceInfo).stubs().will(returnValue(DRV_ERROR_NOT_SUPPORT));
  FlowFuncThreadPool thread_pool("udf_thread_");
  auto func = [](uint32_t thead_idx) {};
  int32_t ret = thread_pool.Init(func, 2);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  thread_pool.WaitForStop();
}

drvError_t halGetDeviceInfoBitmapNotMatch(uint32_t dev_id, int32_t module_type, int32_t info_type, int64_t *value) {
  switch (module_type) {
    case MODULE_TYPE_AICPU:
      if (info_type == INFO_TYPE_OS_SCHED) {
        *value = 1L;
      } else if (info_type == INFO_TYPE_CORE_NUM) {
        *value = 6L;
      } else if (info_type == INFO_TYPE_OCCUPY) {
        *value = 0b11111000L;
      } else {
        return DRV_ERROR_NOT_SUPPORT;
      }
      break;
    case MODULE_TYPE_CCPU:
    case MODULE_TYPE_DCPU:
      if (info_type == INFO_TYPE_OS_SCHED) {
        *value = 1L;
      } else if (info_type == INFO_TYPE_CORE_NUM) {
        *value = 1L;
      } else {
        return DRV_ERROR_NOT_SUPPORT;
      }
      break;
    default:
      return DRV_ERROR_NOT_SUPPORT;
  }
  return DRV_ERROR_NONE;
}

TEST_F(ThreadPoolUTest, device_aicpu_notmatch) {
  GlobalConfig::Instance().SetRunOnAiCpu(true);
  MOCKER(halGetDeviceInfo).stubs().will(invoke(halGetDeviceInfoBitmapNotMatch));
  FlowFuncThreadPool thread_pool("udf_thread_");
  auto func = [](uint32_t thead_idx) {};
  int32_t ret = thread_pool.Init(func, 2);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  thread_pool.WaitForStop();
}

drvError_t halGetDeviceInfoAicpuNotExists(uint32_t dev_id, int32_t module_type, int32_t info_type, int64_t *value) {
  switch (module_type) {
    case MODULE_TYPE_AICPU:
      if (info_type == INFO_TYPE_OS_SCHED) {
        *value = 0L;
      } else if (info_type == INFO_TYPE_CORE_NUM) {
        *value = 0L;
      } else if (info_type == INFO_TYPE_OCCUPY) {
        *value = 0b11111100L;
      } else {
        return DRV_ERROR_NOT_SUPPORT;
      }
      break;
    case MODULE_TYPE_CCPU:
    case MODULE_TYPE_DCPU:
      if (info_type == INFO_TYPE_OS_SCHED) {
        *value = 1L;
      } else if (info_type == INFO_TYPE_CORE_NUM) {
        *value = 1L;
      } else {
        return DRV_ERROR_NOT_SUPPORT;
      }
      break;
    default:
      return DRV_ERROR_NOT_SUPPORT;
  }
  return DRV_ERROR_NONE;
}

TEST_F(ThreadPoolUTest, device_aicpu_not_exists) {
  GlobalConfig::Instance().SetRunOnAiCpu(true);
  MOCKER(halGetDeviceInfo).stubs().will(invoke(halGetDeviceInfoAicpuNotExists));
  FlowFuncThreadPool thread_pool("udf_thread_");
  auto func = [](uint32_t thead_idx) {};
  int32_t ret = thread_pool.Init(func, 2);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  thread_pool.WaitForStop();
}

TEST_F(ThreadPoolUTest, call_system_failed) {
  FlowFuncThreadPool thread_pool("udf_thread_");
  MOCKER(system).stubs().will(returnValue(100));
  auto ret = thread_pool.WriteTidForAffinity();
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}
}  // namespace FlowFunc
