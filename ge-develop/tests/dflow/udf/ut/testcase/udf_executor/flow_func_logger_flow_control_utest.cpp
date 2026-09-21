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
#include "dlog_pub.h"
#include "flow_func/logger/flow_func_logger_flow_control.h"

namespace FlowFunc {
class FlowFuncLoggerFlowControlUTest : public testing::Test {
 protected:
  static constexpr int32_t MODULE_ID_UDF_RUN_LOG = RUN_LOG_MASK | static_cast<int32_t>(UDF);
  static constexpr int32_t MODULE_ID_UDF_DEBUG_LOG = DEBUG_LOG_MASK | static_cast<int32_t>(UDF);

  virtual void SetUp() {
    backup_run_log_level_ = dlog_getlevel(MODULE_ID_UDF_RUN_LOG, &backup_enable_event_);
    backup_debug_log_level_ = dlog_getlevel(MODULE_ID_UDF_DEBUG_LOG, &backup_enable_event_);
  }

  virtual void TearDown() {
    dlog_setlevel(MODULE_ID_UDF_RUN_LOG, backup_run_log_level_, backup_enable_event_);
    dlog_setlevel(MODULE_ID_UDF_DEBUG_LOG, backup_debug_log_level_, backup_enable_event_);
  }

 private:
  int32_t backup_enable_event_ = 1;
  int32_t backup_debug_log_level_ = DLOG_ERROR;
  int32_t backup_run_log_level_ = DLOG_ERROR;
};

// debug log level no flow control
TEST_F(FlowFuncLoggerFlowControlUTest, debug_log_debug_level_test) {
  FlowFuncLoggerFlowControl flow_control(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(MODULE_ID_UDF_DEBUG_LOG, DLOG_DEBUG, 1);
  flow_control.RefreshLimitNum();
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
}

// info log level: limit 100/s, transient limit 1000.
TEST_F(FlowFuncLoggerFlowControlUTest, debug_log_info_level_test) {
  FlowFuncLoggerFlowControl flow_control(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(MODULE_ID_UDF_DEBUG_LOG, DLOG_INFO, 1);
  // first limit set to max value.
  flow_control.RefreshLimitNum();
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  flow_control.RefreshLimitNum();
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh 2 times, limit num is 200;
  flow_control.RefreshLimitNum();
  flow_control.RefreshLimitNum();
  for (int i = 0; i < 200; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh 10 times, limit num is 1000;
  for (int i = 0; i < 10; ++i) {
    flow_control.RefreshLimitNum();
  }
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh over 10 times, limit num is also 1000;
  for (int i = 0; i < 15; ++i) {
    flow_control.RefreshLimitNum();
  }
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());
}

// warn log limit 50/s, transient limit 400.
TEST_F(FlowFuncLoggerFlowControlUTest, debug_log_warn_level_test) {
  FlowFuncLoggerFlowControl flow_control(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(MODULE_ID_UDF_DEBUG_LOG, DLOG_WARN, 1);
  constexpr int32_t add_limit_num_per_time = 50;
  // first limit set to max value.
  flow_control.RefreshLimitNum();
  for (int i = 0; i < 400; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  flow_control.RefreshLimitNum();
  for (int i = 0; i < add_limit_num_per_time; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh 2 times, limit num is 50 * 2;
  flow_control.RefreshLimitNum();
  flow_control.RefreshLimitNum();
  for (int i = 0; i < add_limit_num_per_time * 2; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh 8 times, limit num is 8 * 50;
  for (int i = 0; i < 8; ++i) {
    flow_control.RefreshLimitNum();
  }
  for (int i = 0; i < 8 * add_limit_num_per_time; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh over 8 times, limit num is also 400;
  for (int i = 0; i < 10; ++i) {
    flow_control.RefreshLimitNum();
  }
  for (int i = 0; i < 8 * add_limit_num_per_time; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());
}

// error log limit 50/s, transient limit 400.
TEST_F(FlowFuncLoggerFlowControlUTest, debug_log_error_level_test) {
  FlowFuncLoggerFlowControl flow_control(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(MODULE_ID_UDF_DEBUG_LOG, DLOG_ERROR, 1);
  constexpr int32_t add_limit_num_per_time = 50;
  // first limit set to max value.
  flow_control.RefreshLimitNum();
  for (int i = 0; i < 400; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  flow_control.RefreshLimitNum();
  for (int i = 0; i < add_limit_num_per_time; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh 2 times, limit num is 50 * 2;
  flow_control.RefreshLimitNum();
  flow_control.RefreshLimitNum();
  for (int i = 0; i < add_limit_num_per_time * 2; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh 8 times, limit num is 8 * 50;
  for (int i = 0; i < 8; ++i) {
    flow_control.RefreshLimitNum();
  }
  for (int i = 0; i < 8 * add_limit_num_per_time; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh over 8 times, limit num is also 400;
  for (int i = 0; i < 10; ++i) {
    flow_control.RefreshLimitNum();
  }
  for (int i = 0; i < 8 * add_limit_num_per_time; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());
}

//  run log limit 50/s, transient limit 400.
TEST_F(FlowFuncLoggerFlowControlUTest, run_log_error_level_test) {
  FlowFuncLoggerFlowControl flow_control(FlowFuncLogType::RUN_LOG);
  dlog_setlevel(MODULE_ID_UDF_DEBUG_LOG, DLOG_ERROR, 1);
  constexpr int32_t add_limit_num_per_time = 50;
  // first limit set to max value.
  flow_control.RefreshLimitNum();
  for (int i = 0; i < 400; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  flow_control.RefreshLimitNum();
  for (int i = 0; i < add_limit_num_per_time; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh 2 times, limit num is 50 * 2;
  flow_control.RefreshLimitNum();
  flow_control.RefreshLimitNum();
  for (int i = 0; i < add_limit_num_per_time * 2; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh 8 times, limit num is 8 * 50;
  for (int i = 0; i < 8; ++i) {
    flow_control.RefreshLimitNum();
  }
  for (int i = 0; i < 8 * add_limit_num_per_time; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());

  // refresh over 8 times, limit num is also 400;
  for (int i = 0; i < 10; ++i) {
    flow_control.RefreshLimitNum();
  }
  for (int i = 0; i < 8 * add_limit_num_per_time; ++i) {
    EXPECT_TRUE(flow_control.TryAcquire());
  }
  EXPECT_FALSE(flow_control.TryAcquire());
}
}  // namespace FlowFunc