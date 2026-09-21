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
#include "flow_func/logger/flow_func_logger_impl.h"
#include "flow_func/logger/flow_func_logger_manager.h"

namespace FlowFunc {
class FlowFuncLoggerUTest : public testing::Test {
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

TEST_F(FlowFuncLoggerUTest, debug_log_test) {
  FlowFuncLoggerImpl logger(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(UDF, DLOG_NULL, 1);
  logger.Debug("test debug");
  EXPECT_TRUE(logger.StatisticInfo().find("log_d:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_ERROR, 1);
  logger.Debug("test debug");
  EXPECT_TRUE(logger.StatisticInfo().find("log_d:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_WARN, 1);
  logger.Debug("test debug");
  EXPECT_TRUE(logger.StatisticInfo().find("log_d:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_INFO, 1);
  logger.Debug("test debug");
  EXPECT_TRUE(logger.StatisticInfo().find("log_d:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_DEBUG, 1);
  logger.Debug("test debug");
  EXPECT_TRUE(logger.StatisticInfo().find("log_d:1") != std::string::npos);
}

TEST_F(FlowFuncLoggerUTest, info_log_test) {
  FlowFuncLoggerImpl logger(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(UDF, DLOG_NULL, 1);
  logger.Info("test info");
  EXPECT_TRUE(logger.StatisticInfo().find("log_i:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_ERROR, 1);
  logger.Info("test info");
  EXPECT_TRUE(logger.StatisticInfo().find("log_i:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_WARN, 1);
  logger.Info("test info");
  EXPECT_TRUE(logger.StatisticInfo().find("log_i:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_INFO, 1);
  logger.Info("test info");
  EXPECT_TRUE(logger.StatisticInfo().find("log_i:1") != std::string::npos);

  dlog_setlevel(UDF, DLOG_DEBUG, 1);
  logger.Info("test info");
  EXPECT_TRUE(logger.StatisticInfo().find("log_i:2") != std::string::npos);
}

TEST_F(FlowFuncLoggerUTest, warn_log_test) {
  FlowFuncLoggerImpl logger(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(UDF, DLOG_NULL, 1);
  logger.Warn("test warn");
  EXPECT_TRUE(logger.StatisticInfo().find("log_w:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_ERROR, 1);
  logger.Warn("test warn");
  EXPECT_TRUE(logger.StatisticInfo().find("log_w:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_WARN, 1);
  logger.Warn("test warn");
  EXPECT_TRUE(logger.StatisticInfo().find("log_w:1") != std::string::npos);

  dlog_setlevel(UDF, DLOG_INFO, 1);
  logger.Warn("test warn");
  EXPECT_TRUE(logger.StatisticInfo().find("log_w:2") != std::string::npos);

  dlog_setlevel(UDF, DLOG_DEBUG, 1);
  logger.Warn("test warn");
  EXPECT_TRUE(logger.StatisticInfo().find("log_w:3") != std::string::npos);
}

TEST_F(FlowFuncLoggerUTest, error_log_test) {
  FlowFuncLoggerImpl logger(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(UDF, DLOG_NULL, 1);
  logger.Error("test error");
  EXPECT_TRUE(logger.StatisticInfo().find("log_e:0") != std::string::npos);

  dlog_setlevel(UDF, DLOG_ERROR, 1);
  logger.Error("test error");
  EXPECT_TRUE(logger.StatisticInfo().find("log_e:1") != std::string::npos);

  dlog_setlevel(UDF, DLOG_WARN, 1);
  logger.Error("test error");
  EXPECT_TRUE(logger.StatisticInfo().find("log_e:2") != std::string::npos);

  dlog_setlevel(UDF, DLOG_INFO, 1);
  logger.Error("test error");
  EXPECT_TRUE(logger.StatisticInfo().find("log_e:3") != std::string::npos);

  dlog_setlevel(UDF, DLOG_DEBUG, 1);
  logger.Error("test error");
  EXPECT_TRUE(logger.StatisticInfo().find("log_e:4") != std::string::npos);
}

TEST_F(FlowFuncLoggerUTest, check_debug_log_enable) {
  FlowFuncLoggerImpl logger(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(UDF, DLOG_NULL, 1);
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));

  dlog_setlevel(UDF, DLOG_ERROR, 1);
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));

  dlog_setlevel(UDF, DLOG_WARN, 1);
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));

  dlog_setlevel(UDF, DLOG_INFO, 1);
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));

  dlog_setlevel(UDF, DLOG_DEBUG, 1);
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));
}

TEST_F(FlowFuncLoggerUTest, check_run_log_enable) {
  FlowFuncLoggerImpl logger(FlowFuncLogType::RUN_LOG);
  dlog_setlevel(MODULE_ID_UDF_RUN_LOG, DLOG_NULL, 1);
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));

  dlog_setlevel(MODULE_ID_UDF_RUN_LOG, DLOG_ERROR, 1);
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));

  dlog_setlevel(MODULE_ID_UDF_RUN_LOG, DLOG_WARN, 1);
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));

  dlog_setlevel(MODULE_ID_UDF_RUN_LOG, DLOG_INFO, 1);
  EXPECT_FALSE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));

  dlog_setlevel(MODULE_ID_UDF_RUN_LOG, DLOG_DEBUG, 1);
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::DEBUG));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::INFO));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::WARN));
  EXPECT_TRUE(logger.IsLogEnable(FlowFuncLogLevel::ERROR));
}

TEST_F(FlowFuncLoggerUTest, debug_log_flow_control) {
  FlowFuncLoggerImpl logger(FlowFuncLogType::DEBUG_LOG);
  dlog_setlevel(UDF, DLOG_INFO, 1);
  // first set control 1000
  logger.RefreshLimitNum();

  for (int32_t i = 0; i < 1000; ++i) {
    logger.Info("test info log");
  }
  logger.Error("test error");
  logger.Warn("test warn");
  logger.Warn("test warn");
  logger.Info("test info");
  logger.Info("test info");
  logger.Info("test info");
  logger.Debug("test debug");
  logger.Debug("test debug");
  logger.Debug("test debug");
  logger.Debug("test debug");
  auto statistic_info = logger.StatisticInfo();
  // control static
  EXPECT_TRUE(statistic_info.find("[log_e:1, log_w:2, log_i:3, log_d:0]") != std::string::npos)
      << ", but statistic_info=" << statistic_info;
  // write static
  EXPECT_TRUE(statistic_info.find("[log_e:0, log_w:0, log_i:1000, log_d:0]") != std::string::npos)
      << ", but statistic_info=" << statistic_info;

  // add control 100
  logger.RefreshLimitNum();
  for (int32_t i = 0; i < 100; ++i) {
    logger.Info("test info log");
  }
  logger.Error("test error");
  logger.Warn("test warn");
  logger.Info("test info");
  logger.Debug("test debug");
  logger.Debug("test debug");
  statistic_info = logger.StatisticInfo();
  // control static
  EXPECT_TRUE(statistic_info.find("[log_e:2, log_w:3, log_i:4, log_d:0]") != std::string::npos)
      << ", but statistic_info=" << statistic_info;
  // write static
  EXPECT_TRUE(statistic_info.find("[log_e:0, log_w:0, log_i:1100, log_d:0]") != std::string::npos)
      << ", but statistic_info=" << statistic_info;
}

TEST_F(FlowFuncLoggerUTest, run_log_flow_control) {
  FlowFuncLoggerImpl logger(FlowFuncLogType::RUN_LOG);
  dlog_setlevel(MODULE_ID_UDF_RUN_LOG, DLOG_DEBUG, 1);
  // first set control 400
  logger.RefreshLimitNum();

  for (int32_t i = 0; i < 400; ++i) {
    logger.Debug("test debug log");
  }
  logger.Error("test error");
  logger.Warn("test warn");
  logger.Warn("test warn");
  logger.Info("test info");
  logger.Info("test info");
  logger.Info("test info");
  logger.Debug("test debug");
  logger.Debug("test debug");
  logger.Debug("test debug");
  logger.Debug("test debug");
  auto statistic_info = logger.StatisticInfo();
  // control static
  EXPECT_TRUE(statistic_info.find("[log_e:1, log_w:2, log_i:3, log_d:4]") != std::string::npos)
      << ", but statistic_info=" << statistic_info;
  // write static
  EXPECT_TRUE(statistic_info.find("[log_e:0, log_w:0, log_i:0, log_d:400]") != std::string::npos)
      << ", but statistic_info=" << statistic_info;

  // add control 50
  logger.RefreshLimitNum();
  for (int32_t i = 0; i < 50; ++i) {
    logger.Debug("test debug log");
  }
  logger.Error("test error");
  logger.Warn("test warn");
  logger.Info("test info");
  logger.Debug("test debug");
  statistic_info = logger.StatisticInfo();
  // control static
  EXPECT_TRUE(statistic_info.find("[log_e:2, log_w:3, log_i:4, log_d:5]") != std::string::npos)
      << ", but statistic_info=" << statistic_info;
  // write static
  EXPECT_TRUE(statistic_info.find("[log_e:0, log_w:0, log_i:0, log_d:450]") != std::string::npos)
      << ", but statistic_info=" << statistic_info;
}

TEST_F(FlowFuncLoggerUTest, get_logger_test) {
  auto &debug_logger1 = FlowFuncLoggerManager::Instance().GetLogger(FlowFuncLogType::DEBUG_LOG);
  auto &debug_logger2 = FlowFuncLoggerManager::Instance().GetLogger(FlowFuncLogType::DEBUG_LOG);
  auto &run_logger1 = FlowFuncLoggerManager::Instance().GetLogger(FlowFuncLogType::RUN_LOG);
  auto &run_logger2 = FlowFuncLoggerManager::Instance().GetLogger(FlowFuncLogType::RUN_LOG);
  EXPECT_EQ(&debug_logger1, &debug_logger2);
  EXPECT_EQ(&run_logger1, &run_logger2);
  EXPECT_NE(&debug_logger1, &run_logger1);
}
}  // namespace FlowFunc