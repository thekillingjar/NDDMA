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
class FlowFuncLoggerSTest : public testing::Test {
protected:
    virtual void SetUp()
    {
        backup_log_level_ = dlog_getlevel(UDF, &backup_enable_event_);
    }

    virtual void TearDown()
    {
        dlog_setlevel(UDF, backup_log_level_, backup_enable_event_);
    }

private:
    int32_t backup_enable_event_ = 1;
    int32_t backup_log_level_ = DLOG_ERROR;
};

TEST_F(FlowFuncLoggerSTest, debug_log_test)
{
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

TEST_F(FlowFuncLoggerSTest, info_log_test)
{
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

TEST_F(FlowFuncLoggerSTest, warn_log_test)
{
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

TEST_F(FlowFuncLoggerSTest, error_log_test)
{
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

TEST_F(FlowFuncLoggerSTest, get_logger_test)
{
    auto &debug_logger1 = FlowFuncLoggerManager::Instance().GetLogger(FlowFuncLogType::DEBUG_LOG);
    auto &debug_logger2 = FlowFuncLoggerManager::Instance().GetLogger(FlowFuncLogType::DEBUG_LOG);
    auto &run_logger1 = FlowFuncLoggerManager::Instance().GetLogger(FlowFuncLogType::RUN_LOG);
    auto &run_logger2 = FlowFuncLoggerManager::Instance().GetLogger(FlowFuncLogType::RUN_LOG);
    EXPECT_EQ(&debug_logger1, &debug_logger2);
    EXPECT_EQ(&run_logger1, &run_logger2);
    EXPECT_NE(&debug_logger1, &run_logger1);
}
}