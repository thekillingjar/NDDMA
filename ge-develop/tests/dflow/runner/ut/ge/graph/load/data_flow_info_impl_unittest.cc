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

#include "ge/ge_data_flow_api.h"
#include "dflow/executor/data_flow_info_utils.h"

using namespace std;
using namespace testing;

namespace ge {
class DataFlowInfoTest : public Test {};

TEST_F(DataFlowInfoTest, SetUserData_Failed_InvalidParams) {
  DataFlowInfo info;
  ASSERT_EQ(info.SetUserData(nullptr, 0U), ACL_ERROR_GE_PARAM_INVALID);
  int8_t user_data[65];
  ASSERT_EQ(info.SetUserData(user_data, 0U), ACL_ERROR_GE_PARAM_INVALID);
  ASSERT_EQ(info.SetUserData(user_data, 65U), ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(DataFlowInfoTest, GetUserData_Failed_InvalidParams) {
  DataFlowInfo info;
  ASSERT_EQ(info.GetUserData(nullptr, 0U), ACL_ERROR_GE_PARAM_INVALID);
  int8_t user_data[65];
  ASSERT_EQ(info.GetUserData(user_data, 0U), ACL_ERROR_GE_PARAM_INVALID);
  ASSERT_EQ(info.GetUserData(user_data, 65U), ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(DataFlowInfoTest, check_custom_trans_id) {
  DataFlowInfo info;
  info.SetTransactionId(0);
  EXPECT_FALSE(DataFlowInfoUtils::HasCustomTransactionId(info));
  info.SetTransactionId(100);
  EXPECT_TRUE(DataFlowInfoUtils::HasCustomTransactionId(info));
}

TEST_F(DataFlowInfoTest, InitMsgInfoByDataFlowInfo_not_contains_n_mapping_node) {
  ExchangeService::MsgInfo msg_info{};
  DataFlowInfo info;
  info.SetStartTime(1);
  info.SetEndTime(2);
  info.SetFlowFlags(3);
  info.SetTransactionId(100);
  DataFlowInfoUtils::InitMsgInfoByDataFlowInfo(msg_info, info, false);
  EXPECT_EQ(msg_info.start_time, 1);
  EXPECT_EQ(msg_info.end_time, 2);
  EXPECT_EQ(msg_info.flags, 3);
  EXPECT_EQ(msg_info.trans_id, 0);
  EXPECT_EQ(msg_info.data_flag, 0);
}

TEST_F(DataFlowInfoTest, InitMsgInfoByDataFlowInfo_contains_n_mapping_node) {
  ExchangeService::MsgInfo msg_info{};
  DataFlowInfo info;
  info.SetStartTime(1);
  info.SetEndTime(2);
  info.SetFlowFlags(3);
  info.SetTransactionId(100);
  DataFlowInfoUtils::InitMsgInfoByDataFlowInfo(msg_info, info, true);
  EXPECT_EQ(msg_info.start_time, 1);
  EXPECT_EQ(msg_info.end_time, 2);
  EXPECT_EQ(msg_info.flags, 3);
  EXPECT_EQ(msg_info.trans_id, 100);
  EXPECT_EQ(msg_info.data_flag, kCustomTransIdFlagBit);
}

TEST_F(DataFlowInfoTest, InitDataFlowInfoByMsgInfo_custom_trans_id) {
  ExchangeService::MsgInfo msg_info{};
  msg_info.start_time = 1;
  msg_info.end_time = 2;
  msg_info.flags = 3;
  msg_info.trans_id = 100;
  msg_info.data_flag = kCustomTransIdFlagBit;
  DataFlowInfo info;
  DataFlowInfoUtils::InitDataFlowInfoByMsgInfo(info, msg_info);
  EXPECT_EQ(info.GetStartTime(), 1);
  EXPECT_EQ(info.GetEndTime(), 2);
  EXPECT_EQ(info.GetFlowFlags(), 3);
  EXPECT_EQ(info.GetTransactionId(), 100);
  EXPECT_TRUE(DataFlowInfoUtils::HasCustomTransactionId(info));
}

TEST_F(DataFlowInfoTest, InitDataFlowInfoByMsgInfo_without_custom_trans_id) {
  ExchangeService::MsgInfo msg_info{};
  msg_info.start_time = 1;
  msg_info.end_time = 2;
  msg_info.flags = 3;
  msg_info.trans_id = 100;
  msg_info.data_flag = 0;
  DataFlowInfo info;
  DataFlowInfoUtils::InitDataFlowInfoByMsgInfo(info, msg_info);
  EXPECT_EQ(info.GetStartTime(), 1);
  EXPECT_EQ(info.GetEndTime(), 2);
  EXPECT_EQ(info.GetFlowFlags(), 3);
  EXPECT_EQ(info.GetTransactionId(), 100);
  EXPECT_FALSE(DataFlowInfoUtils::HasCustomTransactionId(info));
}
}  // namespace ge
