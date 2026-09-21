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
#include <thread>

#include "macro_utils/dt_public_scope.h"
#include "common/data_flow/route/rank_table_builder.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;

namespace ge {
class RankTableBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }
};

TEST_F(RankTableBuilderTest, GetRankTableTestSuc) {
  HcomNode node1 = {"192.168.1.101", {{"0", "1966", "", NPU, 0}, {"1", "1777", "", NPU, 1}}};
  HcomNode node2 = {"192.168.1.100", {{"0", "1888", ""}}};
  HcomRankTable rankTable = {"192.168.1.1:111-123", {node1, node2}};
  RankTableBuilder rankTableBuilder;
  rankTableBuilder.SetRankTable(rankTable);
  std::string rankTableStr;
  bool res = rankTableBuilder.GetRankTable(rankTableStr);
  EXPECT_EQ(res, true);

  int32_t rank_id;
  res = rankTableBuilder.GetRankIdByDeviceId("192.168.1.101", 0, rank_id);
  EXPECT_EQ(res, true);

  res = rankTableBuilder.GetRankIdByDeviceId("192.168.1.101", 1, rank_id);
  EXPECT_EQ(res, true);

  res = rankTableBuilder.GetRankIdByDeviceId("192.168.1.101", 3, rank_id);
  EXPECT_EQ(res, false);
}

TEST_F(RankTableBuilderTest, RankTableBuilderTestFail) {
  HcomNode node1 = {"192.168.1.101", {{"0", "1966", ""}, {"1", "1777", ""}}};
  HcomNode node2 = {"192.168.1.100", {{"0", "1888", ""}}};
  HcomRankTable rankTable = {"", {node1, node2}};
  RankTableBuilder rankTableBuilder;
  rankTableBuilder.SetRankTable(rankTable);
  std::string rankTableStr;
  bool res = rankTableBuilder.GetRankTable(rankTableStr);
  EXPECT_EQ(res, false);
}
}  // namespace ge
