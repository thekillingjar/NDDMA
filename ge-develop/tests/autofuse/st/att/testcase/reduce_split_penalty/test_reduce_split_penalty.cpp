/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/test_common_headers.h"
#include "common/test_common_utils.h"

using namespace att;

class STestReduceSplitPenalty : public ::testing::Test {
 public:
  static void SetUpTestCase() { std::cout << "Test begin." << std::endl; }
  static void TearDownTestCase() { std::cout << "Test end." << std::endl; }
  void SetUp() override {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    // V1形态：通过TilingScheduleConfigTableV1自动返回false，无需手动设置
  }
  void TearDown() override {
  }
};

namespace {

using att::test_common::ValidateTilingResult;
using att::test_common::PrintTilingDebugInfo;
using att::test_common::ConstructSingleCaseForReduceSplitPenalty;
using att::test_common::GenTilingImpl;
using att::test_common::CreateTilingMainFunc;
using att::test_common::RunBasicTilingTest;

// Test Reduce split penalty scenario with Reduce on s2t axis, multi-core on merged axis
TEST_F(STestReduceSplitPenalty, reduce_split_penalty_basic) {
  // Build graph with Reduce on s2t axis, multi-core on merged s1Ts2T axis
  // - s2t is the Reduce axis (reduced by Mean operation)
  // - s1Ts2T is the merged axis that gets split for multi-core
  std::vector<ascir::ScheduledResult> schedule_results;
  ASSERT_EQ(ConstructSingleCaseForReduceSplitPenalty(schedule_results), ge::SUCCESS);
  RunBasicTilingTest("ReduceSplitPenalty", schedule_results);
}

TEST_F(STestReduceSplitPenalty, reduce_split_penalty_run)
{
  std::vector<ascir::ScheduledResult> schedule_results;
  ASSERT_EQ(ConstructSingleCaseForReduceSplitPenalty(schedule_results), ge::SUCCESS);
  ASSERT_EQ(GenTilingImpl(schedule_results), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_ReduceSplitPenalty.cpp", std::ios::out);
  oss << "#include \"ReduceSplitPenalty_tiling_data.h\"" << ResultCheckerUtils::DefineCheckerFunction()
      << CreateTilingMainFunc("ReduceSplitPenalty", "64", "245760", {{"S0", "1024"}, {"S1", "1024"}});
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_ReduceSplitPenalty.cpp ReduceSplitPenalty_tiling_func.cpp -o "
      "tiling_func_main_ReduceSplitPenalty -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_ReduceSplitPenalty");
  EXPECT_EQ(ret, 0);
}
}  // namespace
