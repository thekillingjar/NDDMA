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
#include <string>
#include "util/duration.h"
#include "gen_model_info/test_fa_ascir_graph.h"
#include "gen_tiling_impl.h"
#include "base/att_const_values.h"
#include "graph_construct_utils.h"

namespace att {
class DurationUnitTest : public testing::Test {
 public:
  // 前处理：创建一个测试用的空文件
  void SetUp() override {}
  // 后处理：删除测试文件
  void TearDown() override {
  }
};

TEST_F(DurationUnitTest, Ok) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
  options.emplace(kOutputFilePath, kDefaultFilePath);
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "1");
  options.emplace(kGenConfigType, "HighPerf");
  std::string op_name = "OpTest";
  bool res = GenTilingImpl(op_name, graphs, options);
  EXPECT_EQ(res, true);
}

TEST_F(DurationUnitTest, DurationGuard) {
  extern uint32_t kg_duration_level;
  kg_duration_level = 1U;
  EXPECT_EQ(DurationGuardGenCode(TilingFuncDurationType::TILING_FUNC_DURATION_DOTILING), "");
  EXPECT_NE(DurationGuardGenCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL), "");
}
} //namespace