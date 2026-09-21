/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "gen_model_info.h"
#include "test_fa_ascir_graph.h"
#define private public
#include "parser/ascend_graph_parser.h"
#undef private
#include "tests/autofuse/ut/att/utils/graph_construct_utils.h"

using namespace att;

class TestAscendGraphParserConflict : public testing::Test {
 public:
  static void TearDownTestCase() { std::cout << "Test end." << std::endl; }
  static void SetUpTestCase() { std::cout << "Test begin." << std::endl; }
  void SetUp() override {
    tuning_space = std::make_shared<att::TuningSpace>();
    EXPECT_NE(tuning_space, nullptr);
  }
  void TearDown() override {}
  att::TuningSpacePtr tuning_space;
};

// Test basic parsing functionality
TEST_F(TestAscendGraphParserConflict, BasicParserConstruction) {
  // Test that AscendGraphParser can be constructed with tuning_space
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  // Should not crash
}

// Test SubAxis properties
TEST_F(TestAscendGraphParserConflict, SubAxisDefaultProperties) {
  // Test that SubAxis has correct default properties
  auto sub_axis = std::make_shared<att::SubAxis>();
  EXPECT_NE(sub_axis, nullptr);
  EXPECT_FALSE(sub_axis->is_reduce_split_axis);
  EXPECT_FALSE(sub_axis->is_broadcast_split_axis);
}

// Note: Full Reduce split conflict tests require complex graph construction
//       These tests are done in ST tests (test_reduce_split_penalty.cpp)
//       which can properly construct Reduce graphs with split axes
//
// Note: Broadcast-related conflict tests are not included per the plan
