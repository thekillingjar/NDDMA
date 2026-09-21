/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "util/att_utils.h"
#include "gen_model_info.h"

namespace att {
class TestAttUtils : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

// Test CollectReduceAxisNames - Basic case
TEST_F(TestAttUtils, CollectReduceAxisNames_BasicCase) {
  NodeInfo node_info;
  std::set<std::string> reduce_axis_orig_names;

  // Test with empty node - should not crash
  AttUtils::CollectReduceAxisNames(node_info, reduce_axis_orig_names);
  EXPECT_TRUE(reduce_axis_orig_names.empty());
}

// Test CollectReduceAxisNames - Empty input
TEST_F(TestAttUtils, CollectReduceAxisNames_EmptyInput) {
  NodeInfo node_info;
  std::set<std::string> reduce_axis_orig_names;

  AttUtils::CollectReduceAxisNames(node_info, reduce_axis_orig_names);
  EXPECT_TRUE(reduce_axis_orig_names.empty());
}

// Test CollectReduceAxisNames - No Reduce axes
TEST_F(TestAttUtils, CollectReduceAxisNames_NoReduceAxes) {
  // Create tensors without Reduce axes
  auto sub_axis = std::make_unique<SubAxis>();
  sub_axis->name = "A";

  auto input_tensor = std::make_shared<Tensor>();
  input_tensor->name = "input";
  input_tensor->dim_info = {sub_axis.get()};
  input_tensor->repeat = {ge::Symbol(128)};
  input_tensor->stride = {ge::Symbol(1)};

  auto output_tensor = std::make_shared<Tensor>();
  output_tensor->name = "output";
  output_tensor->dim_info = {sub_axis.get()};
  output_tensor->repeat = {ge::Symbol(128)};
  output_tensor->stride = {ge::Symbol(1)};

  NodeInfo node_info;
  node_info.name = "test_node";
  node_info.node_type = "TestOp";
  node_info.inputs = {input_tensor};
  node_info.outputs = {output_tensor};
  std::set<std::string> reduce_axis_orig_names;

  AttUtils::CollectReduceAxisNames(node_info, reduce_axis_orig_names);
  // No reduce axes, so the set should be empty or contain only non-reduce axes
}

// Note: Broadcast-related tests are not included per the plan
}  // namespace att
