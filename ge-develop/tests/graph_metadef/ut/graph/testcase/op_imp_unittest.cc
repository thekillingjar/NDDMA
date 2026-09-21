/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator_reg.h"
#include <gtest/gtest.h>
#include <vector>

namespace ge {
class BroadCastInferUt : public testing::Test {};

TEST_F(BroadCastInferUt, Scalar1) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 3});},
      []() { return std::vector<int64_t>({}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 3}));

  ret_shape.clear();
  ret = BroadCastInfer(
      []() { return std::vector<int64_t>({});},
      []() { return std::vector<int64_t>({1, 2, 3}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 3}));
}

TEST_F(BroadCastInferUt, SameShape) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 3});},
      []() { return std::vector<int64_t>({1, 2, 3}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 3}));
}

TEST_F(BroadCastInferUt, BroadCastDim1) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({3, 2, 1});},
      []() { return std::vector<int64_t>({1, 2, 3}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({3, 2, 3}));
}

TEST_F(BroadCastInferUt, BroadCastRank) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 3, 4});},
      []() { return std::vector<int64_t>({3, 4}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 3, 4}));
}

TEST_F(BroadCastInferUt, BroadCastRankAndDim1) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 1, 4});},
      []() { return std::vector<int64_t>({5, 4}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 5, 4}));
}

TEST_F(BroadCastInferUt, BroadCastFailed_DimDiff) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({1, 2, 3, 4});},
      []() { return std::vector<int64_t>({5, 4}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(BroadCastInferUt, BroadCastRankAndDim1_1) {
  std::vector<int64_t> ret_shape;
  auto ret = BroadCastInfer(
      []() { return std::vector<int64_t>({5, 4});},
      []() { return std::vector<int64_t>({1, 2, 1, 4}); },
      [&ret_shape](const std::vector<int64_t> &out_shape) { ret_shape = out_shape; });
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ret_shape, std::vector<int64_t>({1, 2, 5, 4}));
}
}  // namespace ge
