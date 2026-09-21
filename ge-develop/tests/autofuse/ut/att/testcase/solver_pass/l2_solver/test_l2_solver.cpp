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
#include "base/base_types.h"
#define private public
#define protected public
#include "solver_pass/src/l2_solver.cpp"
using namespace att;

class MockUtL2TileSolver : public L2TileSolver {
public:
  MockUtL2TileSolver() {}
  explicit MockUtL2TileSolver(L2TileInput input) : L2TileSolver(input) {};
  uint64_t GetL2Use() override {
    uint64_t tilel2m = input_.l2_vars[0].value;
    uint64_t tilel2n = input_.l2_vars[1].value;
    uint64_t k = 512;
    uint64_t l2Use = tilel2m * k * 2 + k * tilel2n * 2 + tilel2m * tilel2n * 2;
    return l2Use;
  }
  bool IsClash(uint32_t idx) override {
    if (used_corenum_ <= 1 || used_corenum_ % 2 !=0) {
      return false;
    }
    if (blocknum_per_tile_[idx] % (used_corenum_ / 2) == 0) {
      return true;
    }
    auto blockNumTail =
        total_blocknum_[idx] - (tilenum_[idx] - 1) * blocknum_per_tile_[idx];
    if (blocknum_per_tile_[idx] % (used_corenum_ / 2) == 0) {
      return true;
    }
    return false;
  }
};

class TestL2SolverUt : public ::testing::Test {
public:
  static void TearDownTestCase() { std::cout << "Test end." << std::endl; }
  static void SetUpTestCase() { std::cout << "Test begin." << std::endl; }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
};

TEST_F(TestL2SolverUt, TEST_RUN) {
  L2TileInput input;
  L2Var tilem;
  L2Var tilen;
  tilem.max_value = 8196;
  tilen.max_value = 8196;
  tilem.align = 16;
  tilen.align = 16;
  tilem.base_val = 128;
  tilen.base_val = 256;
  input.l2_vars = new L2Var[2];
  input.l2_vars[0] = tilem;
  input.l2_vars[1] = tilen;
  input.size = 2;
  input.core_num = 24;
  input.l2_size = 128 * 1024 * 1024;
  MockUtL2TileSolver l2_solver(input);
  bool status = l2_solver.Run();
  EXPECT_EQ(status, true);
  uint32_t *output = l2_solver.GetL2Tile();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 7552);
  EXPECT_EQ(output[1], 7680);
}


TEST_F(TestL2SolverUt, TEST_CHECK_INPUT) {
  L2TileInput input;
  L2Var tilem;
  L2Var tilen;
  tilem.max_value = 8196;
  tilen.max_value = 8196;
  tilem.align = 16;
  tilen.align = 16;
  tilem.base_val = 128;
  tilen.base_val = 256;
  input.l2_vars = new L2Var[2];
  input.l2_vars[0] = tilem;
  input.l2_vars[1] = tilen;
  input.size = 2;
  input.core_num = 24;
  input.l2_size = 128 * 1024 * 1024;
  MockUtL2TileSolver l2_solver(input);
  EXPECT_EQ(l2_solver.CheckInput(), true);
  l2_solver.input_.l2_vars[0].align = 0;
  EXPECT_EQ(l2_solver.CheckInput(), false);
  l2_solver.input_.l2_vars[0].align = 16;
  l2_solver.input_.size = 0;
  EXPECT_EQ(l2_solver.CheckInput(), false);
  l2_solver.input_.size = 2;
  
  MockUtL2TileSolver null_l2_solver;
  EXPECT_EQ(null_l2_solver.CheckInput(), false);
}

TEST_F(TestL2SolverUt, TEST_CHECK_SOLVABLE) {
  L2TileInput input;
  L2Var tilem;
  L2Var tilen;
  tilem.max_value = 8196;
  tilen.max_value = 8196;
  tilem.align = 16;
  tilen.align = 16;
  tilem.base_val = 128;
  tilen.base_val = 256;
  input.l2_vars = new L2Var[2];
  input.l2_vars[0] = tilem;
  input.l2_vars[1] = tilen;
  input.size = 2;
  input.core_num = 24;
  input.l2_size = 128 * 1024 * 1024;
  MockUtL2TileSolver l2_solver(input);
  EXPECT_EQ(l2_solver.CheckSolvable(), true);
  l2_solver.input_.l2_size = 1;
  EXPECT_EQ(l2_solver.CheckSolvable(), false); 
}

TEST_F(TestL2SolverUt, TEST_BLOCK_NUM_PER_TILE_1) {
  L2TileInput input;
  L2Var tilem;
  L2Var tilen;
  tilem.max_value = 128;
  tilen.max_value = 8196;
  tilem.align = 16;
  tilen.align = 16;
  tilem.base_val = 128;
  tilen.base_val = 256;
  input.l2_vars = new L2Var[2];
  input.l2_vars[0] = tilem;
  input.l2_vars[1] = tilen;
  input.size = 2;
  input.core_num = 24;
  input.l2_size = 128 * 1024 * 1024;
  MockUtL2TileSolver l2_solver(input);
  bool status = l2_solver.Run();
  EXPECT_EQ(status, true);
}