/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <symengine/functions.h>
#include <symengine/simplify.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>
#include "gtest/gtest.h"
#include "base/base_types.h"
#include "solver_pass/src/l2_solver.cpp"
#include "test_common_utils.h"
using namespace att;

class MockStL2TileSolver : public L2TileSolver {
public:
  MockStL2TileSolver() {}
  explicit MockStL2TileSolver(L2TileInput input) : L2TileSolver(input) {};
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

class TestL2SolverSt : public ::testing::Test {
 public:
  void TearDown() override {
     // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
     // before the destructor).
  }
};

TEST_F(TestL2SolverSt, TEST_CASE_01)
{
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
  MockStL2TileSolver l2Solver(input);
  l2Solver.Run();
  uint32_t *output = l2Solver.GetL2Tile();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 7552);
  EXPECT_EQ(output[1], 7680);
  delete[] input.l2_vars;
}

TEST_F(TestL2SolverSt, TEST_CASE_02)
{
  L2TileInput input;
  L2Var tilem;
  L2Var tilen;
  tilem.max_value = 16;
  tilen.max_value = 16;
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
  MockStL2TileSolver l2Solver(input);
  l2Solver.Run();
  uint32_t *output = l2Solver.GetL2Tile();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 128);
  EXPECT_EQ(output[1], 256);
  delete[] input.l2_vars;
}

TEST_F(TestL2SolverSt, TEST_CASE_03)
{
  L2TileInput input;
  L2Var tilem;
  L2Var tilen;
  tilem.max_value = 16;
  tilen.max_value = 1024;
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
  MockStL2TileSolver l2Solver(input);
  l2Solver.Run();
  uint32_t *output = l2Solver.GetL2Tile();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 128);
  EXPECT_EQ(output[1], 1024);
  delete[] input.l2_vars;
}
