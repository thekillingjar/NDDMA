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
#include "gmock/gmock.h"
#include "base/base_types.h"
#define private public
#define protected public
#include "solver_pass/src/l0_solver.cpp"
#include <symengine/functions.h>
#include <symengine/simplify.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>
using namespace att;

class MockUtL0TileSolver : public L0TileSolver {
  public:
    MockUtL0TileSolver() {}
    explicit MockUtL0TileSolver(L0TileInput input) : L0TileSolver(input) {}
    void SetL0A(uint32_t value) { L0A_ = value; }
    void SetL0B(uint32_t value) { L0B_ = value; }
    void SetL0C(uint32_t value) { L0C_ = value; }
    bool CheckBufferUseValid() override {
      uint32_t l0A = input_.l0_vars[0].value * input_.l0_vars[2].value * 4;
      uint32_t l0B = input_.l0_vars[2].value * input_.l0_vars[1].value * 4;
      uint32_t l0C = input_.l0_vars[0].value * input_.l0_vars[1].value * 4;
      if (l0A > L0A_ || l0B > L0B_ || l0C > L0C_) {
        return false;
      }
      return true;
    }

  private:
    uint32_t L0A_;
    uint32_t L0B_;
    uint32_t L0C_;
};

class TestL0SolverUt : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }
  MockUtL0TileSolver solver;
};


TEST_F(TestL0SolverUt, TEST_CHECK_INPUT)
{
  EXPECT_EQ(solver.CheckInput(), false);

  L0Var M;
  L0Var N;
  L0Var K;
  M.max_value = 32;
  M.bind_multicore = 1;
  M.align = 16;
  M.prompt_align = 16;

  N.max_value = 1024;
  N.bind_multicore = 1;
  N.align = 16;
  N.prompt_align = 256;

  K.max_value = 1024;
  K.bind_multicore = 0;
  K.align = 16;
  K.prompt_align = 256;

  L0TileInput input;
  input.l0_vars = new L0Var[3];
  input.l0_vars[0] = M;
  input.l0_vars[1] = N;
  input.l0_vars[2] = K;
  input.size = 3;
  input.core_num = 24;
  solver = MockUtL0TileSolver(input);
  EXPECT_EQ(solver.CheckInput(), true);
  solver.input_.core_num = 0;
  EXPECT_EQ(solver.CheckInput(), false);
  solver.input_.core_num = 24;
  solver.input_.l0_vars[0].align=0;
  EXPECT_EQ(solver.CheckInput(), false);
  solver.input_.l0_vars[0].align=32;
  EXPECT_EQ(solver.CheckInput(), false);
  solver.input_.l0_vars[0].align=16;
  solver.input_.l0_vars[0].prompt_align=0;
  EXPECT_EQ(solver.CheckInput(), false);

  L0TileInput input_2;
  input_2.l0_vars = new L0Var[0];
  input_2.size = 0;
  input_2.core_num = 24;
  solver = MockUtL0TileSolver(input_2);
  EXPECT_EQ(solver.CheckInput(), false);

  L0TileInput input_3;
  input.l0_vars = new L0Var[4];
  input.size = 4;
  input.core_num = 24;
  solver = MockUtL0TileSolver(input);
  EXPECT_EQ(solver.CheckInput(), false);
}

TEST_F(TestL0SolverUt, TEST_MAX_CORE_NUM)
{
  L0Var M;
  L0Var N;
  L0Var K;

  M.max_value = 32;
  M.bind_multicore = 1;
  M.align = 16;
  M.prompt_align = 16;

  N.max_value = 1024;
  N.bind_multicore = 1;
  N.align = 16;
  N.prompt_align = 256;

  K.max_value = 1024;
  K.bind_multicore = 0;
  K.align = 16;
  K.prompt_align = 256;

  L0TileInput input;
  input.l0_vars = new L0Var[3];
  input.l0_vars[0] = M;
  input.l0_vars[1] = N;
  input.l0_vars[2] = K;
  input.size = 3;
  input.core_num = 24;
  solver = MockUtL0TileSolver(input);
  EXPECT_EQ(solver.MaxCoreNum(input.l0_vars, input.core_num), 8);
}

TEST_F(TestL0SolverUt, TEST_GET_MAC_USE)
{
  L0Var M;
  L0Var N;
  L0Var K;

  M.max_value = 32;
  M.bind_multicore = 1;
  M.align = 16;
  M.prompt_align = 16;

  N.max_value = 1024;
  N.bind_multicore = 1;
  N.align = 16;
  N.prompt_align = 256;

  K.max_value = 1024;
  K.bind_multicore = 0;
  K.align = 16;
  K.prompt_align = 256;

  L0TileInput input;
  input.l0_vars = new L0Var[3];
  input.l0_vars[0] = M;
  input.l0_vars[1] = N;
  input.l0_vars[2] = K;
  input.l0_vars[0].value = 16;
  input.l0_vars[1].value = 16;
  input.l0_vars[2].value = 16;
  input.size = 3;
  input.core_num = 24;
  solver = MockUtL0TileSolver(input);
  EXPECT_EQ(solver.GetMacUse(), 4096);
}

TEST_F(TestL0SolverUt, TEST_CHECK_OUTPUT)
{
  L0Var M;
  L0Var N;
  L0Var K;

  M.max_value = 32;
  M.bind_multicore = 1;
  M.align = 16;
  M.prompt_align = 16;

  N.max_value = 1024;
  N.bind_multicore = 1;
  N.align = 16;
  N.prompt_align = 256;

  K.max_value = 1024;
  K.bind_multicore = 0;
  K.align = 16;
  K.prompt_align = 256;

  L0TileInput input;
  input.l0_vars = new L0Var[3];
  input.l0_vars[0] = M;
  input.l0_vars[1] = N;
  input.l0_vars[2] = K;
  input.size = 3;
  input.core_num = 24;
  solver = MockUtL0TileSolver(input);
  EXPECT_EQ(solver.CheckOutput(), false);
  solver.SetL0A(1);
  solver.SetL0B(1);
  solver.SetL0C(1);
  solver.Run();
  EXPECT_EQ(solver.CheckOutput(), false);
  solver.SetL0A(64 * 1024);
  solver.SetL0B(64 * 1024);
  solver.SetL0C(128 * 1024);
  solver.Run();
  EXPECT_EQ(solver.CheckOutput(), true);
}


TEST_F(TestL0SolverUt, TEST_RUN)
{
  L0Var M;
  L0Var N;
  L0Var K;

  M.max_value = 32;
  M.bind_multicore = 1;
  M.align = 16;
  M.prompt_align = 16;

  N.max_value = 1024;
  N.bind_multicore = 1;
  N.align = 16;
  N.prompt_align = 256;

  K.max_value = 1024;
  K.bind_multicore = 0;
  K.align = 16;
  K.prompt_align = 256;

  L0TileInput input;
  input.l0_vars = new L0Var[3];
  input.l0_vars[0] = M;
  input.l0_vars[1] = N;
  input.l0_vars[2] = K;
  input.size = 3;
  input.core_num = 24;
  uint32_t *best_value = new uint32_t[input.size];
  solver = MockUtL0TileSolver(input);
  solver.SetL0A(1);
  solver.SetL0B(1);
  solver.SetL0C(1);
  EXPECT_EQ(solver.Run(), false);
  solver.SetL0A(64 * 1024);
  solver.SetL0B(64 * 1024);
  solver.SetL0C(128 * 1024);
  EXPECT_EQ(solver.Run(), true);
  uint32_t *output = solver.GetOutput();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 16);
  EXPECT_EQ(output[1], 256);
  EXPECT_EQ(output[2], 64);
  solver.input_.size = 0;
  EXPECT_EQ(solver.Run(), false);
  solver.input_.size = 3;
  solver.input_.l0_vars[0].max_value = 0;
  EXPECT_EQ(solver.Run(), false);
}