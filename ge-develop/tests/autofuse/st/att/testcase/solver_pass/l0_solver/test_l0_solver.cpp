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
#include "solver_pass/src/l0_solver.cpp"
#include "test_common_utils.h"
using namespace att;

class MockStL0TileSolver : public L0TileSolver {
  public:
    MockStL0TileSolver() {}
    explicit MockStL0TileSolver(L0TileInput input) : L0TileSolver(input) {}
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

class case1L0TileSolver : public L0TileSolver {
  public:
    explicit case1L0TileSolver(L0TileInput &input) : L0TileSolver(input) {};
    void SetL1(uint32_t &value) { L1_ = value; }
    void SetL2(uint32_t &value) { L2_ = value; }
    void SetL0A(uint32_t &value) { L0A_ = value; }
    void SetL0B(uint32_t &value) { L0B_ = value; }
    void SetL0C(uint32_t &value) { L0C_ = value; }
    bool CheckBufferUseValid() override;
  private:
    uint32_t L1_;
    uint32_t L2_;
    uint32_t L0A_;
    uint32_t L0B_;
    uint32_t L0C_;
};
bool case1L0TileSolver::CheckBufferUseValid() {
  uint32_t basek_size = input_.l0_vars[0].value;
  uint32_t basem_size = input_.l0_vars[1].value;
  uint32_t basen_size = input_.l0_vars[2].value;
  
  uint32_t L0A =  (4 * basek_size * basem_size);
  if (L0A > L0A_) {
    return false;
   };
  uint32_t L0B =  (4 * basek_size * basen_size);
  if (L0B > L0B_) {
    return false;
   };
  uint32_t L0C =  (4 * basem_size * basen_size);
  if (L0C > L0C_) {
    return false;
   };
  return true;
}

class TestL0SolverSt : public ::testing::Test {
 public:
  void TearDown() override {
     // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
     // before the destructor).
  }
  MockStL0TileSolver solver;
};

TEST_F(TestL0SolverSt, TEST_CASE_01)
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
  solver = MockStL0TileSolver(input);
  solver.SetL0A(64 * 1024);
  solver.SetL0B(64 * 1024);
  solver.SetL0C(128 * 1024);
  EXPECT_EQ(solver.Run(), true);
  uint32_t *output = solver.GetOutput();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 16);
  EXPECT_EQ(output[1], 256);
  EXPECT_EQ(output[2], 64);
  delete[] input.l0_vars;
}

TEST_F(TestL0SolverSt, TEST_CASE_02)
{
  L0Var M;
  L0Var N;
  L0Var K;

  M.max_value = 1024;
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
  solver = MockStL0TileSolver(input);
  solver.SetL0A(64 * 1024);
  solver.SetL0B(64 * 1024);
  solver.SetL0C(128 * 1024);
  EXPECT_EQ(solver.Run(), true);
  uint32_t *output = solver.GetOutput();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 128);
  EXPECT_EQ(output[1], 256);
  EXPECT_EQ(output[2], 64);
  delete[] input.l0_vars;
}


TEST_F(TestL0SolverSt, TEST_CASE_03)
{
  L0Var M;
  L0Var N;
  L0Var K;

  M.max_value = 16;
  M.bind_multicore = 1;
  M.align = 16;
  M.prompt_align = 16;

  N.max_value = 16;
  N.bind_multicore = 1;
  N.align = 16;
  N.prompt_align = 256;

  K.max_value = 16;
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
  solver = MockStL0TileSolver(input);
  solver.SetL0A(64 * 1024);
  solver.SetL0B(64 * 1024);
  solver.SetL0C(128 * 1024);
  EXPECT_EQ(solver.Run(), true);
  uint32_t *output = solver.GetOutput();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 16);
  EXPECT_EQ(output[1], 16);
  EXPECT_EQ(output[2], 16);
  delete[] input.l0_vars;
}

TEST_F(TestL0SolverSt, TEST_CASE_04)
{
  L0Var M;
  L0Var N;
  L0Var K;

  M.max_value = 1;
  M.bind_multicore = 1;
  M.align = 16;
  M.prompt_align = 16;

  N.max_value = 1;
  N.bind_multicore = 1;
  N.align = 16;
  N.prompt_align = 256;

  K.max_value = 1;
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
  solver = MockStL0TileSolver(input);
  solver.SetL0A(64 * 1024);
  solver.SetL0B(64 * 1024);
  solver.SetL0C(128 * 1024);
  EXPECT_EQ(solver.Run(), true);
  uint32_t *output = solver.GetOutput();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 16);
  EXPECT_EQ(output[1], 16);
  EXPECT_EQ(output[2], 16);
  delete[] input.l0_vars;
}

TEST_F(TestL0SolverSt, TEST_CASE_05)
{
  L0TileInput l0_input;
  l0_input.l0_vars = new(std::nothrow) L0Var[3];
  l0_input.size = 3;
  l0_input.core_num = 20;
  L0Var basek_size;
  basek_size.max_value = 1024;
  basek_size.bind_multicore = false;
  basek_size.align = 16;
  basek_size.prompt_align = 16;
  l0_input.l0_vars[0] = basek_size;
  L0Var basem_size;
  basem_size.max_value = 1024;
  basem_size.bind_multicore = false;
  basem_size.align = 16;
  basem_size.prompt_align = 16;
  l0_input.l0_vars[1] = basem_size;
  L0Var basen_size;
  basen_size.max_value = 2048;
  basen_size.bind_multicore = false;
  basen_size.align = 16;
  basen_size.prompt_align = 16;
  l0_input.l0_vars[2] = basen_size;
  case1L0TileSolver solver(l0_input);
  uint32_t L0A = 64 * 1024;
  uint32_t L0B = 64 * 1024;
  uint32_t L0C = 128 * 1024;
  solver.SetL0A(L0A);
  solver.SetL0B(L0B);
  solver.SetL0C(L0C);
  EXPECT_EQ(solver.Run(), true);
  uint32_t *output = solver.GetOutput();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 128);
  EXPECT_EQ(output[1], 128);
  EXPECT_EQ(output[2], 128);
  delete[] l0_input.l0_vars;
}

TEST_F(TestL0SolverSt, TEST_CASE_06)
{
  L0Var M;
  L0Var N;
  L0Var K;

  M.max_value = 1024;
  M.bind_multicore = 0;
  M.align = 16;
  M.prompt_align = 16;

  N.max_value = 2048;
  N.bind_multicore = 0;
  N.align = 16;
  N.prompt_align = 16;

  K.max_value = 1024;
  K.bind_multicore = 0;
  K.align = 16;
  K.prompt_align = 16;

  L0TileInput input;
  input.l0_vars = new L0Var[3];
  input.l0_vars[0] = K;
  input.l0_vars[1] = M;
  input.l0_vars[2] = N;
  input.size = 3;
  input.core_num = 20;
  solver = MockStL0TileSolver(input);
  solver.SetL0A(64 * 1024);
  solver.SetL0B(64 * 1024);
  solver.SetL0C(128 * 1024);
  EXPECT_EQ(solver.Run(), true);
  uint32_t *output = solver.GetOutput();
  EXPECT_NE(output, nullptr);
  EXPECT_EQ(output[0], 256);
  EXPECT_EQ(output[1], 128);
  EXPECT_EQ(output[2], 64);
  delete[] input.l0_vars;
}


