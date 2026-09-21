/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "solver.h"
#include "test_common_utils.h"

const int32_t all = -1;
const int32_t x = 0;
const int32_t leq0 = 0;

class TilingCase13Solver : public GeneralSolver {
 public:
  TilingCase13Solver(SolverConfig &config) : GeneralSolver(config) {}

  void DisplayVarVal(uint64_t vars[]) override;
  double GetObj(uint64_t vars[]) override;
  double GetSmoothObj(uint64_t vars[]) override;
  double GetBuffCost(uint64_t vars[]) override;
  double GetLeqCost(uint64_t vars[]) override;
  bool CheckLocalValid(double leqs[], int32_t idx) override;
  void UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[]) override;
  double GetBuffDiff(uint64_t *vars, double *weight) override;
  double GetLeqDiff(uint64_t *vars, double *weight) override;

  std::vector<std::vector<uint64_t>> GetResult(int32_t solution_num, uint64_t *solution) {
    int32_t idx;
    std::vector<uint64_t> res;
    std::vector<std::vector<uint64_t>> result;
    for (int32_t i = 0; i < solution_num; i++) {
      res.clear();
      for (int32_t j = 0; j < GetVarNum(); j++) {
        res.emplace_back(*(solution + i * GetVarNum() + j));
      }
      result.emplace_back(res);
    }
    return result;
  }
};

double TilingCase13Solver::GetObj(uint64_t vars[]) {
  double d_x = static_cast<double>(vars[x]);
  return ceiling(1024 * ceiling(double(8192) / 6992) / d_x) * d_x;
}

double TilingCase13Solver::GetSmoothObj(uint64_t vars[]) {
  double d_x = static_cast<double>(vars[x]);
  return (1024 * (double(8192) / 6992) / d_x);
}

double TilingCase13Solver::GetBuffCost(uint64_t vars[]) {
  double d_x = static_cast<double>(vars[x]);
  return SMIN(ceiling(1024 * ceiling(double(8192) / 6992) / d_x) - 48, 0.0) *
         SMIN(ceiling(1024 * ceiling(double(8192) / 6992) / d_x) - 48, 0.0);
}

double TilingCase13Solver::GetBuffDiff(uint64_t vars[], double *weight) {
  double d_x = static_cast<double>(vars[x]);
  double block_cost = (1024 * (double(8192) / 6992) / d_x) - 48;
  block_cost *= weight[0] < 0 ? weight[0] : 0;
  return 0;
}

double TilingCase13Solver::GetLeqCost(uint64_t vars[]) {
  double d_x = static_cast<double>(vars[x]);
  return SMAX(ceiling(1024 * ceiling(double(8192) / 6992) / d_x) - 48, 0.0) *
         SMAX(ceiling(1024 * ceiling(double(8192) / 6992) / d_x) - 48, 0.0);
}

double TilingCase13Solver::GetLeqDiff(uint64_t vars[], double *weight) {
  double d_x = static_cast<double>(vars[x]);
  double block_cost = (1024 * (double(8192) / 6992) / d_x) - 48;
  block_cost *= weight[0] > 0 ? weight[0] : 0;
  return block_cost;
}

bool TilingCase13Solver::CheckLocalValid(double leqs[], int32_t idx) {
  if (idx == x) {
    return leqs[leq0] <= 0;
  }
  return false;
}

void TilingCase13Solver::UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[]) {
  double d_x = static_cast<double>(vars[x]);
  if (idx == all) {
    leqs[leq0] = ceiling(1024 * ceiling(double(8192) / 6992) / d_x) - 48;
  } else if (idx == x) {
    leqs[leq0] = ceiling(1024 * ceiling(double(8192) / 6992) / d_x) - 48;
  }
}

void TilingCase13Solver::DisplayVarVal(uint64_t vars[]) {
  std::cout << "a0 = " << vars[0] << std::endl;
}

class ATT_TEST_GENERAL_SOLVER_13 : public ::testing::Test {
 public:
  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    // before the destructor).
  }
};

std::vector<std::vector<uint64_t>> GetSolution() {
  uint64_t top_num = 1;
  uint64_t search_length = 1;
  uint64_t iterations = 100;
  bool get_log = false;
  bool simple_ver = false;
  double momentum_factor = 0.9;

  int32_t num_var = 1;
  int32_t num_leq = 1;
  uint64_t init_vars[num_var] = {1};
  uint64_t upper_bound[num_var] = {100};
  uint64_t lower_bound[num_var] = {1};
  bool update_last[num_var] = {false};

  std::vector<std::vector<uint64_t>> result;
  int32_t solution_num = 0;
  uint64_t *solution = new uint64_t[num_var * top_num];

  SolverInput input = {num_var, num_leq, upper_bound, lower_bound, init_vars, update_last};

  SolverConfig cfg = {top_num, search_length, iterations, simple_ver, get_log, momentum_factor};

  TilingCase13Solver *solver = new TilingCase13Solver(cfg);

  if (solver->Init(input)) {
    if (solver->Run(solution_num, solution)) {
      result = solver->GetResult(solution_num, solution);
    }
  }

  delete solver;
  delete[] solution;
  return result;
}

TEST_F(ATT_TEST_GENERAL_SOLVER_13, test_ceiling) {
  std::vector<std::vector<uint64_t>> expect_result;
  std::vector<std::vector<uint64_t>> ret = GetSolution();
  uint64_t used_core = ceiling(1024.0 * ceiling(8192.0 / 6992.0) / ret[0][0]);
  EXPECT_LT(used_core, 48);

  expect_result = {{64}};
  EXPECT_EQ(ret, expect_result);
}