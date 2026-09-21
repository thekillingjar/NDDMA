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
#include "gtest/gtest.h"
#include "solver.h"

class TilingCaseUT5Solver : public GeneralSolver {
 public:
  TilingCaseUT5Solver(SolverConfig &config) : GeneralSolver(config) {}
  void DisplayVarVal(uint64_t vars[]) override {
    return;
  }
  double GetObj(uint64_t vars[]) override {
    return 0;
  }
  double GetSmoothObj(uint64_t vars[]) override {
    return 0;
  }
  double GetBuffCost(uint64_t vars[]) override {
    return 0;
  }
  double GetBuffDiff(uint64_t vars[], double *weight) override {
    return 0;
  }
  double GetLeqCost(uint64_t vars[]) override {
    double s1s2Tb = static_cast<double>(vars[0]);
    double leq_cost = ceiling(s1_size * ceiling(s2_size / s2t_size) / s1s2Tb) - 48;
    return SMAX(leq_cost, 0.0) * SMAX(leq_cost, 0.0);
  }
  double GetLeqDiff(uint64_t vars[], double *weight) override {
    double s1s2Tb = static_cast<double>(vars[0]);
    double leq_cost = s1_size * s2_size / s2t_size / s1s2Tb - 48;
    leq_cost *= weight[0] > 0 ? weight[0] : 0;
    return leq_cost;
  }
  bool CheckLocalValid(double leqs[], int32_t idx) override {
    return true;
  }
  void UpdateLeqs(uint64_t vars[], int32_t idx, double leqs[]) override {
    double s1s2Tb = static_cast<double>(vars[0]);
    if (idx == -1 || idx == 0) {
      leqs[0] = ceiling(s1_size * ceiling(s2_size / s2t_size) / s1s2Tb) - 48;
    }
  }
  double s1_size = 1024;
  double s2_size = 8192;
  double s2t_size = 6992;
};

class UTTEST_GENERAL_SOLVER_05 : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    uint64_t top_num = 1;
    uint64_t search_length = 1;
    uint64_t iterations = 500;
    bool simple_ver = false;
    bool get_log = false;
    double momentum_factor = 0.9;
    SolverConfig cfg = {top_num, search_length, iterations, simple_ver, get_log, momentum_factor};
    solver_ = new TilingCaseUT5Solver(cfg);
    uint64_t init_vars[1] = {20};
    uint64_t upper_bound[1] = {20};
    uint64_t lower_bound[1] = {20};
    bool update_last[1] = {false};
    SolverInput input = {1, 1, upper_bound, lower_bound, init_vars, update_last};
    solver_->Init(input);
    solver_->Initialize(0);
  }
  void TearDown() override {}
  TilingCaseUT5Solver *solver_;
};

TEST_F(UTTEST_GENERAL_SOLVER_05, test_SetVarInfo) {
  SolverInput input1;
  input1.var_num = 10;
  input1.cur_vars = new uint64_t(10);
  input1.upper_bound = new uint64_t(10);
  input1.lower_bound = new uint64_t(10);
  solver_->var_info_ = new VarInfo(input1);
  EXPECT_EQ(solver_->var_info_->var_num, 10);
}