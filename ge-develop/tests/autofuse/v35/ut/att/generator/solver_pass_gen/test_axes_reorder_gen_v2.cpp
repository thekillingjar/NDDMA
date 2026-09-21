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
#include "generator/solver_pass_gen/axes_reorder_solver/axes_reorder_solver_gen.h"
#include "v35/att/api_perf_register/perf_param_v2.h"
using namespace att;

class TestAxesReorderSolverGenV2 : public ::testing::Test {
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
};

TEST_F(TestAxesReorderSolverGenV2, test_contain_heavy_op_A5) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  TilingScheduleConfigTableV2 tiling_schedule_config_table;
  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.enable_multicore_ub_tradeoff_ = false;
  std::string actual1 = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual1.find("solver.Run(false, true, ") != std::string::npos);

  solver.tiling_schedule_config_ = tiling_schedule_config_table.GetModelTilingScheduleConfig();
  solver.tiling_schedule_config_table_ = &tiling_schedule_config_table;
  std::string actual2 = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual2.find("solver.Run(true, false, ") != std::string::npos);
}