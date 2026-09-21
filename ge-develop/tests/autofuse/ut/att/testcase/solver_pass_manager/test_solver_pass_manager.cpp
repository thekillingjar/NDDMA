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
#include "base/model_info.h"
#define private public
#define protected public
#include "generator/solver_pass_gen/solver_pass_manager.h"
#include "solver_pass_manager/stub_model_info.h"
#include <symengine/functions.h>
#include <symengine/simplify.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>
using namespace att;

class TestSolverPassManager : public ::testing::Test {
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

TEST_F(TestSolverPassManager, TEST_CHECK_ARG_EXIST)
{
  ModelInfo modelInfo = CreateModelInfo();
  ArgsManager args_manager(modelInfo);
  att::SolverPassManager manager(args_manager, {0}, "TilingData");
  std::vector<Expr> args;
  Expr arg0 = CreateExpr("arg0");
  Expr arg1 = CreateExpr("arg1");
  Expr arg2 = CreateExpr("arg2");
  args.emplace_back(arg0);
  args.emplace_back(arg1);
  bool case0 = manager.CheckArgExist(arg2, args);
  EXPECT_EQ(case0, false);
  bool case1 = manager.CheckArgExist(arg0, args);
  EXPECT_EQ(case1, true);
}

TEST_F(TestSolverPassManager, TEST_GET_L0_ARGS)
{
  ModelInfo modelInfo = CreateModelInfo();
  ArgsManager args_manager(modelInfo);
  att::SolverPassManager manager(args_manager, {0}, "TilingData");
  args_manager.Process(false);
  auto l0_args = manager.GetL0Args(args_manager, false);
  EXPECT_EQ(l0_args.size(), 2);
}

TEST_F(TestSolverPassManager, TEST_IS_NEED_SOLVER)
{
  ModelInfo modelInfo = CreateModelInfo();
  ArgsManager args_manager(modelInfo);
  args_manager.Process(false);
  std::vector<ArgsManager> args_managers;
  args_managers.emplace_back(args_manager);
  att::SolverPassManager manager(args_manager, {0}, "TilingData");
  bool is_need = manager.IsNeedSolver(args_managers, SolverType::L0_TILE);
  EXPECT_EQ(is_need, true);
}

TEST_F(TestSolverPassManager, case0)
{
  ModelInfo modelInfo = CreateModelInfo();
  ArgsManager args_manager(modelInfo);
  att::SolverPassManager manager(args_manager, {0}, "TilingData");
  auto res = manager.GenFuncPass();
  std::vector<att::ArgsManager> args_managers;
  args_manager.Process(false);
  args_managers.emplace_back(args_manager);
  std::string base_class_head = manager.GenCommonBaseClassesHead(args_managers);
  std::string base_class_func = manager.GenCommonBaseClassesFunc(args_managers);
  EXPECT_NE(base_class_head, "");
  EXPECT_NE(base_class_func, "");
  std::string impl_code = res.first;
  std::string invoke_code = res.second;
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}