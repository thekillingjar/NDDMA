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
#include "generator/solver_pass_gen/l2_solver/l2_solver_gen.h"
#include <symengine/functions.h>
#include <symengine/simplify.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>
#include "test_common_utils.h"
using namespace att;

class TestL2SolverGen : public ::testing::Test {
 public:
  void TearDown() override {
     // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
     // before the destructor).
  }
};

TEST_F(TestL2SolverGen, TEST_CASE_01)
{
  Expr m = CreateExpr("m");
  Expr n = CreateExpr("n");
  Expr k = CreateExpr("k");

  Expr basem = CreateExpr("basem");
  Expr basen = CreateExpr("basen");

  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");

  std::vector<att::Expr> input_args;
  std::vector<att::Expr> l0_args;
  std::vector<att::Expr> l2_args;
  std::map<att::Expr, Expr, att::ExprCmp> ori_arg_map;
  std::map<att::Expr, Expr, att::ExprCmp> arg_align_map;

  input_args.emplace_back(m);
  input_args.emplace_back(n);
  input_args.emplace_back(k);
  ori_arg_map[basem] = m;
  ori_arg_map[basen] = n;
  ori_arg_map[tilem] = m;
  ori_arg_map[tilen] = n;
  arg_align_map[tilem] = ge::Symbol(16);
  arg_align_map[tilen] = ge::Symbol(16);

  l0_args.emplace_back(basem);
  l0_args.emplace_back(basen);

  l2_args.emplace_back(tilem);
  l2_args.emplace_back(tilen);

  att::Expr l2_use_expr = ((((tilem * k) + (tilen * k)) + (tilem * tilen)) * CreateExpr(2));

  att::L2TileSolverGen *solver_gen = new att::L2TileSolverGen("Case0", "TilingData");
  solver_gen->SetArgAlignMap(arg_align_map);
  solver_gen->SetL0Args(l0_args);
  solver_gen->SetL2Args(l2_args);
  solver_gen->SetL2Use(l2_use_expr);
  solver_gen->SetArgtMaxValueMap(ori_arg_map);
  solver_gen->SetInputArgs(input_args);
  std::string impl_code = solver_gen->GenSolverFuncImpl();
  std::string invoke_code = solver_gen->GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
  delete solver_gen;
}

TEST_F(TestL2SolverGen, TEST_CASE_02)
{
  Expr m = CreateExpr("m");
  Expr n = CreateExpr("n");
  Expr k = CreateExpr("k");

  Expr basem = CreateExpr("basem");
  Expr basen = CreateExpr("basen");

  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");

  std::vector<att::Expr> input_args;
  std::vector<att::Expr> l0_args;
  std::vector<att::Expr> l2_args;
  std::map<att::Expr, Expr, att::ExprCmp> ori_arg_map;
  std::map<att::Expr, Expr, att::ExprCmp> arg_align_map;

  input_args.emplace_back(m);
  input_args.emplace_back(n);
  input_args.emplace_back(k);
  ori_arg_map[basem] = m;
  ori_arg_map[basen] = n;
  ori_arg_map[tilem] = m;
  ori_arg_map[tilen] = n;
  arg_align_map[tilem] = ge::Symbol(16);
  arg_align_map[tilen] = ge::Symbol(256);

  l0_args.emplace_back(basem);
  l0_args.emplace_back(basen);

  l2_args.emplace_back(tilem);
  l2_args.emplace_back(tilen);

  att::Expr l2_use_expr = ((((tilem * k) + (tilen * k)) + (tilem * tilen)) * CreateExpr(2));

  att::L2TileSolverGen *solver_gen = new att::L2TileSolverGen("Case0", "TilingData");
  solver_gen->SetArgAlignMap(arg_align_map);
  solver_gen->SetL0Args(l0_args);
  solver_gen->SetL2Args(l2_args);
  solver_gen->SetL2Use(l2_use_expr);
  solver_gen->SetArgtMaxValueMap(ori_arg_map);
  solver_gen->SetInputArgs(input_args);
  std::string impl_code = solver_gen->GenSolverFuncImpl();
  std::string invoke_code = solver_gen->GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
  delete solver_gen;
}
