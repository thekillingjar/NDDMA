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
#include "generator/solver_pass_gen/l0_solver/l0_solver_gen.h"
#include <symengine/functions.h>
#include <symengine/simplify.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>
using namespace att;

class TestL0SolverGen : public ::testing::Test {
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

TEST_F(TestL0SolverGen, TEST_IS_MULTICORE_ARG)
{
  L0TileSolverGen solver_gen("case0", "TilingData");
  std::vector<Expr> mc_args;
  Expr null_expr;
  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");
  Expr basem = CreateExpr("basem");
  solver_gen.mc_args_.emplace_back(tilem);
  solver_gen.mc_args_.emplace_back(tilen);
  bool case0_is_mc = solver_gen.IsMulticoreArg(tilem);
  bool case1_is_mc = solver_gen.IsMulticoreArg(tilen);
  bool case2_is_mc = solver_gen.IsMulticoreArg(basem);
  bool case3_is_mc = solver_gen.IsMulticoreArg(null_expr);
  
  EXPECT_EQ(case0_is_mc, true);
  EXPECT_EQ(case1_is_mc, true);
  EXPECT_EQ(case2_is_mc, false);
  EXPECT_EQ(case3_is_mc, false);
}

TEST_F(TestL0SolverGen, TEST_IS_BIND_MULTICORE)
{
  L0TileSolverGen solver_gen("case0", "TilingData");
  std::vector<Expr> mc_args;
  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");
  Expr basem = CreateExpr("basem");
  std::map<Expr, Expr, ExprCmp> father_args_map;
  father_args_map[basem] = tilem;
  solver_gen.mc_args_.emplace_back(tilem);
  solver_gen.mc_args_.emplace_back(tilen);
  EXPECT_EQ(solver_gen.IsBindMulticore(basem), false);
  solver_gen.SetFatherArgsMap(father_args_map);
  EXPECT_EQ(solver_gen.IsBindMulticore(basem), true);
}

TEST_F(TestL0SolverGen, TEST_GET_LARGEST_ALIGN)
{
  L0TileSolverGen solver_gen("case0", "TilingData");
  std::vector<Expr> mc_args;
  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");
  Expr basem = CreateExpr("basem");
  std::map<Expr, Expr, ExprCmp> father_args_map;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  father_args_map[basem] = tilem;
  arg_align_map[basem] = ge::Symbol(16);
  arg_align_map[tilem] = ge::Symbol(256);
  arg_align_map[tilen] = ge::Symbol(16);
  solver_gen.mc_args_.emplace_back(tilem);
  solver_gen.mc_args_.emplace_back(tilen);
  solver_gen.SetFatherArgsMap(father_args_map);
  solver_gen.SetArgAlignMap(arg_align_map);
  Expr max_align = ge::Symbol(16);
  solver_gen.GetLargestAlign(basem, max_align);
  EXPECT_EQ(max_align, 256);
}

TEST_F(TestL0SolverGen, TEST_GEN_SOLVER)
{
  L0TileSolverGen solver_gen("case0", "TilingData");
  std::vector<Expr> ori_args;
  std::vector<Expr> l0_args;
  std::vector<Expr> mc_args;
  ExprUintMap const_args;
  ExprExprMap father_args_map;
  ExprExprMap arg_align_map;
  ExprExprMap ori_arg_map;

  Expr m = CreateExpr("m");
  Expr n = CreateExpr("n");
  Expr k = CreateExpr("k");
  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");
  Expr basem = CreateExpr("basem");
  Expr basen = CreateExpr("basen");
  Expr basek = CreateExpr("basek");
  Expr constbl = CreateExpr("bl");
  Expr buffer_use = ((basem * basek) + (basen * basek));

  std::map<HardwareDef, Expr> buffer_use_map;
  buffer_use_map[HardwareDef::L0A] = buffer_use;

  ori_args.emplace_back(m);
  ori_args.emplace_back(n);
  ori_args.emplace_back(k);


  l0_args.emplace_back(basem);
  l0_args.emplace_back(basen);
  l0_args.emplace_back(basek);

  mc_args.emplace_back(tilem);
  mc_args.emplace_back(tilen);

  ori_arg_map[basem] = m;
  ori_arg_map[basen] = n;
  ori_arg_map[basek] = k;

  father_args_map[basem] = tilem;
  father_args_map[basen] = tilen;
  
  arg_align_map[basem] = ge::Symbol(16);
  arg_align_map[basen] = ge::Symbol(16);
  arg_align_map[basek] = ge::Symbol(16);
  arg_align_map[tilem] = ge::Symbol(256);
  arg_align_map[tilen] = ge::Symbol(16);

  const_args[constbl] = 8;

  solver_gen.SetMulticoreArgs(mc_args);
  solver_gen.SetFatherArgsMap(father_args_map);
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetArgtMaxValueMap(ori_arg_map);
  solver_gen.SetL0Args(l0_args);
  solver_gen.SetBufferUseAlg(buffer_use_map);
  solver_gen.SetConstVars(const_args);
  std::string impl_code = solver_gen.GenSolverFuncImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}


TEST_F(TestL0SolverGen, TEST_GEN_SOLVER_ERR)
{
  L0TileSolverGen solver_gen("case0", "TilingData");
  std::vector<Expr> ori_args;
  std::vector<Expr> l0_args;
  std::vector<Expr> mc_args;
  ExprUintMap const_args;
  ExprExprMap father_args_map;
  ExprExprMap arg_align_map;
  ExprExprMap ori_arg_map;

  Expr m = CreateExpr("m");
  Expr n = CreateExpr("n");
  Expr k = CreateExpr("k");
  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");
  Expr basem = CreateExpr("basem");
  Expr basen = CreateExpr("basen");
  Expr basek = CreateExpr("basek");
  Expr constbl = CreateExpr("bl");
  Expr buffer_use = ((basem * basek) + (basen * basek));

  std::map<HardwareDef, Expr> buffer_use_map;
  buffer_use_map[HardwareDef::L0A] = buffer_use;

  ori_args.emplace_back(m);
  ori_args.emplace_back(n);
  ori_args.emplace_back(k);


  l0_args.emplace_back(basem);
  l0_args.emplace_back(basen);
  l0_args.emplace_back(basek);

  mc_args.emplace_back(tilem);
  mc_args.emplace_back(tilen);

  ori_arg_map[basem] = m;
  ori_arg_map[basen] = n;


  father_args_map[basem] = tilem;
  father_args_map[basen] = tilen;
  
  arg_align_map[basem] = ge::Symbol(16);
  arg_align_map[basen] = ge::Symbol(16);

  arg_align_map[tilem] = ge::Symbol(256);
  arg_align_map[tilen] = ge::Symbol(16);

  const_args[constbl] = 8;

  solver_gen.SetMulticoreArgs(mc_args);
  solver_gen.SetFatherArgsMap(father_args_map);
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetArgtMaxValueMap(ori_arg_map);
  solver_gen.SetL0Args(l0_args);
  solver_gen.SetBufferUseAlg(buffer_use_map);
  solver_gen.SetConstVars(const_args);
  std::string impl_code = solver_gen.GenInitTilingData();
  EXPECT_EQ(impl_code, "Solver Gen Error");
}

TEST_F(TestL0SolverGen, TEST_MAX_ALIGN_ZERO)
{
  L0TileSolverGen solver_gen("case0", "TilingData");
  std::vector<Expr> ori_args;
  std::vector<Expr> l0_args;
  std::vector<Expr> mc_args;
  ExprUintMap const_args;
  ExprExprMap father_args_map;
  ExprExprMap arg_align_map;
  ExprExprMap ori_arg_map;

  Expr m = CreateExpr("m");
  Expr n = CreateExpr("n");
  Expr k = CreateExpr("k");
  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");
  Expr basem = CreateExpr("basem");
  Expr basen = CreateExpr("basen");
  Expr basek = CreateExpr("basek");
  Expr constbl = CreateExpr("bl");
  Expr buffer_use = ((basem * basek) + (basen * basek));

  std::map<HardwareDef, Expr> buffer_use_map;
  buffer_use_map[HardwareDef::L0A] = buffer_use;

  ori_args.emplace_back(m);
  ori_args.emplace_back(n);
  ori_args.emplace_back(k);


  l0_args.emplace_back(basem);
  l0_args.emplace_back(basen);
  l0_args.emplace_back(basek);

  mc_args.emplace_back(tilem);
  mc_args.emplace_back(tilen);

  ori_arg_map[basem] = m;
  ori_arg_map[basen] = n;
  ori_arg_map[basek] = k;

  father_args_map[basem] = tilem;
  father_args_map[basen] = tilen;
  
  arg_align_map[basem] = ge::Symbol(0);
  arg_align_map[basen] = ge::Symbol(16);
  arg_align_map[basek] = ge::Symbol(16);
  arg_align_map[tilem] = ge::Symbol(256);
  arg_align_map[tilen] = ge::Symbol(16);

  const_args[constbl] = 8;

  solver_gen.SetMulticoreArgs(mc_args);
  solver_gen.SetFatherArgsMap(father_args_map);
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetArgtMaxValueMap(ori_arg_map);
  solver_gen.SetL0Args(l0_args);
  solver_gen.SetBufferUseAlg(buffer_use_map);
  solver_gen.SetConstVars(const_args);
  std::string impl_code = solver_gen.GenInitTilingData();
  EXPECT_EQ(impl_code, "Solver Gen Error");

}