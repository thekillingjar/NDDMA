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
#include "generator/solver_pass_gen/l0_solver/l0_solver_gen.h"
#include <symengine/functions.h>
#include <symengine/simplify.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>
#include "test_common_utils.h"
using namespace att;

class TestL0SolverGen : public ::testing::Test {
 public:
  void TearDown() override {
     // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
     // before the destructor).
  }
};

TEST_F(TestL0SolverGen, TEST_CASE_01)
{
  L0TileSolverGen solver_gen("case0", "TilingData");
  std::vector<Expr> ori_args;
  std::vector<Expr> l0_args;
  std::vector<Expr> mc_args;
  std::map<Expr, Expr, ExprCmp> father_args_map;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, Expr, ExprCmp> ori_arg_map;

  Expr m = CreateExpr("m");
  Expr n = CreateExpr("n");
  Expr k = CreateExpr("k");
  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");
  Expr basem = CreateExpr("basem");
  Expr basen = CreateExpr("basen");
  Expr basek = CreateExpr("basek");
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

  solver_gen.SetMulticoreArgs(mc_args);
  solver_gen.SetFatherArgsMap(father_args_map);
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetArgtMaxValueMap(ori_arg_map);
  solver_gen.SetL0Args(l0_args);
  solver_gen.SetBufferUseAlg(buffer_use_map);
  std::string impl_code = solver_gen.GenSolverFuncImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  // std::cout<<impl_code<<std::endl;
  // std::cout<<invoke_code<<std::endl;
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}


TEST_F(TestL0SolverGen, TEST_CASE_02)
{
  L0TileSolverGen solver_gen("case1", "TilingData");
  std::vector<Expr> ori_args;
  std::vector<Expr> l0_args;
  std::vector<Expr> mc_args;
  std::map<Expr, Expr, ExprCmp> father_args_map;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, Expr, ExprCmp> ori_arg_map;

  Expr m = CreateExpr("m");
  Expr n = CreateExpr("n");
  Expr k = CreateExpr("k");
  Expr tilem = CreateExpr("tilem");
  Expr tilen = CreateExpr("tilen");
  Expr basem = CreateExpr("basem");
  Expr basen = CreateExpr("basen");
  Expr basek = CreateExpr("basek");
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
  
  arg_align_map[basem] = ge::Symbol(256);
  arg_align_map[basen] = ge::Symbol(16);
  arg_align_map[basek] = ge::Symbol(16);
  arg_align_map[tilem] = ge::Symbol(16);
  arg_align_map[tilen] = ge::Symbol(16);

  solver_gen.SetMulticoreArgs(mc_args);
  solver_gen.SetFatherArgsMap(father_args_map);
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetArgtMaxValueMap(ori_arg_map);
  solver_gen.SetL0Args(l0_args);
  solver_gen.SetBufferUseAlg(buffer_use_map);
  std::string impl_code = solver_gen.GenSolverFuncImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}
