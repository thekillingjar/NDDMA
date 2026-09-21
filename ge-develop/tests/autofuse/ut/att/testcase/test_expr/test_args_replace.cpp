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
#include <iostream>
#define private public
#include "generator/preprocess/args_replace.h"
using namespace att;

class ArgsReplaceUtest : public ::testing::Test {
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
  }

  void TearDown() override {
  }
  void dumps(const std::string& res)
  {
    std::cout << res << std::endl;
  }
};

VarInfo StubVarInfo(const uint32_t align,
                    const Expr& parent_size, bool do_search)
{
  VarInfo info;
  info.align = ge::Symbol(align);
  info.from_axis_size = {parent_size};
  info.do_search = do_search;
  return info;
}


// TEST_F(ArgsReplaceUtest, test_init)
// {
//   ExprInfoMap expr_info_map;
//   Expr basem = CreateExpr("basem");
//   Expr stepm = CreateExpr("stepm");
//   Expr tilem = CreateExpr("tilem");
//   std::map<std::string, std::vector<std::pair<Expr, Expr>>> eq_exprs;
//   eq_exprs["is_no_tail"].emplace_back(std::pair(stepm, basem));
//   eq_exprs[kFatherToChildNoTail].emplace_back(std::pair(stepm, basem));
//   ArgsReplacer args_replacer(expr_info_map);
//   bool res = args_replacer.InitWithEqCons(eq_exprs);
//   EXPECT_FALSE(res);
//   eq_exprs.clear();
//   args_replacer.Reset();
//   eq_exprs[kFatherToChildNoTail].emplace_back(std::pair(stepm, Mul(basem, CreateExpr(2))));
//   res = args_replacer.InitWithEqCons(eq_exprs);
//   EXPECT_TRUE(res);
//   EXPECT_TRUE(args_replacer.expr_factors_map_.empty());
//   EXPECT_TRUE(args_replacer.factor_expr_map_.empty());
//   eq_exprs.clear();
//   args_replacer.Reset();
//   eq_exprs[kFatherToChildNoTail].emplace_back(std::pair(stepm, basem));
//   res = args_replacer.InitWithEqCons(eq_exprs);
//   EXPECT_TRUE(res);
//   EXPECT_TRUE(args_replacer.expr_factors_map_.size() == 1u);
//   EXPECT_TRUE(args_replacer.expr_factors_map_.size() == args_replacer.factor_expr_map_.size());
// }

TEST_F(ArgsReplaceUtest, test_get_leaf_exprs_success_1)
{
  ExprInfoMap expr_info_map;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(16u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(16u, tilem, true));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.expr_factors_map_[stepm].emplace_back(basem);
  args_replacer.factor_expr_map_[basem] = stepm;
  args_replacer.GetLeafExprs();
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.size() == 1u);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_.size() == 1u);
  EXPECT_TRUE(args_replacer.replaced_expr_queue_.size() == 1);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_.find(basem) !=
              args_replacer.ori_expr_new_expr_map_.end());
  dumps(Str(args_replacer.ori_expr_new_expr_map_[basem]));
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.find(CreateExpr("basem_base")) !=
              args_replacer.new_expr_ori_expr_map_.end());
}

TEST_F(ArgsReplaceUtest, test_get_leaf_exprs_failed_1)
{
  ExprInfoMap expr_info_map;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(10u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(16u, tilem, true));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.expr_factors_map_[stepm].emplace_back(basem);
  args_replacer.factor_expr_map_[basem] = stepm;
  auto res = args_replacer.GetLeafExprs();
  EXPECT_FALSE(res);
}

TEST_F(ArgsReplaceUtest, test_get_leaf_exprs_success_2)
{
  ExprInfoMap expr_info_map;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(1u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(16u, tilem, true));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.expr_factors_map_[stepm].emplace_back(basem);
  args_replacer.factor_expr_map_[basem] = stepm;
  auto res = args_replacer.GetLeafExprs();
  EXPECT_TRUE(res);
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.size() == 1);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_.size() == 1);
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.find(basem) !=
              args_replacer.new_expr_ori_expr_map_.end());
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_[basem] == basem);
}

TEST_F(ArgsReplaceUtest, test_get_leaf_exprs_success_3)
{
  ExprInfoMap expr_info_map;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(16u, stepm, false));
  expr_info_map.emplace(stepm, StubVarInfo(16u, tilem, true));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.expr_factors_map_[stepm].emplace_back(basem);
  args_replacer.factor_expr_map_[basem] = stepm;
  auto res = args_replacer.GetLeafExprs();
  EXPECT_TRUE(res);
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.size() == 1);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_.size() == 1);
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.find(basem) !=
              args_replacer.new_expr_ori_expr_map_.end());
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_[basem] == basem);
}

TEST_F(ArgsReplaceUtest, test_get_leaf_exprs_failed_2)
{
  ExprInfoMap expr_info_map;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(16u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(16u, tilem, true));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.expr_factors_map_[stepm].emplace_back(basem);
  args_replacer.factor_expr_map_[basem] = stepm;
  args_replacer.ori_expr_new_expr_map_[basem] = stepm;
  auto res = args_replacer.GetLeafExprs();
  EXPECT_FALSE(res);
}

TEST_F(ArgsReplaceUtest, test_get_leaf_exprs_failed_3)
{
  ExprInfoMap expr_info_map;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.expr_factors_map_[stepm].emplace_back(basem);
  args_replacer.factor_expr_map_[basem] = stepm;
  auto res = args_replacer.GetLeafExprs();
  EXPECT_FALSE(res);
}

TEST_F(ArgsReplaceUtest, test_align_replace)
{
  ExprInfoMap expr_info_map;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(1u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(16u, tilem, true));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.GetAlignReplacedVars(basem);
  args_replacer.GetAlignReplacedVars(stepm);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_.size() == 2);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_[basem] == (basem));
  dumps(Str(args_replacer.ori_expr_new_expr_map_[basem]));
  EXPECT_FALSE(args_replacer.ori_expr_new_expr_map_[stepm] == (stepm));
  dumps(Str(args_replacer.ori_expr_new_expr_map_[stepm]));
}

TEST_F(ArgsReplaceUtest, test_factor_replace_1)
{
  ExprInfoMap expr_info_map;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(1u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(16u, tilem, true));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.expr_factors_map_[stepm].emplace_back(basem);
  args_replacer.factor_expr_map_[basem] = stepm;
  args_replacer.ori_expr_new_expr_map_[basem] = basem;
  args_replacer.GetFactorReplacedVars(stepm);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_.find(stepm) !=
              args_replacer.ori_expr_new_expr_map_.end());
  Expr root_replaces_vars = args_replacer.ori_expr_new_expr_map_[stepm];
  dumps(Str(root_replaces_vars));
  args_replacer.ori_expr_new_expr_map_.clear();
  args_replacer.expr_factors_map_[tilem].emplace_back(stepm);
  args_replacer.factor_expr_map_[stepm] = tilem;
  args_replacer.ori_expr_new_expr_map_[basem] = basem;
  args_replacer.GetFactorReplacedVars(stepm);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_.find(stepm) !=
              args_replacer.ori_expr_new_expr_map_.end());
  Expr no_root_replaces_vars = args_replacer.ori_expr_new_expr_map_[stepm];
  dumps(Str(no_root_replaces_vars));
  EXPECT_FALSE(root_replaces_vars == (no_root_replaces_vars));
}

TEST_F(ArgsReplaceUtest, test_replace_parent_exprs)
{
  ExprInfoMap expr_info_map;
  Expr expr_null;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(1u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(256u, tilem, true));
  expr_info_map.emplace(tilem, StubVarInfo(16u, expr_null, true));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.expr_factors_map_[stepm].emplace_back(basem);
  args_replacer.factor_expr_map_[basem] = stepm;
  args_replacer.expr_factors_map_[tilem].emplace_back(stepm);
  args_replacer.factor_expr_map_[stepm] = tilem;
  args_replacer.GetSelfReplacedVars(basem);
  auto res = args_replacer.ReplaceParentExprs();
  EXPECT_TRUE(res);
  EXPECT_EQ(args_replacer.ori_expr_new_expr_map_.size(), 3);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_[basem] == (basem));
  dumps(Str(args_replacer.ori_expr_new_expr_map_[stepm]));
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.find(CreateExpr("stepm_base"))
              != args_replacer.new_expr_ori_expr_map_.end());
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.find(CreateExpr("tilem_div_align"))
              != args_replacer.ori_expr_new_expr_map_.end());
  dumps(Str(args_replacer.ori_expr_new_expr_map_[tilem]));
}

TEST_F(ArgsReplaceUtest, test_replace_parent_exprs_2)
{
  ExprInfoMap expr_info_map;
  Expr expr_null;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(1u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(256u, tilem, true));
  expr_info_map.emplace(tilem, StubVarInfo(16u, expr_null, false));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.expr_factors_map_[stepm].emplace_back(basem);
  args_replacer.factor_expr_map_[basem] = stepm;
  args_replacer.expr_factors_map_[tilem].emplace_back(stepm);
  args_replacer.factor_expr_map_[stepm] = tilem;
  args_replacer.GetSelfReplacedVars(basem);
  auto res = args_replacer.ReplaceParentExprs();
  EXPECT_TRUE(res);
  EXPECT_EQ(args_replacer.ori_expr_new_expr_map_.size(), 3);
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_[basem] == (basem));
  dumps(Str(args_replacer.ori_expr_new_expr_map_[stepm]));
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.find(CreateExpr("stepm_base"))
              != args_replacer.new_expr_ori_expr_map_.end());
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_.find(tilem)
              != args_replacer.ori_expr_new_expr_map_.end());
  EXPECT_TRUE(args_replacer.new_expr_ori_expr_map_[tilem] == (tilem));
}

TEST_F(ArgsReplaceUtest, test_replace_naive_exprs)
{
  ExprInfoMap expr_info_map;
  Expr expr_null;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(1u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(256u, tilem, true));
  expr_info_map.emplace(tilem, StubVarInfo(16u, expr_null, false));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.ReplaceNaiveExpr();
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_[basem] == (basem));
  EXPECT_TRUE(args_replacer.ori_expr_new_expr_map_[tilem] == (tilem));
}

TEST_F(ArgsReplaceUtest, test_get_result)
{
  ExprInfoMap expr_info_map;
  Expr expr_null;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(1u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(256u, tilem, true));
  expr_info_map.emplace(tilem, StubVarInfo(16u, expr_null, false));
  ArgsReplacer args_replacer(expr_info_map);
  args_replacer.ReplaceNaiveExpr();
  ExprExprMap replaced_vars;
  ExprExprMap replacements;
  ExprExprMap new_expr_replacements;
  args_replacer.GetReplaceResult(replaced_vars, replacements, new_expr_replacements);
  EXPECT_EQ(replacements.size(), 1u);
  EXPECT_EQ(new_expr_replacements.size(), 1u);
  EXPECT_TRUE(replacements.find(stepm) != replacements.end());
}


TEST_F(ArgsReplaceUtest, test_do_replace)
{
  ExprInfoMap expr_info_map;
  Expr expr_null;
  Expr basem = CreateExpr("basem");
  Expr stepm = CreateExpr("stepm");
  Expr tilem = CreateExpr("tilem");
  expr_info_map.emplace(basem, StubVarInfo(1u, stepm, true));
  expr_info_map.emplace(stepm, StubVarInfo(256u, tilem, true));
  expr_info_map.emplace(tilem, StubVarInfo(16u, expr_null, false));
  ArgsReplacer args_replacer(expr_info_map);
  std::map<std::string, std::vector<std::pair<Expr, Expr>>> eq_exprs;
  eq_exprs[kFatherToChildNoTail].emplace_back(std::pair(stepm, basem));
  auto res = args_replacer.DoReplace(eq_exprs);
  EXPECT_TRUE(res);
  ExprExprMap replaced_vars;
  ExprExprMap replacements;
  ExprExprMap new_expr_replacements;
  args_replacer.GetReplaceResult(replaced_vars, replacements, new_expr_replacements);
  EXPECT_EQ(replacements.size(), 1u);
  EXPECT_EQ(new_expr_replacements.size(), 1u);
  EXPECT_TRUE(replacements.find(stepm) != replacements.end());
}
