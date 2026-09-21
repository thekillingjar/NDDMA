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
#include "gen_model_info.h"
#include "test_fa_ascir_graph.h"
#include "parser/ascend_graph_parser.h"
#include "expr_gen/arg_list_manager.h"
#include "graph_construct_utils.h"
#define private public
#include "expr_gen/generate_tiling_expr.h"
#undef private

namespace att {
static TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
class TestGenerateTilingExpr : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override
  {
    ge::AscGraph graph("graph");
    att::FaBeforeAutoFuse(graph);
    att::FaAfterScheduler(graph);
    att::FaAfterQueBufAlloc(graph);
    GraphConstructUtils::UpdateGraphVectorizedStride(graph);

    EXPECT_NE(tuning_space, nullptr);
    att::AscendGraphParser ascend_graph_parser(tuning_space);
    EXPECT_EQ(ascend_graph_parser.GraphParser(graph), ge::SUCCESS);
    EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(tuning_space), ge::SUCCESS);
  }
  void TearDown() override
  {
  }
};

TEST_F(TestGenerateTilingExpr, case1)
{
  std::map<HardwareDef, Expr> hardware_cons;
  GenerateTilingExpr generate_expr(tuning_space);
  EXPECT_EQ(generate_expr.GetCoreConstraint(hardware_cons), ge::SUCCESS);
}

TEST_F(TestGenerateTilingExpr, case2)
{
  std::vector<AttAxisPtr> arg_lists;
  GenerateTilingExpr generate_expr(tuning_space);
  EXPECT_EQ(generate_expr.GetSubAxisArgs(arg_lists), ge::SUCCESS);
}

TEST_F(TestGenerateTilingExpr, case3)
{
  std::map<std::string, std::vector<std::pair<Expr, Expr>>> eq_exprs;
  std::map<std::string, std::vector<Expr>> leq_exprs;
  GenerateTilingExpr generate_expr(tuning_space);
  generate_expr.GetAxisConstraints(eq_exprs, leq_exprs);
  EXPECT_TRUE(!eq_exprs.empty());
  EXPECT_TRUE(!leq_exprs.empty());
}

TEST_F(TestGenerateTilingExpr, case4)
{
  std::map<HardwareDef, Expr> hardware_cons;
  Expr workspaceSize;
  GenerateTilingExpr generate_expr(tuning_space);
  EXPECT_EQ(generate_expr.GetCoreConstraint(hardware_cons), ge::SUCCESS);
}

TEST_F(TestGenerateTilingExpr, case5)
{
  std::map<std::string, Expr> tensor_exprs;
  GenerateTilingExpr generate_expr(tuning_space);
  EXPECT_EQ(generate_expr.GetTensorExpr(tensor_exprs), ge::SUCCESS);
}
}