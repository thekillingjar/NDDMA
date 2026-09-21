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
#define private public
#include "parser/ascend_graph_parser.h"
#include "expr_gen/generate_tiling_expr.h"
#undef private

namespace att {
class TestAscendGraphParser : public ::testing::Test {
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
  }
  void TearDown() override
  {
  }
  std::shared_ptr<ge::AscGraph> graph;
};

TEST_F(TestAscendGraphParser, case_global_container)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  ge::AscTensorAttr ascir_tensor_info;
  TensorPtr tensor = std::make_shared<att::Tensor>();
  ascir_tensor_info.mem.hardware == ge::MemHardware::kMemHardwareGM;
  ascir_tensor_info.mem.reuse_id = 0;
  std::string node_type = "WorkspaceWithInput";
  EXPECT_EQ(ascend_graph_parser.ParseTensorMemInfo(ascir_tensor_info, node_type, tensor), ge::SUCCESS);
}

TEST_F(TestAscendGraphParser, get_need_ub_mc_tradeoff_1_dim)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->loc = HardwareDef::GM;
  Expr dim1 = CreateExpr("s0");
  tensor->repeat = {dim1};
  tensor->gm_stride = {ge::sym::kSymbolOne};
  NodeInfo node;
  node.outputs.emplace_back(tensor); 
  tuning_space->node_infos.emplace_back(node);
  att::GenerateTilingExpr tiling_expr(tuning_space);
  ModelInfo model_info;
  tiling_expr.UpdateNeedUBMCTradeoff(model_info);
  EXPECT_EQ(model_info.tiling_schedule_config.trade_off_config.is_enable, false);
}

TEST_F(TestAscendGraphParser, get_need_ub_mc_tradeoff)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->loc = HardwareDef::GM;
  Expr dim1 = CreateExpr("s0");
  Expr dim2 = CreateExpr("s1");
  Expr dim3 = CreateExpr("s2");
  tensor->repeat = {dim1, dim2, dim3};
  tensor->gm_stride = {dim2 * dim3, dim3 * dim2, ge::sym::kSymbolOne};
  NodeInfo node;
  node.outputs.emplace_back(tensor); 
  tuning_space->node_infos.emplace_back(node);
  att::GenerateTilingExpr tiling_expr(tuning_space);
  ModelInfo model_info;
  tiling_expr.UpdateNeedUBMCTradeoff(model_info);
  EXPECT_EQ(model_info.tiling_schedule_config.trade_off_config.is_enable, true);
}
}