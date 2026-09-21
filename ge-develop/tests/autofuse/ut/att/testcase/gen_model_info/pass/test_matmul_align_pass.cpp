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
#include "pass/pass_mgr.h"
#include "parser/tuning_space.h"
#include "gen_model_info/pass/matmul_align_pass.cpp"

namespace att {
class TestMatmulAlignPass : public ::testing::Test {
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
};
bool TestPass(const TuningSpacePtr &tuning_space, std::map<std::string, std::string> &matmul_config) {
  return true;
}
static std::string test_pass = "test_pass";
REGISTER_GTC_PASS(test_pass, TestPass);

TEST_F(TestMatmulAlignPass, register_pass) {
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  std::map<std::string, std::string> matmul_config;
  auto ret = ATTPassMgr::Instance().GetPass(test_pass)(tuning_space, matmul_config);
  std::vector<PassFunc> res;
  ATTPassMgr::Instance().GetPassList(res);
  EXPECT_NE(res.size(), 0);
}

TEST_F(TestMatmulAlignPass, output_empty) {
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  NodeInfo node_info;
  node_info.trans_config = "test";
  node_info.node_type = "loadTcsm";
  tuning_space->node_infos.emplace_back(node_info);
  std::map<std::string, std::string> matmul_config;
  auto ret = ATTPassMgr::Instance().GetPass(kmatmul_align_pass)(tuning_space, matmul_config);
  EXPECT_EQ(ret, false);
}

TEST_F(TestMatmulAlignPass, no_loadTcsm_node) {
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  NodeInfo node_info;
  node_info.node_type = "test";
  node_info.trans_config = "test";
  tuning_space->node_infos.emplace_back(node_info);
  std::map<std::string, std::string> matmul_config;
  auto ret = ATTPassMgr::Instance().GetPass(kmatmul_align_pass)(tuning_space, matmul_config);
  EXPECT_EQ(ret, false);
}

TEST_F(TestMatmulAlignPass, dim_empty) {
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  NodeInfo node_info;
  node_info.node_type = "loadTcsm";
  node_info.trans_config = "test";
  TensorPtr output_tensor = std::make_shared<Tensor>();
  EXPECT_NE(output_tensor, nullptr);
  node_info.outputs.emplace_back(output_tensor);
  tuning_space->node_infos.emplace_back(node_info);
  std::map<std::string, std::string> matmul_config;
  auto ret = ATTPassMgr::Instance().GetPass(kmatmul_align_pass)(tuning_space, matmul_config);
  EXPECT_EQ(ret, true);
}

TEST_F(TestMatmulAlignPass, success) {
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  NodeInfo node_info;
  node_info.trans_config = "test";
  node_info.node_type = "loadTcsm";
  TensorPtr output_tensor = std::make_shared<Tensor>();
  EXPECT_NE(output_tensor, nullptr);
  SubAxisPtr axis1 = std::make_unique<SubAxis>();
  EXPECT_NE(axis1, nullptr);
  axis1->name = "axis1";
  SubAxisPtr axis2 = std::make_unique<SubAxis>();
  EXPECT_NE(axis2, nullptr);
  axis2->name = "axis2";
  output_tensor->dim_info.emplace_back(axis1.get());
  output_tensor->dim_info.emplace_back(axis2.get());
  node_info.outputs.emplace_back(output_tensor);
  tuning_space->node_infos.emplace_back(node_info);
  std::map<std::string, std::string> matmul_config;
  auto ret = ATTPassMgr::Instance().GetPass(kmatmul_align_pass)(tuning_space, matmul_config);
  EXPECT_EQ(ret, true);
}
}