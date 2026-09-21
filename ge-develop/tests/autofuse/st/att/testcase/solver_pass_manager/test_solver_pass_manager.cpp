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
#include <gmock/gmock.h>
#include "base/base_types.h"
#include "base/model_info.h"
#include "generator/solver_pass_gen/solver_pass_manager.h"
#include "stub/stub_model_info.h"
#include "tiling_code_generator.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "test_common_utils.h"
using namespace att;

class TestSolverPassManager : public ::testing::Test {
 public:
  void TearDown() override {
     // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
     // before the destructor).
  }
};

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
  std::string impl_code = res.first;
  std::string invoke_code = res.second;
  EXPECT_NE(base_class_head, "");
  EXPECT_NE(base_class_func, "");
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}

TEST_F(TestSolverPassManager, NormalStaticUint32Shape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(ge::ExprType::kExprConstantInteger);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  att::TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode("OpTest", model_infos, config, tiling_res), ge::SUCCESS);
  ASSERT_EQ(tiling_res.size(), 4);
}

TEST_F(TestSolverPassManager, NormalStaticRationShape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(ge::ExprType::kExprConstantRation);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode("OpTest", model_infos, config, tiling_res), ge::SUCCESS);
  ASSERT_EQ(tiling_res.size(), 4);
}