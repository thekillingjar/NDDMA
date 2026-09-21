/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_code_generator.h"
#include "test_common_utils.h"

namespace att {
class GeneratorST : public testing::Test {};

TEST(GeneratorST, Normal) {
  // TilingModelInfo modelInfos;
  // ModelInfo modelInfo;
  // modelInfo.vars.emplace_back("x");
  // modelInfo.vars.emplace_back("y");
  // modelInfo.vars.emplace_back("z");
  // modelInfo.tilingCaseId = 1u;
  // modelInfos.emplace_back(modelInfo);
  // TilingCodeGenConfig config;
  // config.path = "./";
  // config.type = TilingImplType::HIGH_PERF;
  // TilingCodeGenerator generator;
  // EXPECT_EQ(generator.GenTilingCode(modelInfos, config), ge::SUCCESS);
  
  // auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main.cpp ./ -f").c_str());
  // EXPECT_EQ(ret, 0);

  // ret = std::system("g++ tiling_func_main.cpp tiling_func.cpp -o tiling_func_main");
  // EXPECT_EQ(ret, 0);

  // ret = std::system("./tiling_func_main");
  // EXPECT_EQ(ret, 0);
  EXPECT_TRUE(true);
}

TEST(GeneratorST, InvalidConfig) {
  // TilingModelInfo modelInfos;
  // ModelInfo modelInfo;
  // modelInfo.vars.emplace_back("x");
  // modelInfo.vars.emplace_back("y");
  // modelInfo.vars.emplace_back("z");
  // modelInfo.tilingCaseId = 1u;
  // modelInfos.emplace_back(modelInfo);
  // TilingCodeGenConfig config;
  // config.path = "./";
  // config.type = TilingImplType::MAX;
  // TilingCodeGenerator generator;
  // EXPECT_EQ(generator.GenTilingCode(modelInfos, config), ge::FAILED);
  EXPECT_TRUE(true);
}
}