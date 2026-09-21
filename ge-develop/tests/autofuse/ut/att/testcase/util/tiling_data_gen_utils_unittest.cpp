/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include "base/att_const_values.h"
#include "generator_utils/tilingdata_gen_utils.h"
namespace att {
class TilingDataGenUtilsUnitTest : public testing::Test {
 public:
  // 前处理：创建一个测试用的空文件
  void SetUp() override {}
  // 后处理：删除测试文件
  void TearDown() override {
  }
};

TEST_F(TilingDataGenUtilsUnitTest, NeedWrittenTilingDataFalse) {
  std::vector<std::string> tiling_data_vars;
  std::set<std::string> var_names;
  EXPECT_FALSE(TilingDataGenUtils::NeedWrittenTilingData(tiling_data_vars, var_names));
}

TEST_F(TilingDataGenUtilsUnitTest, WriteTilingDataElementTrue) {
  ge::CodePrinter printer;
  std::set<std::string> tiling_data_vars;
  std::set<std::string> var_names;
  var_names.insert("new_var");
  TilingDataGenUtils::WriteTilingDataElement(printer, tiling_data_vars, var_names);
  ASSERT_TRUE(tiling_data_vars.size() == 1L);
  EXPECT_TRUE(*tiling_data_vars.begin() == "new_var");
}
} //namespace