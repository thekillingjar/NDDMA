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
#include "util/duration.h"
#include "gen_model_info/test_fa_ascir_graph.h"
#include "util/option_register.h"
#include "base/att_const_values.h"

namespace att {
class OptionsUnitTest : public testing::Test {
 public:
  // 前处理：创建一个测试用的空文件
  void SetUp() override {}
  // 后处理：删除测试文件
  void TearDown() override {}
};

TEST_F(OptionsUnitTest, OptionTest01) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
  options.emplace(kOutputFilePath, kDefaultFilePath);
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "1");
  options.emplace("invalidOption01", "xxx");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName());
  EXPECT_EQ(res, false);
}

TEST_F(OptionsUnitTest, OptionTest02) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
  options.emplace(kOutputFilePath, kDefaultFilePath);
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "2");
  options.emplace(kGenConfigType, "HighPerf");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName());
  EXPECT_EQ(res, false);
}

TEST_F(OptionsUnitTest, OptionTest03) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "");
  options.emplace(kOutputFilePath, kDefaultFilePath);
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "1");
  options.emplace(kGenTilingDataDef, "1");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName());
  EXPECT_EQ(res, false);
}

TEST_F(OptionsUnitTest, OptionTest04) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "1");
  options.emplace(kOutputFilePath, "./$#@");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName());
  EXPECT_EQ(res, false);
}
TEST_F(OptionsUnitTest, OptionTest05) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "*_ttt");
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "1");
  options.emplace(kOutputFilePath, "./$#@");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName());
  EXPECT_EQ(res, false);
}
TEST_F(OptionsUnitTest, OptionTest06) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "");
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenTilingDataDef, "1");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName());
  EXPECT_EQ(res, false);
}
TEST_F(OptionsUnitTest, OptionTest07) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kDurationLevelName, "-1");
  options.emplace(kGenTilingDataDef, "1");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName());
  EXPECT_EQ(res, false);
}
TEST_F(OptionsUnitTest, OptionTest08) {
  std::vector<ge::AscGraph> graphs;
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  graphs.emplace_back(graph);
  std::map<std::string, std::string> options;
  options.emplace(kDurationLevelName, "");
  options.emplace(kGenTilingDataDef, "1");
  std::string op_name = "OpTest";
  std::map<std::string, std::string> inner_options;
  bool res = RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName());
  EXPECT_EQ(res, false);
}

TEST_F(OptionsUnitTest, TestValidateNotEmptyOption) {
  EXPECT_EQ(ValidateNotEmptyOption("test"), true);
  EXPECT_EQ(ValidateNotEmptyOption(""), false);
}
}  // namespace att