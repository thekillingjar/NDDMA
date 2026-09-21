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
#include <cstdio>
#include "../common/code_printer.h"

class CodeprinterTest: public testing::Test {
 public:
  // 前处理：创建一个测试用的空文件
  void SetUp() override {}
  // 后处理：删除测试文件
  void TearDown() override {}
};

TEST_F(CodeprinterTest, test_code_printer){
  ge::CodePrinter code_printer;
  code_printer.AddInclude("iostream");
  code_printer.AddNamespaceBegin("std");
  code_printer.DefineClassBegin("MyPrinter");
  code_printer.DefineFuncBegin("void", "Out", "const std::string& str");
  code_printer.AddLine("cout << str << endl;");
  code_printer.DefineFuncEnd();
  code_printer.DefineClassEnd();
  code_printer.DefineFuncBegin("int", "main", "");
  code_printer.AddLine("MyPrinter printer;");
  code_printer.AddLine("printer.Out(\"Hello world.\");");
  code_printer.DefineFuncEnd();
  code_printer.AddNamespaceEnd("std");
  code_printer.SaveToFile("print_test.cpp");
  auto file_content = code_printer.GetOutputStr();
  EXPECT_TRUE(!file_content.empty());
  code_printer.Reset();
  file_content = code_printer.GetOutputStr();
  EXPECT_TRUE(file_content.empty());
  auto res = std::remove("print_test.cpp");
  EXPECT_TRUE(res == 0);
}
