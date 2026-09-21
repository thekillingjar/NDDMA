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
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "codegen_infershape.h"
#include "codegen.h"

using namespace ge;
using namespace codegen;
using namespace testing;

class CodegenTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CodegenTest, GenGetKernelAndJson_ShouldReturnInvalidString_WhenKernelPathIsInvalid)
{
  codegen::CodegenOptions opt;
  codegen::Codegen codegen(opt);
  std::string kernel_path = "invalid_kernel_path";
  std::string json_path = "invalid_json_path";
  std::string result = codegen.GenGetKernelAndJson(kernel_path, json_path);
  EXPECT_EQ(result, "");
}