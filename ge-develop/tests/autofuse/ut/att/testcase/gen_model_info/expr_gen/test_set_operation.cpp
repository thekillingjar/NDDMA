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
#include "expr_gen/set_operation.h"

namespace att{
class TestSetOperation : public ::testing::Test {
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
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }
};

TEST_F(TestSetOperation, case0)
{
  DimRange dim;
  dim.lower_bound = CreateExpr(0);
  dim.upper_bound = CreateExpr(0);
  Coordinates coord = {dim};
  TensorRange range = {coord};
  SetOperation set_operation;
  Expr res = set_operation.SetComputation(range);
  EXPECT_TRUE(res == 0);

  DimRange dim1;
  dim1.lower_bound = CreateExpr(0);
  dim1.upper_bound = CreateExpr(0);
  auto new_dimranges = set_operation.Diff(dim, dim1);
  EXPECT_TRUE(new_dimranges.empty());

  DimRange dim2;
  dim2.lower_bound = CreateExpr(1);
  dim2.upper_bound = CreateExpr(1);
  new_dimranges = set_operation.Diff(dim2, dim1);
  EXPECT_EQ(new_dimranges.size(), 2);
}
} // namespace att