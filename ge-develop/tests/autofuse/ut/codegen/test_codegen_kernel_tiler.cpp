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
#include "ascir_ops.h"
#include "codegen_kernel.h"

using namespace ge::ops;
using namespace codegen;
using namespace ge::ascir_op;
using namespace testing;


class TilerTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};


/**
 * @tc.name  : Size_ShouldReturnConstExpr_WhenSizeIsConstExpr
 * @tc.number: Size_Test_001
 * @tc.desc  : Test Size method when size is a const expression.
 */
TEST_F(TilerTest, Size_ShouldReturnConstExpr_WhenSizeIsConstExpr) {
  ascir::SizeExpr size = ge::Symbol(10);
  codegen::Tiler tiler;
  EXPECT_EQ(tiler.Size(size, false), "10");
}

/**
 * @tc.name  : Size_ShouldReturnIntCastedConstExpr_WhenSizeIsRational
 * @tc.number: Size_Test_002
 * @tc.desc  : Test Size method when size is a rational const expression.
 */
TEST_F(TilerTest, Size_ShouldReturnIntCastedConstExpr_WhenSizeIsRational) {
  ascir::SizeExpr size = ge::sym::Rational(3, 2);
  codegen::Tiler tiler;
  EXPECT_EQ(tiler.Size(size, false), "(3)/(2)");
}

/**
 * @tc.name  : ActualSize_ShouldReturnConstExpr_WhenSizeIsConstExpr
 * @tc.number: ActualSize_Test_001
 * @tc.desc  : Test ActualSize method when size is a const expression.
 */
TEST_F(TilerTest, ActualSize_ShouldReturnConstExpr_WhenSizeIsConstExpr) {
  ascir::SizeExpr size = ge::Symbol(10);
  codegen::Tiler tiler;
  EXPECT_EQ(tiler.ActualSize(size, false), "10");
}

/**
 * @tc.name  : ActualSize_ShouldReturnIntCastedConstExpr_WhenSizeIsRational
 * @tc.number: ActualSize_Test_001
 * @tc.desc  : Test ActualSize method when size is a rational const expression.
 */
TEST_F(TilerTest, ActualSize_ShouldReturnIntCastedConstExpr_WhenSizeIsRational) {
  ascir::SizeExpr size = ge::sym::Rational(3, 2);
  codegen::Tiler tiler;
  EXPECT_EQ(tiler.ActualSize(size, false), "(3)/(2)");
}