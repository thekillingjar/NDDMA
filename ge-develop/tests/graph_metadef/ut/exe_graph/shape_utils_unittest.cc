/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/shape_utils.h"
#include <gtest/gtest.h>
namespace gert {
class ShapeUtilsUT : public testing::Test {};
TEST_F(ShapeUtilsUT, EnsureNotScalar_ReturnInput_NotScalarShape) {
  Shape s1{1, 2, 3};
  Shape s2{1};
  Shape s3{0, 1};
  ASSERT_EQ(&EnsureNotScalar(s1), &s1);
  ASSERT_EQ(&EnsureNotScalar(s2), &s2);
  ASSERT_EQ(&EnsureNotScalar(s3), &s3);
}
TEST_F(ShapeUtilsUT, EnsureNotScalar_ReturnVec1_ScalarShape) {
  Shape s1{};
  auto &s = EnsureNotScalar(s1);
  ASSERT_FALSE(s.IsScalar());
  ASSERT_EQ(s.GetDimNum(), 1);
  ASSERT_EQ(s.GetDim(0), 1);
}
TEST_F(ShapeUtilsUT, ShapeToString_Empty_Scalar) {
  Shape s;
  EXPECT_TRUE(ShapeToString(s).empty());
}
TEST_F(ShapeUtilsUT, ShapeToString_NoComman_OneDim) {
  Shape s{1};
  EXPECT_EQ(ShapeToString(s), "1");
}
TEST_F(ShapeUtilsUT, ShapeToString_DefaulJoinStr) {
  Shape s{1, 2, 3};
  EXPECT_EQ(ShapeToString(s), "1,2,3");
}
TEST_F(ShapeUtilsUT, ShapeToString_SelfDefinedStr) {
  Shape s{1, 2, 3};
  EXPECT_EQ(ShapeToString(s, ", "), "1, 2, 3");
}
TEST_F(ShapeUtilsUT, ShapeToString_UseComman_NullJoinStr) {
  Shape s{1, 2, 3};
  EXPECT_EQ(ShapeToString(s, nullptr), "1,2,3");
}

TEST_F(ShapeUtilsUT, CalcAlignedSizeByShape_success) {
  Shape s{1, 2, 3};
  size_t ret_tensor_size = 0U;
  CalcAlignedSizeByShape(s, ge::DataType::DT_FLOAT, ret_tensor_size);
  EXPECT_EQ(ret_tensor_size, 64);

  CalcAlignedSizeByShape(s, ge::DataType::DT_FLOAT16, ret_tensor_size);
  EXPECT_EQ(ret_tensor_size, 64);

  CalcAlignedSizeByShape(s, ge::DataType::DT_STRING, ret_tensor_size);
  EXPECT_EQ(ret_tensor_size, 128);

  CalcAlignedSizeByShape(s, ge::DataType::DT_INT4, ret_tensor_size);
  EXPECT_EQ(ret_tensor_size, 64);
}

TEST_F(ShapeUtilsUT, CalcTotalSizeByShape_failed) {
  Shape s{-1, 2, 3};
  size_t ret_tensor_size = 0U;

  EXPECT_NE(CalcAlignedSizeByShape(s, ge::DataType::DT_STRING, ret_tensor_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret_tensor_size, 0);

  Shape s1{std::numeric_limits<int64_t>::max(), 2, 3};
  EXPECT_NE(CalcAlignedSizeByShape(s1, ge::DataType::DT_STRING, ret_tensor_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret_tensor_size, 0);
}
}  // namespace gert
