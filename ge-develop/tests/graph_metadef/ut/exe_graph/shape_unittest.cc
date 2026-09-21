/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/shape.h"
#include <gtest/gtest.h>
namespace gert {
class ShapeUT : public testing::Test {};
TEST_F(ShapeUT, DefaultConstructOk) {
  Shape s;
  EXPECT_EQ(s.GetDimNum(), 0);
}

TEST_F(ShapeUT, ConstructFromListOk) {
  Shape s{8, 3, 224, 224};
  EXPECT_EQ(s.GetDimNum(), 4);
  EXPECT_EQ(s.GetDim(0), 8);
  EXPECT_EQ(s.GetDim(1), 3);
  EXPECT_EQ(s.GetDim(2), 224);
  EXPECT_EQ(s.GetDim(3), 224);
}

TEST_F(ShapeUT, ConstructFromListOverMaxNum) {
  Shape s{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
  EXPECT_EQ(s.GetDimNum(), 0);
}

TEST_F(ShapeUT, EqualOk) {
  Shape s1{8, 3, 224, 224};
  Shape s2{8, 3, 224, 224};
  EXPECT_TRUE(s1 == s2);
  EXPECT_FALSE(s1 != s2);
}

TEST_F(ShapeUT, NotEqualForSize) {
  Shape s1{3, 224, 224};
  Shape s2{8, 3, 224, 224};
  EXPECT_FALSE(s1 == s2);
  EXPECT_TRUE(s1 != s2);
}

TEST_F(ShapeUT, NotEqualForDim) {
  Shape s1{7, 3, 224, 224};
  Shape s2{8, 3, 224, 224};
  EXPECT_FALSE(s1 == s2);
  EXPECT_TRUE(s1 != s2);
}

TEST_F(ShapeUT, GetTensorShapeSizeOk) {
  Shape s2{8, 3, 224, 224};
  EXPECT_EQ(s2.GetShapeSize(), 8 * 3 * 224 *224);
}

TEST_F(ShapeUT, GetScalerShapeSizeOk) {
  Shape s;
  EXPECT_EQ(s.GetShapeSize(), 1);
}

TEST_F(ShapeUT, Get1ShapeSizeOk) {
  Shape s{1};
  EXPECT_EQ(s.GetShapeSize(), 1);
}

TEST_F(ShapeUT, GetShapeOverflow) {
  Shape s{8, 3, 224, std::numeric_limits<int64_t>::max()};
  EXPECT_LT(s.GetShapeSize(), 0);
}

TEST_F(ShapeUT, GetEmptyTensorShapeSize) {
  Shape s2{8, 3, 224, 0};
  EXPECT_EQ(s2.GetShapeSize(), 0);
}

TEST_F(ShapeUT, IsScalarOk) {
  Shape s1;
  Shape s2{8, 3, 224, 0};
  EXPECT_TRUE(s1.IsScalar());
  EXPECT_FALSE(s2.IsScalar());
}

TEST_F(ShapeUT, GetDimNumOk) {
  Shape s1;
  Shape s2{8, 3, 224, 0};
  EXPECT_EQ(s1.GetDimNum(), 0);
  EXPECT_EQ(s2.GetDimNum(), 4);
}

TEST_F(ShapeUT, SetGetDimNumOk) {
  Shape s;
  EXPECT_EQ(s.GetDimNum(), 0);
  s.SetDimNum(4);
  EXPECT_EQ(s.GetDimNum(), 4);
}

TEST_F(ShapeUT, GetDimOk) {
  Shape s1;
  Shape s2{8, 3, 224, 224};

  EXPECT_EQ(s1.GetDim(0), 0);
  EXPECT_EQ(s2.GetDim(0), 8);
  EXPECT_EQ(s2.GetDim(1), 3);
  EXPECT_EQ(s2.GetDim(2), 224);
  EXPECT_EQ(s2.GetDim(3), 224);

  EXPECT_EQ(s1[0], 0);
  EXPECT_EQ(s2[0], 8);
  EXPECT_EQ(s2[1], 3);
  EXPECT_EQ(s2[2], 224);
  EXPECT_EQ(s2[3], 224);
}

TEST_F(ShapeUT, ModifyDimOk) {
  Shape s1;
  Shape s2{8, 3, 224, 224};

  s1[0] = 8;
  s1[1] = 8;
  EXPECT_EQ(s1.GetDim(0), 8);
  EXPECT_EQ(s1.GetDim(1), 8);

  s2[0] = 16;
  s2[1] = 16;
  EXPECT_EQ(s2.GetDim(0), 16);
  EXPECT_EQ(s2.GetDim(1), 16);
}

TEST_F(ShapeUT, SetGetDimOfOutRange) {
  Shape s1;
  EXPECT_EQ(s1.GetDim(25), std::numeric_limits<int64_t>::min());
  s1.SetDim(25, 10);
}

TEST_F(ShapeUT, SetGetDimOk) {
  Shape s{1};
  EXPECT_EQ(s.GetDim(0), 1);
  s.SetDim(0, 10);
  EXPECT_EQ(s.GetDim(0), 10);
}

TEST_F(ShapeUT, AppendDimOk) {
  Shape s{1};
  s.AppendDim(10).AppendDim(20);
  Shape expect_s{1,10,20};
  EXPECT_EQ(s, expect_s);
}

TEST_F(ShapeUT, AppendDimOutOfBounds) {
  Shape s{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
  Shape expect_s{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
  s.AppendDim(10);
  EXPECT_EQ(s, expect_s);
}

TEST_F(ShapeUT, SetScalar) {
  Shape s{1, 2, 3, 4};
  EXPECT_EQ(s.GetDimNum(), 4);
  s.SetScalar();
  EXPECT_EQ(s.GetDimNum(), 0);
}

TEST_F(ShapeUT, CopyConstruct) {
  Shape s{1, 2, 3, 4, 5};
  Shape s_copy(s);
  EXPECT_EQ(s_copy.GetDimNum(), 5);
}


TEST_F(ShapeUT, CopyAssign) {
  Shape s{4,3,2,1};
  Shape s_copy{1, 2, 3, 4, 5};
  EXPECT_EQ(s_copy.GetDimNum(), 5);
  s_copy = s;
  EXPECT_EQ(s_copy.GetDimNum(), 4);
  EXPECT_EQ(s_copy.GetDim(4), 5);

  Shape a{4,3,2,1};
  Shape a_copy{1, 2, 3, 4, 5};
  EXPECT_EQ(a.GetDimNum(), 4);
  a = a_copy;
  EXPECT_EQ(a.GetDimNum(), 5);
  EXPECT_EQ(a.GetDim(4), 5);
}
}  // namespace gert
