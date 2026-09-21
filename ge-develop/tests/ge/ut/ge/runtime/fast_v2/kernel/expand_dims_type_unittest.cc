/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/expand_dims_type.h"
#include <gtest/gtest.h>
namespace gert {
class ExpandDimsTypeUT : public testing::Test {};
TEST_F(ExpandDimsTypeUT, TestDoNotExpandSpecifyAll) {
  auto shape = Shape{2, 3, 5};
  ExpandDimsType edt("000");
  Shape out_shape;
  edt.Expand(shape, out_shape);
  ASSERT_EQ(out_shape, shape);
}
TEST_F(ExpandDimsTypeUT, TestDoNotExpandSpecifyPart1) {
  auto shape = Shape{2, 3, 5};
  ExpandDimsType edt("0");
  Shape out_shape;
  edt.Expand(shape, out_shape);
  ASSERT_EQ(out_shape, shape);
}
TEST_F(ExpandDimsTypeUT, TestDoNotExpandSpecifyPart2) {
  auto shape = Shape{2, 3, 5};
  ExpandDimsType edt("0");
  Shape out_shape;
  edt.Expand(shape, out_shape);
  ASSERT_EQ(out_shape, shape);
}
TEST_F(ExpandDimsTypeUT, TestDoNotExpandSpecifyNone) {
  auto shape = Shape{2, 3, 5};
  ExpandDimsType edt("");
  Shape out_shape;
  edt.Expand(shape, out_shape);
  ASSERT_EQ(out_shape, shape);
}

TEST_F(ExpandDimsTypeUT, ExpandAtHead) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("11000");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 1, 2, 16, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandAtHeadSpecifyPart) {
  auto shape = Shape{2, 16};
  ExpandDimsType edt("110");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(4, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 1, 2, 16}));
}

TEST_F(ExpandDimsTypeUT, ExpandAtHeadSpecifyNone) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("11");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 1, 2, 16, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandAtHeadSpecifyNone1) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("1");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(4, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 2, 16, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandAtLast) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("00011");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 16, 16, 1, 1}));
}
TEST_F(ExpandDimsTypeUT, ExpandAtLast3Dim) {
  auto shape = Shape{2};
  ExpandDimsType edt("0111");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(4, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 1, 1, 1}));
}
TEST_F(ExpandDimsTypeUT, ExpandHeadAndLast) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("10001");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({1, 2, 16, 16, 1}));
}
TEST_F(ExpandDimsTypeUT, ExpandMiddle) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("01010");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(5, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 1, 16, 1, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandMiddleSpecifyPart) {
  auto shape = Shape{2, 16};
  ExpandDimsType edt("011");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(4, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 1, 1, 16}));
}
TEST_F(ExpandDimsTypeUT, ExpandSpecifyPart) {
  auto shape = Shape{2, 16, 16};
  ExpandDimsType edt("100");
  Shape out_shape;
  edt.Expand(shape, out_shape);

  ASSERT_EQ(3, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({2, 16, 16}));
}
}  // namespace gert
