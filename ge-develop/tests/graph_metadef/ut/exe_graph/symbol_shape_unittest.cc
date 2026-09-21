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
#include "exe_graph/runtime/symbolic_tensor.h"
#include "graph/symbolizer/symbolic.h"
#include "expression/const_values.h"
#include "graph/ge_tensor.h"

namespace gert {
class SymbolShapeUT : public testing::Test {};
TEST_F(SymbolShapeUT, DefaultConstructOk) {
  SymbolShape s;
  EXPECT_EQ(s.GetDimNum(), 0);
}

TEST_F(SymbolShapeUT, ConstructFromListOk) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape s{s0, s1, s2, s3};
  EXPECT_EQ(s.GetDimNum(), 4);
  EXPECT_EQ(s.GetDim(0), s0);
  EXPECT_EQ(s.GetDim(1), s1);
  EXPECT_EQ(s.GetDim(2), s2);
  EXPECT_EQ(s.GetDim(3), s3);

  EXPECT_EQ(s.GetDims()[0], s0);
  EXPECT_EQ(s.GetDims()[1], s1);
  EXPECT_EQ(s.GetDims()[2], s2);
  EXPECT_EQ(s.GetDims()[3], s3);
}

TEST_F(SymbolShapeUT, ConstructMaxNum) {
  auto s0 = ge::Symbol("s0");
  SymbolShape s{s0, s0, s0, s0, s0, s0, s0, s0, s0, s0,
                s0, s0, s0, s0, s0, s0, s0, s0, s0, s0,
                s0, s0, s0, s0, s0, s0};
  EXPECT_EQ(s.GetDimNum(), 26);
}

TEST_F(SymbolShapeUT, EqualOk) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape1{s0, s1, s2, s3};
  SymbolShape symbol_shape2{s0, s1, s2, s3};

  EXPECT_TRUE(symbol_shape1 == symbol_shape2);
  EXPECT_FALSE(symbol_shape1 != symbol_shape2);
}

TEST_F(SymbolShapeUT, NotEqualForSize) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape1{s0, s1, s2};
  SymbolShape symbol_shape2{s0, s1, s2, s3};
  EXPECT_FALSE(symbol_shape1 == symbol_shape2);
  EXPECT_TRUE(symbol_shape1 != symbol_shape2);
}

TEST_F(SymbolShapeUT, NotEqualForDim) {
  auto s0 = ge::Symbol("s0");
  auto s0_bak = ge::Symbol(9);
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape1{s0, s1, s2, s3};
  SymbolShape symbol_shape2{s0_bak, s1, s2, s3};
  EXPECT_FALSE(symbol_shape1 == symbol_shape2);
  EXPECT_TRUE(symbol_shape1 != symbol_shape2);
}

TEST_F(SymbolShapeUT, GetTensorShapeSizeOk) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape1{s0, s1, s2, s3};
  auto expr = ge::sym::Mul(ge::sym::Mul(ge::sym::Mul(s0, s1), s2), s3);
  EXPECT_EQ(symbol_shape1.GetSymbolShapeSize(), expr);
}

TEST_F(SymbolShapeUT, GetScalerShapeSizeOk) {
  SymbolShape s;
  EXPECT_EQ(s.GetSymbolShapeSize(), ge::sym::kSymbolOne);
}

TEST_F(SymbolShapeUT, Get1ShapeSizeOk) {
  auto s0 = ge::Symbol("s0");
  SymbolShape s{s0};
  EXPECT_EQ(s.GetSymbolShapeSize(), ge::Symbol("s0"));
}

TEST_F(SymbolShapeUT, GetEmptyTensorShapeSize) {
  auto s0 = ge::Symbol(0);
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape{s0, s1, s2, s3};
  EXPECT_EQ(symbol_shape.GetSymbolShapeSize(), ge::sym::kSymbolZero);
}

TEST_F(SymbolShapeUT, IsScalarOk) {
  SymbolShape symbol_shape1;
  auto s0 = ge::Symbol(0);
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape2{s0, s1, s2, s3};
  EXPECT_TRUE(symbol_shape1.IsScalar());
  EXPECT_FALSE(symbol_shape2.IsScalar());
}

TEST_F(SymbolShapeUT, GetDimNumOk) {
  SymbolShape symbol_shape1;
  auto s0 = ge::Symbol(0);
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape2{s0, s1, s2, s3};
  EXPECT_EQ(symbol_shape1.GetDimNum(), 0);
  EXPECT_EQ(symbol_shape2.GetDimNum(), 4);
}

TEST_F(SymbolShapeUT, GetDimOk) {
  SymbolShape symbol_shape1;
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape2{s0, s1, s2, s3};

  EXPECT_EQ(symbol_shape1.GetDims().size(), 0);

  EXPECT_EQ(symbol_shape2.GetDims()[0], s0);
  EXPECT_EQ(symbol_shape2.GetDims()[1], s1);
  EXPECT_EQ(symbol_shape2.GetDims()[2], s2);
  EXPECT_EQ(symbol_shape2.GetDims()[3], s3);

  EXPECT_EQ(symbol_shape2.MutableDims()[0], s0);
  EXPECT_EQ(symbol_shape2.MutableDims()[1], s1);
  EXPECT_EQ(symbol_shape2.MutableDims()[2], s2);
  EXPECT_EQ(symbol_shape2.MutableDims()[3], s3);
}

TEST_F(SymbolShapeUT, ModifyDimOk) {
  SymbolShape symbol_shape1;
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape2{s0, s1, s2, s3};

  symbol_shape1.MutableDims().emplace_back(s0);
  symbol_shape1.MutableDims().emplace_back(s0);
  EXPECT_EQ(symbol_shape1.GetDims()[0], s0);
  EXPECT_EQ(symbol_shape1.GetDims()[1], s0);

  symbol_shape2.MutableDims()[0] = s1;
  symbol_shape2.MutableDims()[1] = s2;
  EXPECT_EQ(symbol_shape2.GetDims()[0], s1);
  EXPECT_EQ(symbol_shape2.GetDims()[1], s2);
  EXPECT_EQ(symbol_shape2.GetDims()[2], s2);
  EXPECT_EQ(symbol_shape2.GetDims()[3], s3);
}

TEST_F(SymbolShapeUT, SetGetDimOk) {
  auto one = ge::Symbol(1);
  SymbolShape s{one};
  EXPECT_EQ(s.GetDims()[0], one);
  auto s10 = ge::Symbol(10);
  s.MutableDims()[0] = s10;
  EXPECT_EQ(s.GetDims()[0], ge::Symbol(10));
}

TEST_F(SymbolShapeUT, AppendDimOk) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol(10);
  SymbolShape symbol_shape1{s0};
  symbol_shape1.AppendDim(s1).AppendDim(s2);
  SymbolShape expect_s{s0, s1, s2};
  EXPECT_EQ(symbol_shape1, expect_s);
}

TEST_F(SymbolShapeUT, AppendDimOutOfBounds) {
  auto s0 = ge::Symbol("s0");
  SymbolShape s{s0, s0, s0, s0, s0, s0, s0, s0, s0,
                s0, s0, s0, s0, s0, s0, s0, s0, s0,
                s0, s0, s0, s0, s0, s0, s0};
  SymbolShape expect_s{s0, s0, s0, s0, s0, s0, s0, s0, s0,
                       s0, s0, s0, s0, s0, s0, s0, s0, s0,
                       s0, s0, s0, s0, s0, s0, s0, s0};
  s.AppendDim(s0);
  EXPECT_EQ(s, expect_s);
}

TEST_F(SymbolShapeUT, SetScalar) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  SymbolShape symbol_shape2{s0, s1, s2, s3};
  EXPECT_EQ(symbol_shape2.GetDimNum(), 4);
  symbol_shape2.SetScalar();
  EXPECT_EQ(symbol_shape2.GetDimNum(), 0);
}

TEST_F(SymbolShapeUT, CopyConstruct) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s2");
  const SymbolShape symbol_shape1{s0, s1, s2, s3};
  const SymbolShape symbol_shape_copy(symbol_shape1);
  EXPECT_EQ(symbol_shape_copy.GetDimNum(), 4);
}

TEST_F(SymbolShapeUT, CopyAssign) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  auto s4 = ge::Symbol(128);
  SymbolShape symbol_shape1{s0, s1, s2, s3};
  SymbolShape symbol_shape_copy(symbol_shape1);

  EXPECT_EQ(symbol_shape_copy.GetDimNum(), 4);
  symbol_shape_copy = symbol_shape1;
  EXPECT_EQ(symbol_shape_copy.GetDimNum(), 4);

  SymbolShape a{s4, s3, s2, s1};
  SymbolShape a_copy{s0, s1, s2, s3, s4};
  EXPECT_EQ(a.GetDimNum(), 4);
  a = a_copy;
  EXPECT_EQ(a.GetDimNum(), 5);
  EXPECT_EQ(a.GetDims()[4], s4);

  std::vector<ge::Symbol> res{s0, s1, s2, s3, s4};
  EXPECT_EQ(a.GetDims().size(), 5);
  EXPECT_EQ(a.GetDims().at(0), s0);
  EXPECT_EQ(a.GetDims().at(2), s2);
  EXPECT_EQ(a.GetDims().at(4), s4);
  a.MutableDims()[0] = s1;
  EXPECT_EQ(a.GetDims().at(0), s1);
  a.MutableDims()[2] = s1;
  EXPECT_EQ(a.GetDims().at(2), s1);
}

// SymbolShape的基础测试
TEST_F(SymbolShapeUT, SymbolShapeTest) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  
  SymbolShape symbol_shape({s0, s1, s2, s3});
  EXPECT_EQ(symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(symbol_shape.GetDim(0), s0);
  EXPECT_EQ(symbol_shape.GetDim(1), s1);
  symbol_shape.MutableDim(0) = s2;
  EXPECT_EQ(symbol_shape.GetDim(0), s2);
  symbol_shape.AppendDim(s3);
  EXPECT_EQ(symbol_shape.GetDimNum(), 5);
  EXPECT_EQ(symbol_shape.GetSymbolShapeSize(), s1 * s2 * s2 * s3 * s3);
  EXPECT_EQ(symbol_shape.IsScalar(), false);
  symbol_shape.Clear();
  EXPECT_EQ(symbol_shape.GetDimNum(), 0);

  // scalar测试
  SymbolShape symbol_shape_scalar;
  EXPECT_EQ(symbol_shape_scalar.IsScalar(), true);
  EXPECT_EQ(symbol_shape_scalar.GetDimNum(), 0);
  EXPECT_EQ(symbol_shape_scalar.GetSymbolShapeSize(), 1);

  // Mutabe测试
  SymbolShape symbol_shape2({s0, s1, s2, s3});
  symbol_shape2.MutableDim(0) = s1;
  EXPECT_EQ(symbol_shape2.GetDim(0), s1);
  SymbolShape symbol_shape2_res({s1, s1, s2, s3});
  EXPECT_EQ(symbol_shape2, symbol_shape2_res);

  symbol_shape2.MutableDims() = {s2, s2, s2, s2};
  SymbolShape symbol_shape2_res2({s2, s2, s2, s2});
  EXPECT_EQ(symbol_shape2, symbol_shape2_res2);
  EXPECT_NE(symbol_shape2, symbol_shape2_res);

  // 等于不等于测试
  SymbolShape symbol_shape3({s0, s1, s2});
  SymbolShape symbol_shape4({s0, s1, s2, s3});
  EXPECT_EQ(symbol_shape3 == symbol_shape3, true);
  EXPECT_EQ(symbol_shape3 == symbol_shape4, false);
  EXPECT_EQ(symbol_shape3 != symbol_shape3, false);
  EXPECT_EQ(symbol_shape3 != symbol_shape4, true);

  symbol_shape3.AppendDim(s3);
  EXPECT_EQ(symbol_shape3 == symbol_shape4, true);
  EXPECT_EQ(symbol_shape3 != symbol_shape4, false);
}

// SymbolShape的拷贝构造、移动构造、拷贝赋值、移动赋值测试
TEST_F(SymbolShapeUT, SymbolShapeCopyTest) {
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  SymbolShape symbol_shape({s0, s1, s2, s3});

  SymbolShape symbol_shape_copy(symbol_shape);
  EXPECT_EQ(symbol_shape_copy.GetDimNum(), 4);
  EXPECT_EQ(symbol_shape_copy.GetDim(0), s0);
  EXPECT_EQ(symbol_shape_copy.GetDim(1), s1);

  SymbolShape symbol_shape_move(std::move(symbol_shape_copy));
  EXPECT_EQ(symbol_shape_move.GetDimNum(), 4);
  EXPECT_EQ(symbol_shape_move.GetDim(0), s0);
  EXPECT_EQ(symbol_shape_move.GetDim(1), s1);
  // 移动赋值后，symbol_shape_move的内容没了
  EXPECT_EQ(symbol_shape_copy.GetDimNum(), 0);

  symbol_shape_copy = symbol_shape_move;
  EXPECT_EQ(symbol_shape_copy.GetDimNum(), 4);
  EXPECT_EQ(symbol_shape_copy.GetDim(0), s0);
  EXPECT_EQ(symbol_shape_copy.GetDim(1), s1);
  // 拷贝赋值后，symbol_shape_move的内容不变
  EXPECT_EQ(symbol_shape_move.GetDimNum(), 4);

  symbol_shape_move = std::move(symbol_shape_copy);
  EXPECT_EQ(symbol_shape_move.GetDimNum(), 4);
  EXPECT_EQ(symbol_shape_move.GetDim(0), s0);
  EXPECT_EQ(symbol_shape_move.GetDim(1), s1);
}

TEST_F(SymbolShapeUT, SymbolShapeSizeCacheTest) {
  SymbolShape symbol_shape;
  EXPECT_EQ(symbol_shape.GetSymbolShapeSize(), 1);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  SymbolShape symbol_shape1({s0, s1, s2, s3});
  auto size = symbol_shape1.GetSymbolShapeSize();
  EXPECT_EQ(size, s0 * s1 * s2 * s3);

  symbol_shape1.MutableDim(0) = s1;
  EXPECT_EQ(symbol_shape1.GetSymbolShapeSize(), s1 * s1 * s2 * s3);

  symbol_shape1.MutableDims() = {s2, s2, s2, s2};
  EXPECT_EQ(symbol_shape1.GetSymbolShapeSize(), s2 * s2 * s2 * s2);

  symbol_shape1.AppendDim(s3);
  EXPECT_EQ(symbol_shape1.GetSymbolShapeSize(), s2 * s2 * s2 * s2 * s3);
}
}  // namespace gert
