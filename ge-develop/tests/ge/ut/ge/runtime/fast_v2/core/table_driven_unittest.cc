/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/table_driven.h"
#include <gtest/gtest.h>
#include "graph/types.h"
namespace gert {
namespace {
class Foo {
 public:
  Foo() : a_(0), b_(0) {}
  Foo(int64_t a, int64_t b) : a_(a), b_(b) {}
  Foo(Foo &&_) = default;
  Foo(const Foo &) = default;
  Foo &operator=(Foo &&) = default;
  Foo &operator=(const Foo &) = default;

  [[nodiscard]] int64_t GetA() const {
    return a_;
  }
  void SetA(int64_t a) {
    a_ = a;
  }
  [[nodiscard]] int64_t GetB() const {
    return b_;
  }
  void SetB(int64_t b) {
    b_ = b;
  }

 private:
  int64_t a_;
  int64_t b_;
};
}  // namespace
class TableDrivenUT : public testing::Test {};
TEST_F(TableDrivenUT, NotFound_UseDefaultValue) {
  auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, int32_t>(-1)
                   .Add(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, 1)
                   .Add(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, 10)
                   .Add(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 11)
                   .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN, 100);
  EXPECT_EQ(table.Find(ge::FORMAT_ND, ge::FORMAT_ND), -1);
  EXPECT_EQ(table.Find(ge::FORMAT_ND, ge::FORMAT_NC1HWC0), -1);

  ASSERT_NE(table.FindPointer(ge::FORMAT_ND, ge::FORMAT_ND), nullptr);
  ASSERT_NE(table.FindPointer(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ), nullptr);
  EXPECT_EQ(*table.FindPointer(ge::FORMAT_ND, ge::FORMAT_ND), -1);
  EXPECT_EQ(*table.FindPointer(ge::FORMAT_ND, ge::FORMAT_NC1HWC0), -1);
}
TEST_F(TableDrivenUT, Found) {
  auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, int32_t>(-1)
                   .Add(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, 1)
                   .Add(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, 10)
                   .Add(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 11)
                   .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN, 100);

  EXPECT_EQ(table.Find(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ), 1);
  EXPECT_EQ(table.Find(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0), 10);

  ASSERT_NE(table.FindPointer(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ), nullptr);
  ASSERT_NE(table.FindPointer(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0), nullptr);
  EXPECT_EQ(*table.FindPointer(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ), 1);
  EXPECT_EQ(*table.FindPointer(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0), 10);
}
TEST_F(TableDrivenUT, OutOfRange) {
  auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, int32_t>(-1)
                   .Add(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, 1)
                   .Add(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, 10)
                   .Add(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 11)
                   .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN, 100);
  EXPECT_EQ(table.Find(ge::FORMAT_END, ge::FORMAT_FRACTAL_NZ), -1);
  EXPECT_EQ(table.Find(ge::FORMAT_NCHW, ge::FORMAT_END), -1);

  ASSERT_EQ(table.FindPointer(ge::FORMAT_END, ge::FORMAT_FRACTAL_NZ), nullptr);
  ASSERT_EQ(table.FindPointer(ge::FORMAT_NHWC, ge::FORMAT_END), nullptr);
}
TEST_F(TableDrivenUT, Class) {
  auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, Foo>(Foo{-1, -1})
                   .Add(ge::FORMAT_NCHW, ge::FORMAT_NHWC, Foo(10, 20))
                   .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NHWC, Foo(100, 200));

  // not register, default value
  ASSERT_NE(table.FindPointer(ge::FORMAT_ND, ge::FORMAT_ND), nullptr);
  ASSERT_EQ(table.FindPointer(ge::FORMAT_ND, ge::FORMAT_ND)->GetA(), -1);
  ASSERT_EQ(table.FindPointer(ge::FORMAT_ND, ge::FORMAT_ND)->GetA(), -1);

  ASSERT_NE(table.FindPointer(ge::FORMAT_NCHW, ge::FORMAT_NHWC), nullptr);
  ASSERT_EQ(table.FindPointer(ge::FORMAT_NCHW, ge::FORMAT_NHWC)->GetA(), 10);
  ASSERT_EQ(table.FindPointer(ge::FORMAT_NCHW, ge::FORMAT_NHWC)->GetB(), 20);
}
TEST_F(TableDrivenUT, ModifyOk) {
  auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, Foo>(Foo{-1, -1})
                   .Add(ge::FORMAT_NCHW, ge::FORMAT_NHWC, Foo(10, 20))
                   .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NHWC, Foo(100, 200));

  // not register, default value
  ASSERT_NE(table.FindPointer(ge::FORMAT_NCHW, ge::FORMAT_NHWC), nullptr);
  ASSERT_EQ(table.FindPointer(ge::FORMAT_NCHW, ge::FORMAT_NHWC)->GetA(), 10);
  ASSERT_EQ(table.FindPointer(ge::FORMAT_NCHW, ge::FORMAT_NHWC)->GetB(), 20);

  table.FindPointer(ge::FORMAT_NCHW, ge::FORMAT_NHWC)->SetA(100);
  ASSERT_EQ(table.FindPointer(ge::FORMAT_NCHW, ge::FORMAT_NHWC)->GetA(), 100);
  ASSERT_EQ(table.FindPointer(ge::FORMAT_NCHW, ge::FORMAT_NHWC)->GetB(), 20);
}
}  // namespace gert