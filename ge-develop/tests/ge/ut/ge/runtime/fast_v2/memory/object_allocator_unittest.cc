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
#include "kernel/memory/util/object_allocator.h"

using namespace gert;

namespace {
struct Foo : LinkNode<Foo> {
  Foo(int a) : x(a) {}

  int value() const {
    return x;
  }

  void value(int v) {
    this->x = v;
  }

  ~Foo() {
    this->x = 0;
  }

 private:
  volatile int x;
};
}  // namespace

struct ObjectAllocatorUnitTest : public testing::Test {
 protected:
  static constexpr size_t FOO_MAX = 3;
  ObjectAllocator<Foo> fooAllocator{FOO_MAX};
};

TEST_F(ObjectAllocatorUnitTest, should_alloc_and_free_without_construct_and_destruct) {
  ASSERT_EQ(FOO_MAX, fooAllocator.GetAvailableSize());

  auto f = new (fooAllocator.AllocMem()) Foo(5);
  ASSERT_TRUE(f != nullptr);
  ASSERT_EQ(5, f->value());
  ASSERT_EQ(FOO_MAX - 1, fooAllocator.GetAvailableSize());

  fooAllocator.FreeMem(*f);
  ASSERT_EQ(5, f->value());
  ASSERT_EQ(FOO_MAX, fooAllocator.GetAvailableSize());
}

TEST_F(ObjectAllocatorUnitTest, should_new_construct_delete_destruct) {
  ASSERT_EQ(FOO_MAX, fooAllocator.GetAvailableSize());

  auto f = fooAllocator.New(5);
  ASSERT_TRUE(f != nullptr);
  ASSERT_EQ(5, f->value());
  ASSERT_EQ(FOO_MAX - 1, fooAllocator.GetAvailableSize());

  fooAllocator.Free(*f);
  ASSERT_EQ(0, f->value());
  ASSERT_EQ(FOO_MAX, fooAllocator.GetAvailableSize());
}

TEST_F(ObjectAllocatorUnitTest, should_extend_auto) {
  Foo *foos[FOO_MAX + 1] = {nullptr};

  for (size_t i = 0; i < FOO_MAX; i++) {
    foos[i] = fooAllocator.New(i + 1);
    ASSERT_TRUE(foos[i] != nullptr);
  }

  ASSERT_EQ(0, fooAllocator.GetAvailableSize());

  foos[FOO_MAX] = fooAllocator.New(FOO_MAX + 1);
  ASSERT_TRUE(foos[FOO_MAX] != nullptr);
  ASSERT_EQ(FOO_MAX + 1, foos[FOO_MAX]->value());
  ASSERT_EQ(0, fooAllocator.GetAvailableSize());

  fooAllocator.Free(*foos[0]);
  ASSERT_EQ(1, fooAllocator.GetAvailableSize());

  foos[0] = fooAllocator.New(1);
  ASSERT_TRUE(foos[0] != nullptr);
  ASSERT_EQ(0, fooAllocator.GetAvailableSize());

  for (auto &f : foos) {
    fooAllocator.Free(*f);
  }
  ASSERT_EQ(4, fooAllocator.GetAvailableSize());
}
