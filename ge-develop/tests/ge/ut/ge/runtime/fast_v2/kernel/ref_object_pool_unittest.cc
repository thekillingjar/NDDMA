/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/ref_object_pool.h"
#include <gtest/gtest.h>
namespace gert {
namespace {
struct Tc { // test class
  explicit Tc(int64_t a) : a(a), v(a) {}
  int64_t a;
  std::vector<int64_t> v;
};
}
class RefObjectPoolUT : public testing::Test {};
TEST_F(RefObjectPoolUT, AcquireAndRelease_Ok) {
  RefObjectPool<Tc> pool;
  auto o1 = pool.Acquire(10);
  ASSERT_NE(o1, nullptr);
  pool.Release(o1);
  // no error when deconstruct pool
}
TEST_F(RefObjectPoolUT, AcquireReuse_Ok) {
  RefObjectPool<Tc, 1> pool;
  auto o1 = pool.Acquire(10);
  ASSERT_NE(o1, nullptr);
  pool.Release(o1);
  auto o2 = pool.Acquire(20);
  ASSERT_EQ(o1, o2);
  pool.Release(o2);
}
TEST_F(RefObjectPoolUT, AutoExtend_Ok) {
  RefObjectPool<Tc, 2> pool;
  std::vector<Tc *> os;
  for (int32_t i = 0; i < 10; ++i) {
    auto o = pool.Acquire(i);
    ASSERT_NE(o, nullptr);
    os.push_back(o);
  }
  for (auto o : os) {
    pool.Release(o);
  }
}
TEST_F(RefObjectPoolUT, AcquireAndRelease_Ok_WhenAutoExtend) {
  RefObjectPool<Tc, 2> pool;
  std::set<Tc *> os1, os2;
  for (int32_t i = 0; i < 10; ++i) {
    auto o = pool.Acquire(i);
    ASSERT_NE(o, nullptr);
    os1.insert(o);
  }
  for (auto o : os1) {
    pool.Release(o);
  }

  for (int32_t i = 0; i < 10; ++i) {
    auto o = pool.Acquire(i);
    ASSERT_NE(o, nullptr);
    os2.insert(o);
  }
  for (auto o : os2) {
    pool.Release(o);
  }

  ASSERT_EQ(os1.size(), 10U);
  ASSERT_EQ(os1, os2);
}
TEST_F(RefObjectPoolUT, AutoRelease_Ok_WhenDesctruct) {
  RefObjectPool<Tc, 2> pool;
  std::set<Tc *> os1;
  for (int32_t i = 0; i < 10; ++i) {
    auto o = pool.Acquire(i);
    ASSERT_NE(o, nullptr);
    os1.insert(o);
  }
  pool.Release(*(os1.begin()));
  // no release, no leak
}
}