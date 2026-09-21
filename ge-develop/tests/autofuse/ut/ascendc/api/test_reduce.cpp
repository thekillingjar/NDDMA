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

#include "tikicpulib.h"
#include "kernel_operator.h"
using namespace AscendC;

#include "utils.h"
#include "test_api_utils.h"
#include "reduce.h"

class TestApiReduceLast : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(TestApiReduceLast, Test_reduce_min_ab_to_a1) {
  // 构造测试输入和预期结果
  int a = this->GetParam()[0];
  int b = this->GetParam()[1];
  int stride = this->GetParam()[2];
  auto *x = (half*)AscendC::GmAlloc(sizeof(half) * a * stride);
  auto *y = (half*)AscendC::GmAlloc(sizeof(half) * a * 1);
  half expect[a];

  for (int i = 0; i < a; i++) {
    expect[i] = (double)(i * b);
    for (int j = 0; j < stride; j++) {
      x[i * stride + j] = (double)(i * b + j);
    }
  }

  // 构造Api调用函数
  auto kernel = [](int32_t a, int32_t b, int32_t stride, half *x, half *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * a * stride);
    tpipe.InitBuffer(ybuf, sizeof(half) * a * 1);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * stride);
    GmToUb(l_y, y, a);

    ReduceLast<half, WholeReduceMinAdapt, Min>(l_y, l_x, a, b, stride, l_tmp);

    UbToGm(y, l_y, a);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, stride, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
      auto diff = (double)(y[i] - expect[i]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST_P(TestApiReduceLast, Test_reduce_mean_ab_to_a1) {
  // 构造测试输入和预期结果
  int a = this->GetParam()[0];
  int b = this->GetParam()[1];
  int stride = this->GetParam()[2];
  auto *x = (float*)AscendC::GmAlloc(sizeof(float) * a * stride);
  auto *y = (float*)AscendC::GmAlloc(sizeof(float) * a * 1);
  float expect[a];

  for (int i = 0; i < a; i++) {
    expect[i] = 0;
    for (int j = 0; j < stride; j++) {
      x[i * stride + j] = (double)(i * b + j);
      if (j < b) {
        expect[i] += x[i * stride + j];
      }
    }
    expect[i] /= b;
  }

  // 构造Api调用函数
  auto kernel = [](int32_t a, int32_t b, int32_t stride, float *x, float *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(float) * a * stride);
    tpipe.InitBuffer(ybuf, sizeof(float) * a * 1);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<float>();
    auto l_y = ybuf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * stride);
    GmToUb(l_y, y, a);

    ReduceLast<float, WholeReduceMeanAdapt, Add>(l_y, l_x, a, b, stride, l_tmp);

    UbToGm(y, l_y, a);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, stride, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    auto diff = (double)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiReduceLast,
    ::testing::Values(std::vector<int>{32, 32, 32},
                      std::vector<int>{300, 32, 32},
                      std::vector<int>{32, 128, 128},
                      std::vector<int>{1, 50, 64}));

TEST(TestApiReduceSumInt32, test_ab_to_a) {
  // 构造测试输入和预期结果
  uint32_t a = 16, b = 32;
  auto *x = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * a);
  int32_t expect[a];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = (int32_t)1;
  }
  x[5] = (int32_t)2;
  x[6] = (int32_t)(-2);
  x[20] = (int32_t)2;
  x[52] = (int32_t)3;

  for (uint32_t i = 0; i < a; i++) {
    expect[i] = 0;
    for (uint32_t j = 0; j < b; j++) {
      uint32_t idx = i * b + j;
      expect[i] += x[idx];
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, int32_t *x, int32_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(int32_t) * a);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int32_t>();
    auto l_y = ybuf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);

    uint32_t shape[] = {a, b};

    ReduceSumInt32<int32_t, AscendC::Pattern::Reduce::AR, false>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, a);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < a; i++) {
    auto diff = (double)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiReduceSumInt32, test_ab_to_b) {
  // 构造测试输入和预期结果
  uint32_t a = 16, b = 64;
  auto *x = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * b);
  int32_t expect[b];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = (int32_t)1;
  }
  x[5] = (int32_t)2;
  x[6] = (int32_t)(-2);
  x[20] = (int32_t)2;
  x[52] = (int32_t)3;

  for (uint32_t i = 0; i < b; i++) {
    expect[i] = 0;
    for (uint32_t j = 0; j < a; j++) {
      uint32_t idx = j * b + i;
      expect[i] += x[idx];
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, int32_t *x, int32_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(int32_t) * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int32_t>();
    auto l_y = ybuf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);

    uint32_t shape[] = {a, b};

    ReduceSumInt32<int32_t, AscendC::Pattern::Reduce::RA, false>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < b; i++) {
    auto diff = (int32_t)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiReduceSumInt32, test_ab_to_a_unalign) {
  // 构造测试输入和预期结果
  uint32_t a = 8, b = 8;
  auto *x = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  auto *y = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * b);
  int32_t expect[a];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = i % b == 7 ? 0 : 2;
  }

  for (uint32_t i = 0; i < a; i++) {
    expect[i] = 14;
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, int32_t *x, int32_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(int32_t) * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int32_t>();
    auto l_y = ybuf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);

    uint32_t shape[] = {a, 7};

    ReduceSumInt32<int32_t, AscendC::Pattern::Reduce::AR, false>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < a; i++) {
    auto diff = (double)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}
