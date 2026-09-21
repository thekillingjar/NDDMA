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
#include "duplicate.h"
#include "reduce_prod.h"

TEST(TestApiReduceProd, test_ab_to_a) {
  // 构造测试输入和预期结果
  uint32_t a = 16, b = 32;
  auto *x = (float*)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (float*)AscendC::GmAlloc(sizeof(float) * a);
  float expect[a];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = (double)1;
  }
  x[5] = (double)2;
  x[6] = (double)(-2);
  x[20] = (double)2;
  x[52] = (double)3;

  for (uint32_t i = 0; i < a; i++) {
    expect[i] = 1;
    for (uint32_t j = 0; j < b; j++) {
      uint32_t idx = i * b + j;
      expect[i] *= x[idx];
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, float *x, float *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(float) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(float) * a);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<float>();
    auto l_y = ybuf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);

    uint32_t shape[] = {a, b};

    ReduceProdExtend<float, true>(l_y, l_x, l_tmp, shape, true);

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


TEST(TestApiReduceProd, test_ab_to_b) {
  // 构造测试输入和预期结果
  uint32_t a = 16, b = 64;
  auto *x = (float*)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (float*)AscendC::GmAlloc(sizeof(float) * b);
  float expect[b];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = (double)1;
  }
  x[5] = (double)2;
  x[6] = (double)(-2);
  x[20] = (double)2;
  x[52] = (double)3;

  for (uint32_t i = 0; i < b; i++) {
    expect[i] = 1;
    for (uint32_t j = 0; j < a; j++) {
      uint32_t idx = j * b + i;
      expect[i] *= x[idx];
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, float *x, float *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(float) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(float) * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<float>();
    auto l_y = ybuf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);

    uint32_t shape[] = {a, b};

    ReduceProdExtend<float, false>(l_y, l_x, l_tmp, shape, true);

    UbToGm(y, l_y, b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  uint32_t diff_count = 0;
  for (uint32_t i = 0; i < b; i++) {
    auto diff = (double)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiReduceProd, test_ab_to_2) {
  // 构造测试输入和预期结果
  uint32_t a = 1, b = 32;
  auto *x = (float*)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (float*)AscendC::GmAlloc(sizeof(float) * b);
  float expect[a];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = (double)0;
  }
  x[0] = (double)5;
  x[1] = (double)(-2);

  expect[0] = -10;

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, float *x, float *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(float) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(float) * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<float>();
    auto l_y = ybuf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);

    uint32_t shape[] = {1, 2};

    ReduceProdExtend<float, true>(l_y, l_x, l_tmp, shape, true);

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

TEST(TestApiReduceProd, test_ab_to_a_unalign) {
  // 构造测试输入和预期结果
  uint32_t a = 8, b = 8;
  auto *x = (float*)AscendC::GmAlloc(sizeof(float) * a * b);
  auto *y = (float*)AscendC::GmAlloc(sizeof(float) * b);
  float expect[a];

  for (uint32_t i = 0; i < a * b; i++) {
    x[i] = i % b == 7 ? 0 : 2;
  }

  for (uint32_t i = 0; i < a; i++) {
    expect[i] = 128;
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t a, uint32_t b, float *x, float *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(float) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(float) * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<float>();
    auto l_y = ybuf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);

    uint32_t shape[] = {a, 7};

    ReduceProdExtend<float, true>(l_y, l_x, l_tmp, shape, true);

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
