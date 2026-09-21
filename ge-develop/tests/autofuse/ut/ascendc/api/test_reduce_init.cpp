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
#include "reduce_init.h"

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

class TestApiReduceInit : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiReduceInit, Test_ar_reduce_fold) {
  // 构造测试输入和预期结果
  int32_t dim_a = this->GetParam()[0];
  int32_t total_dim_r = this->GetParam()[1];
  int32_t tail_dim_r = this->GetParam()[2];
  auto *x0 = (float *)AscendC::GmAlloc(sizeof(float) * dim_a * total_dim_r);
  auto *x1 = (float *)AscendC::GmAlloc(sizeof(float) * dim_a * total_dim_r);
  auto *y = (float *)AscendC::GmAlloc(sizeof(float) * dim_a * total_dim_r);
  float expect[dim_a * total_dim_r];

  for (int i = 0; i < dim_a; i++) {
    for (int j = 0; j < total_dim_r; j++) {
      expect[i * total_dim_r + j] = (float)((i * total_dim_r + j) * 2);
      x0[i * total_dim_r + j] = (float)(i * total_dim_r + j);
      x1[i * total_dim_r + j] = (float)(i * total_dim_r + j);
    }
  }

  // 构造Api调用函数
  auto kernel = [](int32_t dim_a, int32_t total_dim_r, int32_t tail_dim_r, float *x0, float *x1, float *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x0buf, x1buf, ybuf;
    tpipe.InitBuffer(x0buf, sizeof(float) * dim_a * total_dim_r);
    tpipe.InitBuffer(x1buf, sizeof(float) * dim_a * total_dim_r);
    tpipe.InitBuffer(ybuf, sizeof(float) * dim_a * total_dim_r);

    auto x0_tensor = x0buf.Get<float>();
    auto x1_tensor = x1buf.Get<float>();
    auto y_tensor = ybuf.Get<float>();

    GmToUb(x0_tensor, x0, dim_a * total_dim_r);
    GmToUb(x1_tensor, x1, dim_a * total_dim_r);

    EntireTailFold<float, false, true, 2>(y_tensor, x0_tensor, x1_tensor, dim_a, total_dim_r, tail_dim_r);

    UbToGm(y, y_tensor, dim_a * total_dim_r);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, dim_a, total_dim_r, tail_dim_r, x0, x1, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < dim_a; i++) {
    for (int j = 0; j < total_dim_r; j++) {
      auto diff = (double)(y[i * total_dim_r + j] - expect[i * total_dim_r + j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST_P(TestApiReduceInit, Test_reduce_init_max) {
  // 构造测试输入和预期结果
  int32_t dim_a = this->GetParam()[0];
  int32_t dim_r = this->GetParam()[1];
  int32_t dim_r_current = this->GetParam()[2];
  int32_t inner_r = this->GetParam()[3];
  int32_t inner_r_align = (inner_r + 7) / 8 * 8;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * dim_a * dim_r);
  float expect[dim_a * dim_r];

  for (int i = 0; i < dim_a * (dim_r / inner_r_align); i++) {
    for (int j = 0; j < inner_r_align; j++) {
      int32_t index = i * inner_r_align + j;
      if (j >= inner_r) {
        expect[index] = -INFINITY;
        x[index] = 1.0f;
      } else {
        expect[index] = 1.0f;
        x[index] = 1.0f;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint64_t dim_a, uint64_t dim_r, uint64_t dim_r_current, uint64_t inner_r, float *x) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf;
    tpipe.InitBuffer(xbuf, sizeof(float) * dim_a * dim_r);

    auto x_tensor = xbuf.Get<float>();

    GmToUb(x_tensor, x, dim_a * dim_r);

    ReduceInit<float, 1, false>(x_tensor, dim_a, dim_r, dim_r_current, inner_r);

    UbToGm(x, x_tensor, dim_a * dim_r);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, dim_a, dim_r, dim_r_current, inner_r, x);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < dim_a; i++) {
    for (int j = 0; j < dim_r; j++) {
      auto diff = (double)(x[i * dim_r + j] - expect[i * dim_r + j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST_P(TestApiReduceInit, Test_reduce_init_min) {
  // 构造测试输入和预期结果
  int32_t dim_a = this->GetParam()[0];
  int32_t dim_r = this->GetParam()[1];
  int32_t dim_r_current = this->GetParam()[2];
  int32_t inner_r = this->GetParam()[3];
  int32_t inner_r_align = (inner_r + 7) / 8 * 8;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * dim_a * dim_r);
  float expect[dim_a * dim_r];

  for (int i = 0; i < dim_a * (dim_r / inner_r_align); i++) {
    for (int j = 0; j < inner_r_align; j++) {
      int32_t index = i * inner_r_align + j;
      if (j >= inner_r) {
        expect[index] = INFINITY;
        x[index] = 1.0f;
      } else {
        expect[index] = 1.0f;
        x[index] = 1.0f;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint64_t dim_a, uint64_t dim_r, uint64_t dim_r_current, uint64_t inner_r, float *x) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf;
    tpipe.InitBuffer(xbuf, sizeof(float) * dim_a * dim_r);

    auto x_tensor = xbuf.Get<float>();

    GmToUb(x_tensor, x, dim_a * dim_r);

    ReduceInit<float, 0, false>(x_tensor, dim_a, dim_r, dim_r_current, inner_r);

    UbToGm(x, x_tensor, dim_a * dim_r);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, dim_a, dim_r, dim_r_current, inner_r, x);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < dim_a; i++) {
    for (int j = 0; j < dim_r; j++) {
      auto diff = (double)(x[i * dim_r + j] - expect[i * dim_r + j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiReduceInit,
                         ::testing::Values(std::vector<int>{32, 32, 32, 7}, std::vector<int>{300, 32, 32, 7}));

class TestApiReduceInit_RepeatStrideOverLimit : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiReduceInit_RepeatStrideOverLimit, Test_reduce_init_sum) {
  // 构造测试输入和预期结果
  int32_t dim_a = this->GetParam()[0];
  int32_t dim_r = this->GetParam()[1];
  int32_t dim_r_current = this->GetParam()[2];
  int32_t inner_r = this->GetParam()[3];
  int32_t inner_r_align = (inner_r + 7) / 8 * 8;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * dim_a * dim_r);
  float expect[dim_a * dim_r];

  for (int i = 0; i < dim_a * (dim_r / inner_r_align); i++) {
    for (int j = 0; j < inner_r_align; j++) {
      int32_t index = i * inner_r_align + j;
      if (j >= inner_r) {
        expect[index] = 0;
        x[index] = 1.0f;
      } else {
        expect[index] = 1.0f;
        x[index] = 1.0f;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint64_t dim_a, uint64_t dim_r, uint64_t dim_r_current, uint64_t inner_r, float *x) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf;
    tpipe.InitBuffer(xbuf, sizeof(float) * dim_a * dim_r);

    auto x_tensor = xbuf.Get<float>();

    GmToUb(x_tensor, x, dim_a * dim_r);

    ReduceInit<float, 2, true>(x_tensor, dim_a, dim_r, dim_r_current, inner_r);

    UbToGm(x, x_tensor, dim_a * dim_r);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, dim_a, dim_r, dim_r_current, inner_r, x);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < dim_a; i++) {
    for (int j = 0; j < dim_r; j++) {
      auto diff = (double)(x[i * dim_r + j] - expect[i * dim_r + j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiReduceInit_RepeatStrideOverLimit,
                         ::testing::Values(std::vector<int>{4, 2136, 2136, 2132}));