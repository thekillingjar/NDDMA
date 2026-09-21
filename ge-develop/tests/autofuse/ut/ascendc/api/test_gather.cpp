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
#include "gather.h"

template<class T>
void UbToGm(T* gm, LocalTensor<T>& local, uint64_t size) {
  for (int i = 0; i < size; i++) {
    gm[i] = local.GetValue(i);
  }
}

template<class T1, class T2>
void TestGatherExtendLastAxis(std::vector<uint32_t> &input1_shape,
  std::vector<uint32_t> &input2_shape, std::vector<uint32_t> &output_shape) {
  int input1_size = input1_shape[0];
  int input2_size = input2_shape[0] * input2_shape[1];
  int output_size = output_shape[0] * output_shape[1];

  T1 *x1 = (T1*)AscendC::GmAlloc(sizeof(T1) * input1_size);
  T2 *x2 = (T2*)AscendC::GmAlloc(sizeof(T2) * input2_size);
  T1 *y = (T1*)AscendC::GmAlloc(sizeof(T1) * output_size);
  T1 *expect = (T1*)AscendC::GmAlloc(sizeof(T1) * output_size);

  for (int i = 0; i < input1_shape[0]; ++i) {
    x1[i] = (T1)i;
  }

  for (int i = 0; i < input2_shape[0]; ++i) {
    for (int j = 0; j < input2_shape[1]; ++j) {
        x2[i*input2_shape[1] + j] = (T2)j;
    }
  }

  for (int i = 0; i < output_shape[0]; ++i) {
    for (int j = 0; j < output_shape[1]; ++j) {
        expect[i*output_shape[1] + j] = x1[x2[i*output_shape[1] + j]];
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t param_last_axis_size, uint32_t vectorized_axis_size,
                   T1 *x1, T2 *x2, T1 *y) {
    TPipe pipe;
    TBuf<TPosition::VECIN> outTensor, tmpTensor;
    uint64_t tmpSize = vectorized_axis_size * sizeof(T2);
    pipe.InitBuffer(tmpTensor, tmpSize);
    pipe.InitBuffer(outTensor, tmpSize);
    LocalTensor<uint8_t> tmp_buf = tmpTensor.Get<uint8_t>();
    LocalTensor<T1> output = outTensor.Get<T1>();

    AscendC::GlobalTensor<T2> indices;
    indices.SetGlobalBuffer((__gm__ T2 *)x2);
    AscendC::GlobalTensor<T1> params;
    params.SetGlobalBuffer((__gm__ T1 *)x1);
    AscendC::GlobalTensor<T1> result;
    result.SetGlobalBuffer((__gm__ T1 *)y);
    GatherExtend<T1, T2>(output, params, indices, param_last_axis_size, vectorized_axis_size, tmp_buf);
    UbToGm(y, output, vectorized_axis_size);
  };

  //调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(kernel, 1, input1_shape[0], input2_size / 2, x1, x2, y);

  //验证结果
  int diff_count = 0;
  for (int i = 0; i < input2_size / 2; ++i) {
    auto diff = (double)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
    }
  }
  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiGather, Test_gather_last_axis) {
  std::vector<std::vector<uint32_t>> input1_shape = {{4000}};
  std::vector<std::vector<uint32_t>> input2_shape = {{100,100}};
  std::vector<std::vector<uint32_t>> output_shape = {{100,100}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestGatherExtendLastAxis<float, int32_t>(input1_shape[idx], input2_shape[idx], output_shape[idx]);
    TestGatherExtendLastAxis<float, int64_t>(input1_shape[idx], input2_shape[idx], output_shape[idx]);
  }
}