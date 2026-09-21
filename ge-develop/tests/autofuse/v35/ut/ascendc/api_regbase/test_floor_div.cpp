/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "floor_div.h"

using namespace AscendC;

template <typename T>
struct TensorFloorDivInputParam {
  T *y{};
  T *exp{};
  T *src0{};
  T *src1{};
  uint32_t size{0};
  uint32_t out_size{0};
};

class TestRegbaseApiFloorDiv :public testing::Test {
 protected:
  template <typename T>
  static void InvokeKernelWithTwoTensorInput(TensorFloorDivInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;

    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, 65312);

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();


    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_x2, param.src1, param.size);
    FloorDivExtend(l_y, l_x1, l_x2, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }


  template <typename T>
  static void CreateTensorInput(TensorFloorDivInputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    int input_range = 10;

    std::mt19937 eng(1);
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    // 构造src1的随机生成器
    std::mt19937 eng1(3);                                  // Seed the generator
    std::uniform_int_distribution distr1(1, input_range);  // Define the range

    for (int i = 0; i < param.size; i++) {
      T input = distr(eng);  // Use the secure random number generator
      T input1 = distr1(eng1);
      param.src0[i] = input;
      param.src1[i] = input1;
      param.exp[i] = std::floor((double)input / (double)input1);
    }
  }

  template <typename T>
  static uint32_t Valid(T *y, T *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (y[i] != exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void FloorDivTest(uint32_t size) {
    TensorFloorDivInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelWithTwoTensorInput(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestRegbaseApiFloorDiv, FloorDiv_Test) {
  FloorDivTest<half>(ONE_BLK_SIZE / sizeof(half));
  FloorDivTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  FloorDivTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  FloorDivTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  FloorDivTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  FloorDivTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  FloorDivTest<float>(ONE_BLK_SIZE / sizeof(float));
  FloorDivTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  FloorDivTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  FloorDivTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  FloorDivTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  FloorDivTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) / 2 /sizeof(float));
}