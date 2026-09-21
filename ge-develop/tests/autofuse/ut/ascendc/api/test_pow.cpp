/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * test_compare.cpp
 */

#include <cmath>
#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "utils.h"
// 保持在utils.h之后
// 保持在duplicate.h之后
#include "pow.h"

using namespace AscendC;

namespace ge {
template <typename T>
struct PowCase1InputParam {
  T *y{};
  T *exp{};
  T *src0{};
  T *src1{};
  uint32_t size{0};
  BinaryRepeatParams a;
};
template <typename T>
struct PowCase2InputParam {
  T *y{};
  T *exp{};
  T src0;
  T *src1{};
  uint32_t size{0};
  BinaryRepeatParams a;
};
template <typename T>
struct PowCase3InputParam {
  T *y{};
  T *exp{};
  T *src0{};
  T src1;
  uint32_t size{0};
  BinaryRepeatParams a;
};

class TestApiPowUT : public testing::Test {
 protected:
  // =======================================Case3========================================
  template <typename T>
  static uint32_t ValidCase3(PowCase3InputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void CreateCase3Input(PowCase3InputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    int input_range = 3;
    std::mt19937 eng(1);                                         // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    auto src0 = distr(eng);  // Use the secure random number generator
    param.src0 = src0;
    for (int i = 0; i < param.size; i++) {
      auto src1 = distr(eng);  // Use the secure random number generator
      param.src1[i] = src1;
      param.exp[i] = pow(src0, src1);
    }
  }

  template <typename T>
  static void InvoPowCase3Kernel(PowCase3InputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> src2buf, ybuf, tmp;
    tpipe.InitBuffer(src2buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_src2 = src2buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_src2, param.src1, param.size);
    Pow(l_y, param.src0, l_src2, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void PowCase3Test(uint32_t size) {
    PowCase3InputParam<T> param{};
    param.size = size;

    CreateCase3Input(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvoPowCase3Kernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = ValidCase3(param);
    EXPECT_EQ(diff_count, 0);
  }
  // =======================================Case2========================================
  template <typename T>
  static uint32_t ValidCase2(PowCase2InputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void CreateCase2Input(PowCase2InputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    int input_range = 3;
    std::mt19937 eng(1);                                         // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    auto src1 = distr(eng);  // Use the secure random number generator
    param.src1 = src1;
    for (int i = 0; i < param.size; i++) {
      auto src0 = distr(eng);  // Use the secure random number generator
      param.src0[i] = src0;
      param.exp[i] = pow(src0, src1);
    }
  }

  template <typename T>
  static void InvoPowCase2Kernel(PowCase2InputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> src1buf, ybuf, tmp;
    tpipe.InitBuffer(src1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_src1 = src1buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_src1, param.src0, param.size);
    Pow(l_y, l_src1, param.src1, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void PowCase2Test(uint32_t size) {
    PowCase2InputParam<T> param{};
    param.size = size;

    CreateCase2Input(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvoPowCase2Kernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = ValidCase2(param);
    EXPECT_EQ(diff_count, 0);
  }
  // =======================================Case1========================================
  template <typename T>
  static uint32_t ValidCase1(PowCase1InputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void CreateCase1Input(PowCase1InputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    int input_range = 3;
    std::mt19937 eng(1);                                         // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    for (int i = 0; i < param.size; i++) {
      auto src1 = distr(eng);  // Use the secure random number generator
      auto src2 = distr(eng);  // Use the secure random number generator
      param.src0[i] = src1;
      param.src1[i] = src2;
      param.exp[i] = pow(src1, src2);
    }
  }

  template <typename T>
  static void InvoPowCase1Kernel(PowCase1InputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> src1buf, src2buf, ybuf, tmp;
    tpipe.InitBuffer(src1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(src2buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_src1 = src1buf.Get<T>();
    LocalTensor<T> l_src2 = src2buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_src1, param.src0, param.size);
    GmToUb(l_src2, param.src1, param.size);
    Pow(l_y, l_src1, l_src2, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void PowCase1Test(uint32_t size) {
    PowCase1InputParam<T> param{};
    param.size = size;

    CreateCase1Input(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvoPowCase1Kernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = ValidCase1(param);
    EXPECT_EQ(diff_count, 0);
  }

};

TEST_F(TestApiPowUT, PowCase1) {
  PowCase1Test<int32_t>(64);
  PowCase1Test<int32_t>(72);
  PowCase1Test<int32_t>(256);
  PowCase1Test<int32_t>(264);
  PowCase1Test<int32_t>(2056);
}
TEST_F(TestApiPowUT, PowCase2) {
  PowCase1Test<int32_t>(64);
  PowCase1Test<int32_t>(72);
  PowCase1Test<int32_t>(256);
  PowCase1Test<int32_t>(264);
  PowCase1Test<int32_t>(2056);
}
TEST_F(TestApiPowUT, PowCase3) {
  PowCase1Test<int32_t>(64);
  PowCase1Test<int32_t>(72);
  PowCase1Test<int32_t>(256);
  PowCase1Test<int32_t>(264);
  PowCase1Test<int32_t>(2056);
}

}  // namespace ge
