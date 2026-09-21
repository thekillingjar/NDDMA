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
 * test_clipbyvalue.cpp
 */

#include <cmath>
#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "utils.h"
// 保持在utils.h之后
#include "duplicate.h"
// 保持在duplicate.h之后
#include "clipbyvalue.h"

using namespace AscendC;

namespace ge {
template <typename T>
struct ClipByValueCase1InputParam {
  T *y{};
  T *exp{};
  T *src0{};
  T *src1{};
  T *src2{};
  uint32_t size{0};
  BinaryRepeatParams a;
};

template <typename T>
struct ClipByValueCase2InputParam {
  T *y{};
  T *exp{};
  T *src0{};
  T src1;
  T src2;
  uint32_t size{0};
  BinaryRepeatParams a;
};

template <typename T>
struct ClipByValueCase3InputParam {
  T *y{};
  T *exp{};
  T *src0{};
  T *src1{};
  T src2;
  uint32_t size{0};
  BinaryRepeatParams a;
};

template <typename T>
struct ClipByValueCase4InputParam {
  T *y{};
  T *exp{};
  T *src0{};
  T src1;
  T *src2{};
  uint32_t size{0};
  BinaryRepeatParams a;
};

template <typename T>
struct ClipByValueCase5InputParam {
  T *y{};
  T *exp{};
  T src0;
  T src1;
  T src2;
  uint32_t size{0};
  BinaryRepeatParams a;
};

class TestApiClipByValueUT : public testing::Test {
 protected:
  // =======================================Case5========================================

  template <typename T>
  static uint32_t ValidCase5(ClipByValueCase5InputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void CreateCase5Input(ClipByValueCase5InputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    int input_range = 10;
    std::mt19937 eng(1);                                         // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    auto src0 = distr(eng);  // Use the secure random number generator
    auto src1 = distr(eng);  // Use the secure random number generator
    auto src2 = distr(eng);  // Use the secure random number generator

    param.src0 = src0;
    param.src1 = src1;
    param.src2 = src2;

    if (src0 < src1) {
      for (int i = 0; i < param.size; i++) {
        param.exp[i] = param.src1;
      }
    } else if (src0 > src2){
      for (int i = 0; i < param.size; i++) {
        param.exp[i] = param.src2;
      }
    } else {
      for (int i = 0; i < param.size; i++) {
        param.exp[i] = param.src0;
      }
    }
  }

  template <typename T>
  static void InvoClipByValueCase5Kernel(ClipByValueCase5InputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf, tmp;
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    ClipByValue(l_y, param.src0, param.src1, param.src2);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void ClipByValueCase5Test(uint32_t size) {
    ClipByValueCase5InputParam<T> param{};
    param.size = size;

    CreateCase5Input(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvoClipByValueCase5Kernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = ValidCase5(param);
    EXPECT_EQ(diff_count, 0);
  }
  // =======================================Case4========================================
  template <typename T>
  static uint32_t ValidCase4(ClipByValueCase4InputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void CreateCase4Input(ClipByValueCase4InputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src2 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    int input_range = 10;
    std::mt19937 eng(1);                                         // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    auto src1 = distr(eng);  // Use the secure random number generator
    do {
      src1 = distr(eng);  // Use the secure random number generator
    } while (src1 > 5);
    param.src1 = src1; // Min

    for (int i = 0; i < param.size; i++) {
      auto input = distr(eng);  // Use the secure random number generator
      param.src0[i] = input;

      auto src2 = distr(eng);  // Use the secure random number generator
      do {
        src2 = distr(eng);  // Use the secure random number generator
      } while (src2 <= src1);
      param.src2[i] = src2; // Max
      
      if ((T)input < param.src1) {
        param.exp[i] = param.src1;
      } else if ((T)input > param.src2[i]){
        param.exp[i] = param.src2[i];
      } else {
        param.exp[i] = input;
      }
    }
  }

  template <typename T>
  static void InvoClipByValueCase4Kernel(ClipByValueCase4InputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf, maxbuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(maxbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_max = maxbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_max, param.src2, param.size);
    ClipByValue(l_y, l_x1, param.src1, l_max, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void ClipByValueCase4Test(uint32_t size) {
    ClipByValueCase4InputParam<T> param{};
    param.size = size;

    CreateCase4Input(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvoClipByValueCase4Kernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = ValidCase4(param);
    EXPECT_EQ(diff_count, 0);
  }
  // =======================================Case3========================================
  template <typename T>
  static uint32_t ValidCase3(ClipByValueCase3InputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void CreateCase3Input(ClipByValueCase3InputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    int input_range = 10;
    std::mt19937 eng(1);                                         // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    auto src2 = distr(eng);  // Use the secure random number generator
    do {
      src2 = distr(eng);  // Use the secure random number generator
    } while (src2 < 5);
    param.src2 = src2; // Max

    for (int i = 0; i < param.size; i++) {
      auto input = distr(eng);  // Use the secure random number generator
      param.src0[i] = input;

      auto src1 = distr(eng);  // Use the secure random number generator
      do {
        src1 = distr(eng);  // Use the secure random number generator
      } while (src1 >= src2);
      param.src1[i] = src1; // Min
      
      if ((T)input < param.src1[i]) {
        param.exp[i] = param.src1[i];
      } else if ((T)input > param.src2){
        param.exp[i] = param.src2;
      } else {
        param.exp[i] = input;
      }
    }
  }

  template <typename T>
  static void InvoClipByValueCase3Kernel(ClipByValueCase3InputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf, minbuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(minbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_min = minbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_min, param.src1, param.size);
    ClipByValue(l_y, l_x1, l_min, param.src2, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void ClipByValueCase3Test(uint32_t size) {
    ClipByValueCase3InputParam<T> param{};
    param.size = size;

    CreateCase3Input(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvoClipByValueCase3Kernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = ValidCase3(param);
    EXPECT_EQ(diff_count, 0);
  }

  // =======================================Case2========================================
  template <typename T>
  static uint32_t ValidCase2(ClipByValueCase2InputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }
  template <typename T>
  static void CreateCase2Input(ClipByValueCase2InputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    int input_range = 10;
    std::mt19937 eng(1);                                         // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    auto src1 = distr(eng);  // Use the secure random number generator
    auto src2 = distr(eng);  // Use the secure random number generator
    if (src1 < src2) {
      param.src1 = src1; // Min
      param.src2 = src2; // Max
    } else {
      param.src1 = src2; // Min
      param.src2 = src1; // Max
    }

    for (int i = 0; i < param.size; i++) {
      auto input = distr(eng);  // Use the secure random number generator
      param.src0[i] = input;
      
      if ((T)input < param.src1) {
        param.exp[i] = param.src1;
      } else if ((T)input > param.src2){
        param.exp[i] = param.src2;
      } else {
        param.exp[i] = input;
      }
    }
  }

  template <typename T>
  static void InvoClipByValueCase2Kernel(ClipByValueCase2InputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, 8192);

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    ClipByValue(l_y, l_x1, param.src1, param.src2, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void ClipByValueCase2Test(uint32_t size) {
    ClipByValueCase2InputParam<T> param{};
    param.size = size;

    CreateCase2Input(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvoClipByValueCase2Kernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = ValidCase2(param);
    EXPECT_EQ(diff_count, 0);
  }

  // =======================================Case1========================================
  template <typename T>
  static uint32_t ValidCase1(ClipByValueCase1InputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (param.y[i] != param.exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }
  template <typename T>
  static void CreateCase1Input(ClipByValueCase1InputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src2 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    int input_range = 10;
    std::mt19937 eng(1);                                         // Seed the generator
    std::uniform_int_distribution distr(0, input_range);  // Define the range

    for (int i = 0; i < param.size; i++) {
      auto input = distr(eng);  // Use the secure random number generator
      param.src0[i] = input;
      auto src1 = distr(eng);  // Use the secure random number generator
      auto src2 = distr(eng);  // Use the secure random number generator
      if (src1 < src2) {
        param.src1[i] = src1; // Min
        param.src2[i] = src2; // Max
      } else {
        param.src1[i] = src2; // Min
        param.src2[i] = src1; // Max
      }
      
      if ((T)input < param.src1[i]) {
        param.exp[i] = param.src1[i];
      } else if ((T)input > param.src2[i]){
        param.exp[i] = param.src2[i];
      } else {
        param.exp[i] = input;
      }
    }
  }

  template <typename T>
  static void InvoClipByValueCase1Kernel(ClipByValueCase1InputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, minbuf, maxbuf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(minbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(maxbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_min = minbuf.Get<T>();
    LocalTensor<T> l_max = maxbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_min, param.src1, param.size);
    GmToUb(l_max, param.src2, param.size);
    ClipByValue(l_y, l_x1, l_min, l_max, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void ClipByValueCase1Test(uint32_t size) {
    ClipByValueCase1InputParam<T> param{};
    param.size = size;

    CreateCase1Input(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvoClipByValueCase1Kernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = ValidCase1(param);
    EXPECT_EQ(diff_count, 0);
  }

};

TEST_F(TestApiClipByValueUT, ClipByValue_case1) {
  ClipByValueCase1Test<float>(64);
  ClipByValueCase1Test<float>(72);
  ClipByValueCase1Test<float>(256);
  ClipByValueCase1Test<float>(264);
  ClipByValueCase1Test<float>(2056);
  //输入非32B对齐
  ClipByValueCase1Test<float>(2060);
  ClipByValueCase1Test<half>(128);
  ClipByValueCase1Test<half>(144);
  ClipByValueCase1Test<half>(512);
  ClipByValueCase1Test<half>(528);
  ClipByValueCase1Test<half>(4112);
  //输入非32B对齐
  ClipByValueCase1Test<half>(1928);
}

TEST_F(TestApiClipByValueUT, ClipByValue_case2) {
  ClipByValueCase2Test<float>(64);
  ClipByValueCase2Test<float>(72);
  ClipByValueCase2Test<float>(256);
  ClipByValueCase2Test<float>(264);
  ClipByValueCase2Test<float>(2056);
  //输入非32B对齐
  ClipByValueCase2Test<float>(2060);
  ClipByValueCase2Test<float>(8416); // 4496 + 3920
  ClipByValueCase2Test<half>(128);
  ClipByValueCase2Test<half>(144);
  ClipByValueCase2Test<half>(512);
  ClipByValueCase2Test<half>(528);
  ClipByValueCase2Test<half>(4112);
  //输入非32B对齐
  ClipByValueCase2Test<half>(1928);

  // size大于tumpbuf size/2的场景
  ClipByValueCase2Test<half>(7052);
}

TEST_F(TestApiClipByValueUT, ClipByValue_case3) {
  ClipByValueCase3Test<float>(64);
  ClipByValueCase3Test<float>(72);
  ClipByValueCase3Test<float>(256);
  ClipByValueCase3Test<float>(264);
  ClipByValueCase3Test<float>(2056);
  //输入非32B对齐
  ClipByValueCase3Test<float>(2060);
  ClipByValueCase3Test<half>(128);
  ClipByValueCase3Test<half>(144);
  ClipByValueCase3Test<half>(512);
  ClipByValueCase3Test<half>(528);
  ClipByValueCase3Test<half>(4112);
  //输入非32B对齐
  ClipByValueCase3Test<half>(1928);
}

TEST_F(TestApiClipByValueUT, ClipByValue_case4) {
  ClipByValueCase4Test<float>(64);
  ClipByValueCase4Test<float>(72);
  ClipByValueCase4Test<float>(256);
  ClipByValueCase4Test<float>(264);
  ClipByValueCase4Test<float>(2056);
  //输入非32B对齐
  ClipByValueCase4Test<float>(2060);
  ClipByValueCase4Test<half>(128);
  ClipByValueCase4Test<half>(144);
  ClipByValueCase4Test<half>(512);
  ClipByValueCase4Test<half>(528);
  ClipByValueCase4Test<half>(4112);
  //输入非32B对齐
  ClipByValueCase4Test<half>(1928);
}

TEST_F(TestApiClipByValueUT, ClipByValue_case5) {
  ClipByValueCase5Test<half>(256);
  ClipByValueCase5Test<float>(256);
}

// TEST_F(TestApiClipByValueUT, ClipByValue_case3) {
//   ClipByValueCase3Test<half>(256);
// }
// TEST_F(TestApiClipByValueUT, ClipByValue_case4) {
//   ClipByValueCase4Test<half>(256);
// }
}  // namespace ge
