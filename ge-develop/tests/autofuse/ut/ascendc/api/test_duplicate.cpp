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
 * test_duplicate.cpp
 */

#include <cmath>
#include "gtest/gtest.h"
#include "test_api_utils.h"
#include "tikicpulib.h"
#include "utils.h"
// 保持duplicate.h在utils.h之下
#include "duplicate.h"

using namespace AscendC;

namespace ge {
template <typename T>
struct DuplicateInputParam {
  T *y{};
  T *src{};
  uint32_t size{};
};

class TestApiDuplicateUT : public testing::Test {
 protected:
  template <typename T>
  static void CreateInput(DuplicateInputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = (T *)AscendC::GmAlloc(sizeof(T) * param.size);
    param.src = (T *)AscendC::GmAlloc(sizeof(T));
    if (std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value) {
      param.src[0] = 0xDEADBEEF12345678;
    } else {
      param.src[0] = 0xDE;
    }
  }

  template <typename T>
  static uint32_t Valid(DuplicateInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (!DefaultCompare(param.y[i], param.src[0])) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void InvokeKernel(DuplicateInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> ybuf, tmp;
    tpipe.InitBuffer(ybuf, sizeof(T) * param.size);
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    Duplicate(l_y, param.src[0], param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void DuplicateTest(uint32_t size) {
    DuplicateInputParam<T> param{};
    param.size = size;

    CreateInput(param);

    // 构造Api调用函数
    auto kernel = [&param]() { InvokeKernel<T>(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestApiDuplicateUT, Duplicate_int64_1BLK) {
  DuplicateTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_int64_1PRT) {
  DuplicateTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_int64_MAX_RPT) {
  DuplicateTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_int64_LT_BLK) {
  DuplicateTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_int64_LT_MAX_RPT) {
  DuplicateTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_int64_LT_RPT) {
  DuplicateTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_int64_LT_MAX) {
  DuplicateTest<int64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                          (ONE_BLK_SIZE - sizeof(int64_t))) /
                         sizeof(int64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_uint64_1BLK) {
  DuplicateTest<uint64_t>(ONE_BLK_SIZE / sizeof(uint64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_uint64_1PRT) {
  DuplicateTest<uint64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_uint64_MAX_RPT) {
  DuplicateTest<uint64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_uint64_LT_BLK) {
  DuplicateTest<uint64_t>((ONE_BLK_SIZE - sizeof(uint64_t)) / sizeof(uint64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_uint64_LT_MAX_RPT) {
  DuplicateTest<uint64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_uint64_LT_RPT) {
  DuplicateTest<uint64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_uint64_LT_MAX) {
  DuplicateTest<uint64_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                          (ONE_BLK_SIZE - sizeof(uint64_t))) /
                         sizeof(uint64_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_u8_LT_1BLK) {
  DuplicateTest<uint8_t>(1);
}

TEST_F(TestApiDuplicateUT, Duplicate_u8_LT_1BLK2) {
  DuplicateTest<uint8_t>(15);
}

TEST_F(TestApiDuplicateUT, Duplicate_u8_1BLK) {
  DuplicateTest<uint8_t>(ONE_BLK_SIZE / sizeof(uint8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_u8_1PRT) {
  DuplicateTest<uint8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_u8_MAX_RPT) {
  DuplicateTest<uint8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_u8_LT_BLK) {
  DuplicateTest<uint8_t>((ONE_BLK_SIZE - sizeof(uint8_t)) / sizeof(uint8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_u8_LT_MAX_RPT) {
  DuplicateTest<uint8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_u8_LT_RPT) {
  DuplicateTest<uint8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_u8_LT_MAX) {
  DuplicateTest<uint8_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                          (ONE_BLK_SIZE - sizeof(uint8_t))) / sizeof(uint8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_s8_LT_1BLK) {
  DuplicateTest<int8_t>(1);
}

TEST_F(TestApiDuplicateUT, Duplicate_s8_LT_1BLK2) {
  DuplicateTest<int8_t>(15);
}

TEST_F(TestApiDuplicateUT, Duplicate_s8_1BLK) {
  DuplicateTest<int8_t>(ONE_BLK_SIZE / sizeof(int8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_s8_1PRT) {
  DuplicateTest<int8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_s8_MAX_RPT) {
  DuplicateTest<int8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_s8_LT_BLK) {
  DuplicateTest<int8_t>((ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_s8_LT_MAX_RPT) {
  DuplicateTest<int8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_s8_LT_RPT) {
  DuplicateTest<int8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t));
}

TEST_F(TestApiDuplicateUT, Duplicate_s8_LT_MAX) {
  DuplicateTest<int8_t>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                          (ONE_BLK_SIZE - sizeof(int8_t))) / sizeof(int8_t));
}
}  // namespace ge
