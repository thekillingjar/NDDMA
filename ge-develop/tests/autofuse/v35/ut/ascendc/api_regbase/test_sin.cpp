/**
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
 */

/**
 * test_sin.cpp
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"

using namespace AscendC;

namespace ge {

template <typename T>
struct SinInputParam {
  T *x{};
  T *y{};
  T *exp{};
  uint32_t size{};
};

class TestRegbaseApiSinUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(SinInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x, param.x, param.size);
    Sin(l_y, l_x, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(SinInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      std::uniform_real_distribution distr(-6.28f, 6.28f); // -2 pi ~ 2 pi
      param.x[i] = distr(eng);
      param.exp[i] = param.exp[i] = static_cast<T>(sin(static_cast<double>(param.x[i])));
    }
  }

  template <typename T>
  static uint32_t Valid(SinInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      if (!DefaultCompare(param.y[i], param.exp[i])) {
        diff_count++;
      }
    }
    return diff_count;
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void SinTest(uint32_t size) {
    SinInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestRegbaseApiSinUT, Add_TensorTensor_Test) {
  SinTest<half>(ONE_BLK_SIZE / sizeof(half));
  SinTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  SinTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  SinTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  SinTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  SinTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  SinTest<float>(ONE_BLK_SIZE / sizeof(float));
  SinTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  SinTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  SinTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  SinTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  SinTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) / 2 /sizeof(float));
}

}  // namespace ge