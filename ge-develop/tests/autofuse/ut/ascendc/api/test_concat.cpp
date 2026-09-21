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
#include "concat.h"


namespace {
template<typename T>
T Gcd(T a, T b) {
  while (b != 0) {
    T tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}
}  // namespace
template<class T>
void GmToUb(LocalTensor<T>& local, T* gm, const ConcatParams<T, 2>& src) {
  for (int i = 0; i < src.shape[0]; i++) {
    for (int j = 0; j < src.stride[0]; j++) {
      if (j < src.shape[1]) {
        local.SetValue(i * src.stride[0] + j , gm[i * src.shape[1] + j]);
      } else {
        local.SetValue(i * src.stride[0] + j , 66);
      }
    }
  }
}

template<class T>
void UbToGm(T* gm, LocalTensor<T>& local, const ConcatParams<T, 2>& dst) {
  int align_num = (sizeof(T) == 4) ? 8 : 16;
  int align = (dst.shape[1] + align_num - 1) / align_num * align_num;
  for (int i = 0; i < dst.shape[0]; i++) {
    for (int j = 0; j < dst.stride[0]; j++) {
      if (j < dst.shape[1]) {
        gm[i * dst.shape[1] + j] = local.GetValue(i * dst.stride[0] + j);
      }
    }
  }
}

template<class T>
void GmToUb3D(LocalTensor<T>& local, T* gm, const ConcatParams<T, 3>& src, bool inter) {
  if (inter) {
    for (int i = 0; i < src.shape[0]; i++) {
      for (int j = 0; j < src.shape[1]; j++) {
        for (int k = 0; k < src.shape[2]; k++) {
          local.SetValue(i * src.stride[0]  + j * src.shape[2] + k , gm[i * src.shape[1] * src.shape[2] + j * src.shape[2] + k]);
        }
      }
    }
    return;
  }
  for (int i = 0; i < src.shape[0]; i++) {
    for (int j = 0; j < src.shape[1]; j++) {
      for (int k = 0; k < src.stride[1]; k++) {
        if (k < src.shape[2]) {
          local.SetValue(i * src.shape[1] * src.stride[1] + j * src.stride[1] + k , gm[i * src.shape[1] * src.shape[2] + j * src.shape[2] + k]);
        } else {
          local.SetValue(i * src.shape[1] * src.stride[1] + j * src.stride[1] + k , 66);
        }
     }
    }
  }
}

template<class T>
void UbToGm3D(T* gm, LocalTensor<T>& local, const ConcatParams<T, 3>& dst, bool inter) {
  if (inter) {
    for (int i = 0; i < dst.shape[0]; i++) {
      for (int j = 0; j < dst.shape[1]; j++) {
        for (int k = 0; k < dst.shape[2]; k++) {
          gm[i * dst.shape[1] * dst.shape[2] + j * dst.shape[2] + k] = local.GetValue(i * dst.stride[0] + j * dst.shape[2] + k);
        }
      }
    }
    return;
  }
  for (int i = 0; i < dst.shape[0]; i++) {
    for (int j = 0; j < dst.shape[1]; j++) {
      for (int k = 0; k < dst.stride[1]; k++) {
        if (k < dst.shape[2]) {
          gm[i * dst.shape[1] * dst.shape[2] + j * dst.shape[2] + k] = local.GetValue(i * dst.shape[1] * dst.stride[1] + j * dst.stride[1] + k);
        }
      }
    }
  }
}

template<class T>
void TestConcatExtendLastAxis(std::vector<uint32_t> &input1_shape,
  std::vector<uint32_t> &input2_shape, std::vector<uint32_t> &output_shape) {
  int input1_size = input1_shape[0] * input1_shape[1];
  int input2_size = input2_shape[0] * input2_shape[1];
  int output_size = output_shape[0] * output_shape[1];

  T *x1 = (T*)AscendC::GmAlloc(sizeof(T) * input1_size);
  T *x2 = (T*)AscendC::GmAlloc(sizeof(T) * input2_size);
  T *y = (T*)AscendC::GmAlloc(sizeof(T) * output_size);
  T *expect = (T*)AscendC::GmAlloc(sizeof(T) * output_size);

  for (int i = 0; i < input1_shape[0]; i++) {
    for (int j = 0; j < input1_shape[1]; j++) {
      x1[i*input1_shape[1] + j] = j;
    }
  }

  for (int i = 0; i < input2_shape[0]; i++) {
    for (int j = 0; j < input2_shape[1]; j++) {
      x2[i*input2_shape[1] + j] = j + input1_shape[1];
    }
  }

  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      expect[i * output_shape[1] + j] = j;
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t first_axis_size, uint32_t x1_last_axis_size, uint32_t x2_last_axis_size,
                uint32_t y_last_axis_size, T *x1, T *x2, T *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
    uint32_t align_size = 32 / sizeof(T);
    uint32_t x1_stride = (x1_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x2_stride = (x2_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t y_stride = ((y_last_axis_size + align_size - 1) / align_size) * align_size;
    tpipe.InitBuffer(x1buf, sizeof(T) * first_axis_size * x1_stride);
    tpipe.InitBuffer(x2buf, sizeof(T) * first_axis_size * x2_stride);
    tpipe.InitBuffer(ybuf,  sizeof(T) * first_axis_size * y_stride);
    tpipe.InitBuffer(tmp, sizeof(T) * 8 * 1024);

    auto l_x1 = x1buf.Get<T>();
    auto l_x2 = x2buf.Get<T>();
    auto l_y  = ybuf.Get<T>();
    auto l_tmp = tmp.Get<uint8_t>();

    const uint32_t concat_dim = 1;
    ConcatParams<T, 2> dst = {
      {first_axis_size, y_last_axis_size},
      {y_stride, 1},
      &l_y,
    };

    ConcatParams<T, 2> srcs[2] = {
      {{first_axis_size, x1_last_axis_size}, {x1_stride, 1}, &l_x1,},
      {{first_axis_size, x2_last_axis_size}, {x2_stride, 1}, &l_x2,},
    };

    GmToUb(l_x1, x1, srcs[0]);
    GmToUb(l_x2, x2, srcs[1]);

    ConcatExtend<T, 2, 2>(dst, srcs, concat_dim, l_tmp);

    UbToGm(y, l_y, dst);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(kernel, 1, input1_shape[0], input1_shape[1], input2_shape[1], output_shape[1], x1, x2, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      auto diff = (double)(y[i*output_shape[1] + j] - expect[i*output_shape[1] + j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  GmFree(x1);
  GmFree(x2);
  GmFree(y);
  GmFree(expect);
}

TEST(TestApiConcat, Test_concat_last_axis) {
  std::vector<std::vector<std::vector<uint32_t>>> shape = {
    {{1, 1}, {1, 1}, {1, 2},},
    {{5, 3}, {5, 4}, {5, 7},},
    {{13, 3}, {13, 2}, {13, 5},},
    {{13, 3}, {13, 6}, {13, 9},},
    {{16, 2}, {16, 3}, {16, 5},},
    {{16, 13}, {16, 7}, {16, 20}},
    {{31, 17}, {31, 6}, {31, 23},},
    {{31, 33}, {31, 6}, {31, 39},},
    {{32, 8}, {32, 16}, {32, 24},},
    {{64, 14}, {64, 13}, {64, 27}},
    {{31, 7}, {31, 97}, {31, 104},},
    {{31, 1}, {31, 47}, {31, 48},},
    {{16, 15}, {16, 150}, {16, 165}},
    {{15, 31}, {15, 129}, {15, 160}},
    {{15, 300}, {15, 129}, {15, 429}},
    {{23, 1}, {23, 127}, {23, 128}},
    {{48, 10}, {48, 1}, {48, 11}},
    {{33, 13}, {33, 17}, {33, 30}},
    {{2, 2}, {2, 246}, {2, 248}},
    };
  for (auto idx = 0; idx < shape.size(); idx++) {
    TestConcatExtendLastAxis<float>(shape[idx][0], shape[idx][1], shape[idx][2]);
    TestConcatExtendLastAxis<half>(shape[idx][0], shape[idx][1], shape[idx][2]);
    TestConcatExtendLastAxis<uint8_t>(shape[idx][0], shape[idx][1], shape[idx][2]);
    TestConcatExtendLastAxis<int64_t>(shape[idx][0], shape[idx][1], shape[idx][2]);
  }
}

template<class T>
void Test8inputConcatExtendLastAxis(std::vector<uint32_t> &input1_shape,
  std::vector<uint32_t> &input2_shape,
  std::vector<uint32_t> &input3_shape,
  std::vector<uint32_t> &input4_shape,
  std::vector<uint32_t> &input5_shape,
  std::vector<uint32_t> &input6_shape,
  std::vector<uint32_t> &input7_shape,
  std::vector<uint32_t> &input8_shape) {
  int input1_size = input1_shape[0] * input1_shape[1];
  int input2_size = input2_shape[0] * input2_shape[1];
  int input3_size = input3_shape[0] * input3_shape[1];
  int input4_size = input4_shape[0] * input4_shape[1];
  int input5_size = input5_shape[0] * input5_shape[1];
  int input6_size = input6_shape[0] * input6_shape[1];
  int input7_size = input7_shape[0] * input7_shape[1];
  int input8_size = input8_shape[0] * input8_shape[1];
  std::vector<uint32_t> output_shape;
  output_shape.push_back(input1_shape[0]);
  output_shape.push_back(input1_shape[1] + input2_shape[1] + input3_shape[1] +
    input4_shape[1] + input5_shape[1] + input6_shape[1] + input7_shape[1] + input8_shape[1]);
  int output_size = output_shape[0] * output_shape[1];
  
  T *x1 = (T*)AscendC::GmAlloc(sizeof(T) * input1_size);
  T *x2 = (T*)AscendC::GmAlloc(sizeof(T) * input2_size);
  T *x3 = (T*)AscendC::GmAlloc(sizeof(T) * input3_size);
  T *x4 = (T*)AscendC::GmAlloc(sizeof(T) * input4_size);
  T *x5 = (T*)AscendC::GmAlloc(sizeof(T) * input5_size);
  T *x6 = (T*)AscendC::GmAlloc(sizeof(T) * input6_size);
  T *x7 = (T*)AscendC::GmAlloc(sizeof(T) * input7_size);
  T *x8 = (T*)AscendC::GmAlloc(sizeof(T) * input8_size);
  T *y = (T*)AscendC::GmAlloc(sizeof(T) * output_size);
  T *expect = (T*)AscendC::GmAlloc(sizeof(T) * output_size);
  std::vector<T *> to_free{x1, x2, x3, x4, x5, x6, x7, x8, y, expect};

  for (int i = 0; i < input1_shape[0]; i++) {
    for (int j = 0; j < input1_shape[1]; j++) {
      x1[i*input1_shape[1] + j] = j;
    }
  }

  for (int i = 0; i < input2_shape[0]; i++) {
    for (int j = 0; j < input2_shape[1]; j++) {
      x2[i*input2_shape[1] + j] = j + input1_shape[1];
    }
  }

  for (int i = 0; i < input3_shape[0]; i++) {
    for (int j = 0; j < input3_shape[1]; j++) {
      x3[i*input3_shape[1] + j] = j + input1_shape[1] + input2_shape[1];
    }
  }

  for (int i = 0; i < input4_shape[0]; i++) {
    for (int j = 0; j < input4_shape[1]; j++) {
      x4[i*input4_shape[1] + j] = j + input1_shape[1] + input2_shape[1] + input3_shape[1];
    }
  }

  for (int i = 0; i < input5_shape[0]; i++) {
    for (int j = 0; j < input5_shape[1]; j++) {
      x5[i*input5_shape[1] + j] = j + input1_shape[1] + input2_shape[1] + input3_shape[1] + input4_shape[1];
    }
  }

  for (int i = 0; i < input6_shape[0]; i++) {
    for (int j = 0; j < input6_shape[1]; j++) {
      x6[i*input6_shape[1] + j] = j + input1_shape[1] + input2_shape[1] + input3_shape[1] + input4_shape[1]
      + input5_shape[1];
    }
  }

  for (int i = 0; i < input7_shape[0]; i++) {
    for (int j = 0; j < input7_shape[1]; j++) {
      x7[i*input7_shape[1] + j] = j + input1_shape[1] + input2_shape[1] + input3_shape[1] + input4_shape[1]
      + input5_shape[1] + input6_shape[1];
    }
  }

  for (int i = 0; i < input8_shape[0]; i++) {
    for (int j = 0; j < input8_shape[1]; j++) {
      x8[i*input8_shape[1] + j] = j + input1_shape[1] + input2_shape[1] + input3_shape[1] + input4_shape[1]
      + input5_shape[1] + input6_shape[1] + input7_shape[1];
    }
  }

  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      expect[i * output_shape[1] + j] = j;
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint32_t first_axis_size, uint32_t x1_last_axis_size, uint32_t x2_last_axis_size,
    uint32_t x3_last_axis_size,
    uint32_t x4_last_axis_size,
    uint32_t x5_last_axis_size,
    uint32_t x6_last_axis_size,
    uint32_t x7_last_axis_size,
    uint32_t x8_last_axis_size,
    uint32_t y_last_axis_size, T *x1, T *x2, T *x3, T *x4, T *x5, T *x6, T *x7, T *x8, T *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, x3buf, x4buf, x5buf, x6buf, x7buf, x8buf, ybuf, tmp;
    uint32_t align_size = 32 / sizeof(T);
    uint32_t x1_stride = (x1_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x2_stride = (x2_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x3_stride = (x3_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x4_stride = (x4_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x5_stride = (x5_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x6_stride = (x6_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x7_stride = (x7_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x8_stride = (x8_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t y_stride = ((y_last_axis_size + align_size - 1) / align_size) * align_size;
    tpipe.InitBuffer(x1buf, sizeof(T) * first_axis_size * x1_stride);
    tpipe.InitBuffer(x2buf, sizeof(T) * first_axis_size * x2_stride);
    tpipe.InitBuffer(x3buf, sizeof(T) * first_axis_size * x3_stride);
    tpipe.InitBuffer(x4buf, sizeof(T) * first_axis_size * x4_stride);
    tpipe.InitBuffer(x5buf, sizeof(T) * first_axis_size * x5_stride);
    tpipe.InitBuffer(x6buf, sizeof(T) * first_axis_size * x6_stride);
    tpipe.InitBuffer(x7buf, sizeof(T) * first_axis_size * x7_stride);
    tpipe.InitBuffer(x8buf, sizeof(T) * first_axis_size * x8_stride);
    tpipe.InitBuffer(ybuf,  sizeof(T) * first_axis_size * y_stride);
    tpipe.InitBuffer(tmp, 4 * 16 * 1024);

    auto l_x1 = x1buf.Get<T>();
    auto l_x2 = x2buf.Get<T>();
    auto l_x3 = x3buf.Get<T>();
    auto l_x4 = x4buf.Get<T>();
    auto l_x5 = x5buf.Get<T>();
    auto l_x6 = x6buf.Get<T>();
    auto l_x7 = x7buf.Get<T>();
    auto l_x8 = x8buf.Get<T>();
    auto l_y  = ybuf.Get<T>();
    auto l_tmp = tmp.Get<uint8_t>();

    const uint32_t concat_dim = 1;
    ConcatParams<T, 2> dst = {
      {first_axis_size, y_last_axis_size},
      {y_stride, 1},
      &l_y,
    };

    ConcatParams<T, 2> srcs[8] = {
      {{first_axis_size, x1_last_axis_size}, {x1_stride, 1}, &l_x1,},
      {{first_axis_size, x2_last_axis_size}, {x2_stride, 1}, &l_x2,},
      {{first_axis_size, x3_last_axis_size}, {x3_stride, 1}, &l_x3,},
      {{first_axis_size, x4_last_axis_size}, {x4_stride, 1}, &l_x4,},
      {{first_axis_size, x5_last_axis_size}, {x5_stride, 1}, &l_x5,},
      {{first_axis_size, x6_last_axis_size}, {x6_stride, 1}, &l_x6,},
      {{first_axis_size, x7_last_axis_size}, {x7_stride, 1}, &l_x7,},
      {{first_axis_size, x8_last_axis_size}, {x8_stride, 1}, &l_x8,},
    };

    GmToUb(l_x1, x1, srcs[0]);
    GmToUb(l_x2, x2, srcs[1]);
    GmToUb(l_x3, x3, srcs[2]);
    GmToUb(l_x4, x4, srcs[3]);
    GmToUb(l_x5, x5, srcs[4]);
    GmToUb(l_x6, x6, srcs[5]);
    GmToUb(l_x7, x7, srcs[6]);
    GmToUb(l_x8, x8, srcs[7]);

    ConcatExtend<T, 2, 8>(dst, srcs, concat_dim, l_tmp);

    UbToGm(y, l_y, dst);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(kernel, 1, input1_shape[0], input1_shape[1], input2_shape[1], 
    input3_shape[1],
    input4_shape[1],
    input5_shape[1],
    input6_shape[1],
    input7_shape[1],
    input8_shape[1],
    output_shape[1], x1, x2, x3, x4, x5, x6, x7, x8, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      auto diff = (double)(y[i*output_shape[1] + j] - expect[i*output_shape[1] + j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  for (auto addr : to_free) {
    GmFree(addr);
  }
}

TEST(TestApiConcat, Test_8input_concat_last_axis) {
  std::vector<std::vector<std::vector<uint32_t>>> shape = {
    {{16, 5}, {16, 1}, {16, 13}, {16, 2}, {16, 17}, {16, 9}, {16, 1}, {16, 1},},
    {{31, 7}, {31, 104}, {31, 13}, {31, 2}, {31, 105}, {31, 9}, {31, 1}, {31, 1},},
    {{5, 5}, {5, 1}, {5, 13}, {5, 2}, {5, 17}, {5, 9}, {5, 1}, {5, 1},},
    {{32, 5}, {32, 1}, {32, 13}, {32, 65}, {32, 17}, {32, 9}, {32, 1}, {32, 51},},
    {{15, 15}, {32, 224}, {32, 13}, {32, 2}, {32, 225}, {32, 9}, {32, 1}, {32, 1},},
    {{15, 5}, {15, 1}, {15, 13}, {15, 150}, {15, 135}, {15, 9}, {15, 1}, {15, 1},},
    {{17, 31}, {17, 128}, {17, 1}, {17, 32}, {17, 135}, {17, 9}, {17, 1}, {17, 1},},
    {{17, 1}, {17, 1}, {17, 1}, {17, 1}, {17, 1}, {17, 1}, {17, 1}, {17, 1},},
    {{2, 115}, {2, 115}, {2, 115}, {2, 115}, {2, 115}, {2, 115}, {2, 115}, {2, 115},},
    {{16, 32}, {16, 32}, {16, 32}, {16, 32}, {16, 32}, {16, 32}, {16, 32}, {16, 32},},
    {{33, 13}, {33, 17}, {33, 1}, {33, 17}, {33, 8}, {33, 8}, {33, 8}, {33, 8},},
    {{3, 1}, {3, 1}, {3, 1}, {3, 1}, {3, 16}, {3, 16}, {3, 16}, {3, 16},},
    {{1, 8}, {1, 257}, {1, 257}, {1, 12}, {1, 383}, {1, 2}, {1, 1}, {1, 170},},
  };
  for (auto idx = 0; idx < shape.size(); idx++) {
    Test8inputConcatExtendLastAxis<float>(shape[idx][0], shape[idx][1],
      shape[idx][2], shape[idx][3], shape[idx][4], shape[idx][5], shape[idx][6],
      shape[idx][7]);
    Test8inputConcatExtendLastAxis<half>(shape[idx][0], shape[idx][1],
      shape[idx][2], shape[idx][3], shape[idx][4], shape[idx][5], shape[idx][6],
      shape[idx][7]);
    Test8inputConcatExtendLastAxis<uint8_t>(shape[idx][0], shape[idx][1],
      shape[idx][2], shape[idx][3], shape[idx][4], shape[idx][5], shape[idx][6],
      shape[idx][7]);
    Test8inputConcatExtendLastAxis<int64_t>(shape[idx][0], shape[idx][1],
      shape[idx][2], shape[idx][3], shape[idx][4], shape[idx][5], shape[idx][6],
      shape[idx][7]);
  }
}

template<class T>
void TestConcatExtendInterAxis(std::vector<uint32_t> &input1_shape,
  std::vector<uint32_t> &input2_shape, std::vector<uint32_t> &output_shape, const uint32_t concat_dim) {
  int align = 32 / sizeof(T);
  int input1_size = input1_shape[0] * input1_shape[1] * ((input1_shape[2] + align - 1) / align * align);
  int input2_size = input2_shape[0] * input2_shape[1] * ((input2_shape[2] + align - 1) / align * align);
  int output_size = output_shape[0] * output_shape[1] * ((output_shape[2] + align - 1) / align * align);

  T *x1 = (T*)AscendC::GmAlloc(sizeof(T) * input1_size);
  T *x2 = (T*)AscendC::GmAlloc(sizeof(T) * input2_size);
  T *y = (T*)AscendC::GmAlloc(sizeof(T) * output_size);
  T *expect = (T*)AscendC::GmAlloc(sizeof(T) * output_size);

  for (int i = 0; i < input1_shape[0]; i++) {
    for (int j = 0; j < input1_shape[1]; j++) {
      for (int k = 0; k < input1_shape[2]; k++) {
        x1[i*(input1_shape[1] * input1_shape[2]) + j * input1_shape[2] + k] = k;
      }
    }
  }

  for (int i = 0; i < input2_shape[0]; i++) {
    for (int j = 0; j < input2_shape[1]; j++) {
      for (int k = 0; k < input2_shape[2]; k++) {
        x2[i*(input2_shape[1] * input2_shape[2])  + j * input2_shape[2] + k] = k;
      }
    }
  }
  
  if (concat_dim == 1) {
    for (int i = 0; i < output_shape[0]; i++) {
      for (int j = 0; j < output_shape[1]; j++) {
        for (int k = 0; k < output_shape[2]; k++) {
          expect[i*output_shape[1] * output_shape[2] + j * output_shape[2] + k] = k;
        }
      }
    }
  } else {
    for (int i = 0; i < output_shape[0]; i++) {
      for (int j = 0; j < output_shape[1]; j++) {
        for (int k = 0; k < output_shape[2]; k++) {
          if (k < input1_shape[2]) {
            expect[i*output_shape[1] * output_shape[2] + j * output_shape[2] + k] = k;
          } else {
            expect[i*output_shape[1] * output_shape[2] + j * output_shape[2] + k] = k - input1_shape[2];
          }
        }
      }
    }
  }
  

  // 构造Api调用函数
  auto kernel = [](uint32_t first_axis_size, uint32_t x1_inter_axis_size, uint32_t x2_inter_axis_size,
                uint32_t x1_last_axis_size, uint32_t x2_last_axis_size, T *x1, T *x2, T *y, uint32_t concat_dim) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
    uint32_t align_size = 32 / sizeof(T);
    uint32_t x1_inter_stride = (x1_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x2_inter_stride = (x2_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t y_stride = ((x1_last_axis_size + x2_last_axis_size) + align_size - 1) / align_size  * align_size;
    uint32_t x1_last_axis_size_aligned = (x1_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t x2_last_axis_size_aligned = (x2_last_axis_size + align_size - 1) / align_size  * align_size;
    uint32_t y_inter_size = 0;
    uint32_t y_last_size = 0;
    uint32_t y_inter_stride = 0;
    tpipe.InitBuffer(x1buf, sizeof(T) * first_axis_size * x1_inter_axis_size * x1_inter_stride);
    tpipe.InitBuffer(x2buf, sizeof(T) * first_axis_size * x2_inter_axis_size * x2_inter_stride);
    if (concat_dim == 1) {
      x1_inter_stride = x1_inter_axis_size * x1_last_axis_size_aligned;
      x2_inter_stride = x2_inter_axis_size * x2_last_axis_size_aligned;
      y_inter_size = x1_inter_axis_size + x2_inter_axis_size;
      y_last_size = x1_last_axis_size;
      y_inter_stride = y_inter_size * ((y_last_size  + align_size - 1) / align_size  * align_size);
      tpipe.InitBuffer(ybuf,  sizeof(T) * first_axis_size * y_inter_stride);
    } else {
      y_inter_size = x1_inter_axis_size;
      y_last_size = x1_last_axis_size + x2_last_axis_size;
      y_inter_stride = y_stride;
      tpipe.InitBuffer(ybuf,  sizeof(T) * first_axis_size * y_inter_size *y_stride);
    }

    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x1 = x1buf.Get<T>();
    auto l_x2 = x2buf.Get<T>();
    auto l_y  = ybuf.Get<T>();
    auto l_tmp = tmp.Get<uint8_t>();

    ConcatParams<T, 3> srcs[2];
    ConcatParams<T, 3> dst;
    bool inter = false;
    if (concat_dim == 1) {
      inter = true;
      srcs[0] = {{first_axis_size, x1_inter_axis_size, x1_last_axis_size}, {x1_inter_stride, x1_last_axis_size_aligned, 1}, &l_x1,};
      srcs[1] = {{first_axis_size, x2_inter_axis_size, x2_last_axis_size}, {x2_inter_stride, x2_last_axis_size_aligned, 1}, &l_x2,};
      dst = {
        {first_axis_size, y_inter_size, y_last_size},
        {y_inter_stride, x1_last_axis_size_aligned, 1},
        &l_y,
      };
    } else {
      srcs[0] = {{first_axis_size, x1_inter_axis_size, x1_last_axis_size}, {x1_inter_axis_size * x1_inter_stride, x1_inter_stride, 1}, &l_x1,};
      srcs[1] = {{first_axis_size, x2_inter_axis_size, x2_last_axis_size}, {x2_inter_axis_size * x2_inter_stride, x2_inter_stride, 1}, &l_x2,};
      dst = {
        {first_axis_size, y_inter_size, y_last_size},
        {y_inter_size * y_inter_stride, y_inter_stride, 1},
        &l_y,
      };
    }

    GmToUb3D(l_x1, x1, srcs[0], false);
    GmToUb3D(l_x2, x2, srcs[1], false);

    ConcatExtend<T, 3, 2>(dst, srcs, concat_dim, l_tmp);

    UbToGm3D(y, l_y, dst, false);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(kernel, 1, input1_shape[0], input1_shape[1], input2_shape[1], input1_shape[2], input2_shape[2], x1, x2, y, concat_dim);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      for (int k =0; k < output_shape[2]; k++) {
        auto diff = (double)(y[i*output_shape[1] * output_shape[2] + j * output_shape[2] + k] -
          expect[i*output_shape[1] * output_shape[2] + j * output_shape[2] + k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  std::vector<T *> to_free{x1, x2, y, expect};
  for (auto addr : to_free) {
    GmFree(addr);
  }
}

TEST(TestApiConcat, Test_concat_inter_axis_001) {
   std::vector<std::vector<uint32_t>> input1_shape = {{3, 4, 133}};
   std::vector<std::vector<uint32_t>> input2_shape = {{3, 4, 133}};
   std::vector<std::vector<uint32_t>> output_shape = {{3, 8, 133}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatExtendInterAxis<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
    TestConcatExtendInterAxis<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
    TestConcatExtendInterAxis<uint8_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
    TestConcatExtendInterAxis<int64_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
  }
}

TEST(TestApiConcat, Test_concat_inter_axis_002) {
  std::vector<std::vector<uint32_t>> input1_shape = {{3, 1, 133}};
  std::vector<std::vector<uint32_t>> input2_shape = {{3, 1, 133}};
  std::vector<std::vector<uint32_t>> output_shape = {{3, 2, 133}};
 for (auto idx = 0; idx < input1_shape.size(); idx++) {
   TestConcatExtendInterAxis<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
   TestConcatExtendInterAxis<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
   TestConcatExtendInterAxis<uint8_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
   TestConcatExtendInterAxis<int64_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
 }
}

TEST(TestApiConcat, Test_concat_inter_axis_003) {
  std::vector<std::vector<uint32_t>> input1_shape = {{3, 1, 133}};
  std::vector<std::vector<uint32_t>> input2_shape = {{3, 8, 133}};
  std::vector<std::vector<uint32_t>> output_shape = {{3, 9, 133}};
 for (auto idx = 0; idx < input1_shape.size(); idx++) {
   TestConcatExtendInterAxis<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
   TestConcatExtendInterAxis<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
   TestConcatExtendInterAxis<uint8_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
   TestConcatExtendInterAxis<int64_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
 }
}

TEST(TestApiConcat, Test_concat_inter_axis_004) {
  std::vector<std::vector<uint32_t>> input1_shape = {{3, 1, 1}};
  std::vector<std::vector<uint32_t>> input2_shape = {{3, 1, 1}};
  std::vector<std::vector<uint32_t>> output_shape = {{3, 2, 1}};
 for (auto idx = 0; idx < input1_shape.size(); idx++) {
   TestConcatExtendInterAxis<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
   TestConcatExtendInterAxis<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
   TestConcatExtendInterAxis<uint8_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
   TestConcatExtendInterAxis<int64_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 1);
 }
}

TEST(TestApiConcat, Test_concat_last_axis_and_vectorized_axis_great_than_2_001) {
   std::vector<std::vector<uint32_t>> input1_shape = {{3, 4, 133}};
   std::vector<std::vector<uint32_t>> input2_shape = {{3, 4, 133}};
   std::vector<std::vector<uint32_t>> output_shape = {{3, 4, 133 * 2}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatExtendInterAxis<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], 2);
    TestConcatExtendInterAxis<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], 2);
    TestConcatExtendInterAxis<uint8_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 2);
    TestConcatExtendInterAxis<int64_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 2);
  }
}

TEST(TestApiConcat, Test_concat_last_axis_and_vectorized_axis_great_than_2_002) {
  std::vector<std::vector<uint32_t>> input1_shape = {{3, 1, 133}};
  std::vector<std::vector<uint32_t>> input2_shape = {{3, 1, 133}};
  std::vector<std::vector<uint32_t>> output_shape = {{3, 1, 133 * 2}};
 for (auto idx = 0; idx < input1_shape.size(); idx++) {
   TestConcatExtendInterAxis<float>(input1_shape[idx], input2_shape[idx], output_shape[idx], 2);
   TestConcatExtendInterAxis<half>(input1_shape[idx], input2_shape[idx], output_shape[idx], 2);
   TestConcatExtendInterAxis<uint8_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 2);
   TestConcatExtendInterAxis<int64_t>(input1_shape[idx], input2_shape[idx], output_shape[idx], 2);
 }
}

template<class ConcatContextType>
void TestConcatSmallTailExtend(std::vector<uint32_t> &input1_shape,
                               std::vector<uint32_t> &input2_shape,
                               std::vector<uint32_t> &output_shape,
                               bool padded = false) {
  using T = typename ConcatContextType::DataType;
  auto input1_size = input1_shape[0] * input1_shape[1];
  auto input2_size = input2_shape[0] * input2_shape[1];
  auto output_size = output_shape[0] * output_shape[1];
  T *x1 = (T *) AscendC::GmAlloc(sizeof(T) * input1_size);
  T *x2 = (T *) AscendC::GmAlloc(sizeof(T) * input2_size);
  T *y = (T *) AscendC::GmAlloc(sizeof(T) * output_size);
  T *expect = (T *) AscendC::GmAlloc(sizeof(T) * output_size);

  int32_t value = 1;
  for (int i = 0; i < input1_shape[0]; i++) {
    for (int j = 0; j < input1_shape[1]; j++) {
      x1[i * input1_shape[1] + j] = value++;
      expect[i * output_shape[1] + j] = x1[i * input1_shape[1] + j];
    }
  }

  value = 1;
  for (int i = 0; i < input2_shape[0]; i++) {
    for (int j = 0; j < input2_shape[1]; j++) {
      x2[i * input2_shape[1] + j] = value++;
      expect[i * output_shape[1] + input1_shape[1] + j] = x2[i * input2_shape[1] + j] ;
    }
  }

  // 构造Api调用函数
  auto kernel = [padded](uint32_t first_axis_size,
                   uint32_t x1_last_axis_size,
                   uint32_t x2_last_axis_size,
                   uint32_t y_last_axis_size,
                   T *x1,
                   T *x2,
                   T *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
    uint32_t align_size = 32 / sizeof(T);
    uint32_t x1_stride = (!padded) ? x1_last_axis_size : (x1_last_axis_size + align_size - 1) / align_size * align_size;
    uint32_t x2_stride = (!padded) ? x2_last_axis_size : (x2_last_axis_size + align_size - 1) / align_size * align_size;
    uint32_t y_stride = (!padded) ? y_last_axis_size : ((y_last_axis_size + align_size - 1) / align_size) * align_size;
    tpipe.InitBuffer(x1buf, sizeof(T) * first_axis_size * x1_stride);
    tpipe.InitBuffer(x2buf, sizeof(T) * first_axis_size * x2_stride);
    tpipe.InitBuffer(ybuf, sizeof(T) * first_axis_size * y_stride);
    tpipe.InitBuffer(tmp, 64 * 1024);

    auto l_x1 = x1buf.Get<T>();
    auto l_x2 = x2buf.Get<T>();
    auto l_y = ybuf.Get<T>();
    auto l_tmp = tmp.Get<uint8_t>();

    const uint32_t concat_dim = 1;
    ConcatParams<T, 2> dst = {
        {first_axis_size, y_last_axis_size},
        {y_last_axis_size, 1},
        &l_y,
    };

    ConcatParams<T, 2> srcs[2] = {
        {{first_axis_size, x1_last_axis_size}, {x1_stride, 1}, &l_x1,},
        {{first_axis_size, x2_last_axis_size}, {x2_stride, 1}, &l_x2,},
    };

    GmToUb(l_x1, x1, srcs[0]);
    GmToUb(l_x2, x2, srcs[1]);

    auto gcd = Gcd(16U, x1_last_axis_size);
    gcd = Gcd(gcd, x2_last_axis_size);
    if constexpr (IsSameDim<ConcatContextType>()) {
      gcd = 1;
    }

    const uint32_t kScaleToB16 = sizeof(T) / sizeof(uint16_t);
    const uint32_t kEltNumPerBlock = kAddrListSize * kDataBlockSize / sizeof(T);
    ConcatTiling<2> concat_tiling{
        .gcd = gcd,
        .tmp_buf_size = 64 * 1024,
        .dst_dim_size = y_last_axis_size,
        .dst_row_num_unit = concat_tiling.dst_dim_size * kScaleToB16,
        .max_repeat_times = (concat_tiling.tmp_buf_size >> 10U) / (concat_tiling.dst_dim_size / concat_tiling.gcd),
        .max_element_num = concat_tiling.max_repeat_times * (concat_tiling.dst_dim_size / concat_tiling.gcd) * kEltNumPerBlock,
        .max_orig_row_num = concat_tiling.max_element_num / concat_tiling.dst_dim_size, // 非尾块, 每次loop的原始行数
        .src_dim_sizes = {x1_last_axis_size, x2_last_axis_size},
        .src_strides = {concat_tiling.max_orig_row_num * x1_stride, concat_tiling.max_orig_row_num * x2_stride},
        .src_buffer_offsets = {0, concat_tiling.max_repeat_times * (concat_tiling.src_dim_sizes[0] / concat_tiling.gcd) * kEltNumPerBlock},
        // strides[index] * ConcatContextType::kDataTypeSize / kDataBlockSize),
        .gather_mask_repeat_strides = {static_cast<uint16_t>(x1_stride * sizeof(T) / 32), static_cast<uint16_t>(x2_stride * sizeof(T) / 32)},
        .gather_mask_dim_sizes = {x1_last_axis_size, x2_last_axis_size}
    };

    concat_tiling.first_copy_repeat_times =
        concat_tiling.max_repeat_times * kAddrListSize / kScaleToB16 / concat_tiling.gcd;
    concat_tiling.last_trans_repeat_times =
        concat_tiling.max_repeat_times * (concat_tiling.dst_dim_size / concat_tiling.gcd);
    concat_tiling.per_repeat_size = (concat_tiling.dst_dim_size / concat_tiling.gcd) * kEltNumPerBlock;

    ConcatInputList<T, ConcatContextType::kInputNum> input_list {
        .src_tensor_base_addrs = {(T *)l_x1.GetPhyAddr(), (T *)l_x2.GetPhyAddr()},
        .src_tensors = {&l_x1, &l_x2}
    };
    ConcatContextType context;
    context.total_row_num = first_axis_size;
    context.input_list = &input_list;
    ConcatExtendV2<ConcatContextType>(context, concat_tiling, l_y, l_tmp);

    UbToGm(y, l_y, dst);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(kernel, 1, input1_shape[0], input1_shape[1], input2_shape[1], output_shape[1], x1, x2, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      auto diff = (double) (y[i * output_shape[1] + j] - expect[i * output_shape[1] + j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  std::vector<T *> to_free{x1, x2, y, expect};
  for (auto addr : to_free) {
    GmFree(addr);
  }
}

template<class ConcatContextType>
void TestConcatSmallTailExtend3Inputs(std::vector<uint32_t> &input1_shape,
                                      std::vector<uint32_t> &input2_shape,
                                      std::vector<uint32_t> &input3_shape,
                                      std::vector<uint32_t> &output_shape,
                                      bool padded = false) {
  using T = typename ConcatContextType::DataType;
  auto input1_size = input1_shape[0] * input1_shape[1];
  auto input2_size = input2_shape[0] * input2_shape[1];
  auto input3_size = input3_shape[0] * input3_shape[1];
  auto output_size = output_shape[0] * output_shape[1];
  T *x1 = (T *) AscendC::GmAlloc(sizeof(T) * input1_size);
  T *x2 = (T *) AscendC::GmAlloc(sizeof(T) * input2_size);
  T *x3 = (T *) AscendC::GmAlloc(sizeof(T) * input3_size);
  T *y = (T *) AscendC::GmAlloc(sizeof(T) * output_size);
  T *expect = (T *) AscendC::GmAlloc(sizeof(T) * output_size);

  int32_t value = 1;
  for (int i = 0; i < input1_shape[0]; i++) {
    for (int j = 0; j < input1_shape[1]; j++) {
      x1[i * input1_shape[1] + j] = value++;
      expect[i * output_shape[1] + j] = x1[i * input1_shape[1] + j];
    }
  }

  value = 1;
  for (int i = 0; i < input2_shape[0]; i++) {
    for (int j = 0; j < input2_shape[1]; j++) {
      x2[i * input2_shape[1] + j] = value++;
      expect[i * output_shape[1] + input1_shape[1] + j] = x2[i * input2_shape[1] + j] ;
    }
  }

  value = 1;
  for (int i = 0; i < input3_shape[0]; i++) {
    for (int j = 0; j < input3_shape[1]; j++) {
      x3[i * input3_shape[1] + j] = value++;
      expect[i * output_shape[1] + input1_shape[1] + input2_shape[1] + j] = x3[i * input3_shape[1] + j] ;
    }
  }

  // 构造Api调用函数
  auto kernel = [padded](uint32_t first_axis_size,
                         uint32_t x1_last_axis_size,
                         uint32_t x2_last_axis_size,
                         uint32_t x3_last_axis_size,
                         uint32_t y_last_axis_size,
                         T *x1,
                         T *x2,
                         T *x3,
                         T *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, x3buf, ybuf, tmp;
    uint32_t align_size = 32 / sizeof(T);
    uint32_t x1_stride = (!padded) ? x1_last_axis_size : (x1_last_axis_size + align_size - 1) / align_size * align_size;
    uint32_t x2_stride = (!padded) ? x2_last_axis_size : (x2_last_axis_size + align_size - 1) / align_size * align_size;
    uint32_t x3_stride = (!padded) ? x3_last_axis_size : (x3_last_axis_size + align_size - 1) / align_size * align_size;
    uint32_t y_stride = (!padded) ? y_last_axis_size : ((y_last_axis_size + align_size - 1) / align_size) * align_size;
    tpipe.InitBuffer(x1buf, sizeof(T) * first_axis_size * x1_stride);
    tpipe.InitBuffer(x2buf, sizeof(T) * first_axis_size * x2_stride);
    tpipe.InitBuffer(x3buf, sizeof(T) * first_axis_size * x3_stride);
    tpipe.InitBuffer(ybuf, sizeof(T) * first_axis_size * y_stride);
    tpipe.InitBuffer(tmp, 64 * 1024);

    auto l_x1 = x1buf.Get<T>();
    auto l_x2 = x2buf.Get<T>();
    auto l_x3 = x3buf.Get<T>();
    auto l_y = ybuf.Get<T>();
    auto l_tmp = tmp.Get<uint8_t>();
    const uint32_t concat_dim = 1;
    ConcatParams<T, 2> dst = {
        {first_axis_size, y_last_axis_size},
        {y_last_axis_size, 1},
        &l_y,
    };
    ConcatParams<T, 2> srcs[3] = {
        {{first_axis_size, x1_last_axis_size}, {x1_stride, 1}, &l_x1,},
        {{first_axis_size, x2_last_axis_size}, {x2_stride, 1}, &l_x2,},
        {{first_axis_size, x3_last_axis_size}, {x3_stride, 1}, &l_x3,},
    };

    GmToUb(l_x1, x1, srcs[0]);
    GmToUb(l_x2, x2, srcs[1]);
    GmToUb(l_x3, x3, srcs[2]);

    auto gcd = Gcd(16U, x1_last_axis_size);
    gcd = Gcd(gcd, x2_last_axis_size);
    gcd = Gcd(gcd, x3_last_axis_size);
    if constexpr (IsSameDim<ConcatContextType>()) {
      gcd = 1;
    }
    const uint32_t kScaleToB16 = sizeof(T) / sizeof(uint16_t);
    const uint32_t kEltNumPerBlock = kAddrListSize * kDataBlockSize / sizeof(T);

    ConcatTiling<3> concat_tiling{
        .gcd = gcd,
        .tmp_buf_size = 64 * 1024,
        .dst_dim_size = y_last_axis_size,
        .dst_row_num_unit = concat_tiling.dst_dim_size * kScaleToB16,
        .max_repeat_times = (concat_tiling.tmp_buf_size >> 10U) / (concat_tiling.dst_dim_size / concat_tiling.gcd),
        .max_element_num = concat_tiling.max_repeat_times * (concat_tiling.dst_dim_size / concat_tiling.gcd)
            * kEltNumPerBlock,
        .max_orig_row_num = concat_tiling.max_element_num / concat_tiling.dst_dim_size, // 非尾块, 每次loop的原始行数
        .src_dim_sizes = {x1_last_axis_size, x2_last_axis_size, x3_last_axis_size},
        .src_strides = {concat_tiling.max_orig_row_num * x1_stride, concat_tiling.max_orig_row_num * x2_stride,
                        concat_tiling.max_orig_row_num * x3_stride},
        .src_buffer_offsets = {
            0,
            concat_tiling.max_repeat_times * (concat_tiling.src_dim_sizes[0] / concat_tiling.gcd) * kEltNumPerBlock,
            concat_tiling.src_buffer_offsets[1] +
                concat_tiling.max_repeat_times * (concat_tiling.src_dim_sizes[1] / concat_tiling.gcd)
                    * kEltNumPerBlock},
        // strides[index] * ConcatContextType::kDataTypeSize / kDataBlockSize),
        .gather_mask_repeat_strides = {static_cast<uint16_t>(x1_stride * sizeof(T) / 32),
                                       static_cast<uint16_t>(x2_stride * sizeof(T) / 32),
                                       static_cast<uint16_t>(x3_stride * sizeof(T) / 32)},
        .gather_mask_dim_sizes = {x1_last_axis_size, x2_last_axis_size, x3_last_axis_size}
    };

    concat_tiling.first_copy_repeat_times =
        concat_tiling.max_repeat_times * kAddrListSize / kScaleToB16 / concat_tiling.gcd;
    concat_tiling.last_trans_repeat_times =
        concat_tiling.max_repeat_times * (concat_tiling.dst_dim_size / concat_tiling.gcd);
    concat_tiling.per_repeat_size = (concat_tiling.dst_dim_size / concat_tiling.gcd) * kEltNumPerBlock;

    std::cout << "tmp_buf_size = " << concat_tiling.tmp_buf_size << std::endl;
    std::cout << "dst_dim_size = " << concat_tiling.dst_dim_size << std::endl;
    std::cout << "dst_row_num_unit = " << concat_tiling.dst_row_num_unit << std::endl;
    std::cout << "max_repeat_times = " << concat_tiling.max_repeat_times << std::endl;
    std::cout << "max_element_num = " << concat_tiling.max_element_num << std::endl;

    ConcatContextType context;
    ConcatInputList<T, ConcatContextType::kInputNum> input_list {
      .src_tensor_base_addrs = {(T *)l_x1.GetPhyAddr(), (T *)l_x2.GetPhyAddr(), (T *)l_x3.GetPhyAddr()},
      .src_tensors = {&l_x1, &l_x2, &l_x3}
    };
    context.total_row_num = first_axis_size;
    context.input_list = &input_list;
    ConcatExtendV2<ConcatContextType>(context, concat_tiling, l_y, l_tmp);

    UbToGm(y, l_y, dst);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(kernel, 1, input1_shape[0], input1_shape[1], input2_shape[1], input3_shape[1], output_shape[1],
              x1, x2, x3, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      auto diff = (double) (y[i * output_shape[1] + j] - expect[i * output_shape[1] + j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  std::vector<T *> to_free{x1, x2, x3, y, expect};
  for (auto addr : to_free) {
    GmFree(addr);
  }
}

template<class ConcatContextType>
void TestConcatSmallTailExtend2ndLastDim(std::vector<uint32_t> &input1_shape,
                                         std::vector<uint32_t> &input2_shape,
                                         std::vector<uint32_t> &output_shape,
                                         const std::vector<bool> &padded) {
  using T = typename ConcatContextType::DataType;
  auto input1_size = input1_shape[0] * input1_shape[1] * input1_shape[2];
  auto input2_size = input2_shape[0] * input2_shape[1] * input2_shape[2];
  auto output_size = output_shape[0] * output_shape[1] * output_shape[2];
  T *x1 = (T *) AscendC::GmAlloc(sizeof(T) * input1_size);
  T *x2 = (T *) AscendC::GmAlloc(sizeof(T) * input2_size);
  T *y = (T *) AscendC::GmAlloc(sizeof(T) * output_size);
  T *expect = (T *) AscendC::GmAlloc(sizeof(T) * output_size);

  auto output_stride_1 = output_shape[2];
  auto output_stride_0 = output_stride_1 * output_shape[1];
  int32_t value = 1;
  for (int i = 0; i < input1_shape[0]; i++) {
    for (int j = 0; j < input1_shape[1]; j++) {
      for (int k = 0; k < input1_shape[2]; k++) {
        auto stride1 = input1_shape[2];
        auto stride0 = stride1 * input1_shape[1];
        x1[i * stride0 + j * stride1 + k] = value;
        expect[i * output_stride_0 + j * output_stride_1 + k] = value;
        value++;
      }
    }
  }

  value = 1;
  for (int i = 0; i < input2_shape[0]; i++) {
    for (int j = 0; j < input2_shape[1]; j++) {
      for (int k = 0; k < input2_shape[2]; k++) {
        auto stride1 = input2_shape[2];
        auto stride0 = stride1 * input2_shape[1];
        x2[i * stride0 + j * stride1 + k] = value;
        expect[i * output_stride_0 + j * output_stride_1 + (input1_shape[1] * input1_shape[2]) + k] = value;
        value++;
      }
    }
  }

  // 构造Api调用函数
  auto kernel =
      [&padded](uint32_t first_axis_size,
               uint32_t x1_sec_axis_size,
               uint32_t x2_sec_axis_size,
               uint32_t last_axis_size,
               T *x1,
               T *x2,
               T *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
    uint32_t align_size = 32 / sizeof(T);
    uint32_t x1_stride = (!padded[0]) ? last_axis_size : (last_axis_size + align_size - 1) / align_size * align_size;
    uint32_t x2_stride = (!padded[1]) ? last_axis_size : (last_axis_size + align_size - 1) / align_size * align_size;
    uint32_t y_stride = last_axis_size;
    tpipe.InitBuffer(x1buf, sizeof(T) * first_axis_size * x1_sec_axis_size * x1_stride);
    tpipe.InitBuffer(x2buf, sizeof(T) * first_axis_size * x2_sec_axis_size *x2_stride);
    tpipe.InitBuffer(ybuf, sizeof(T) * first_axis_size * (x1_sec_axis_size + x2_sec_axis_size ) * y_stride);
    tpipe.InitBuffer(tmp, 64 * 1024);

    auto l_x1 = x1buf.Get<T>();
    auto l_x2 = x2buf.Get<T>();
    auto l_y = ybuf.Get<T>();
    auto l_tmp = tmp.Get<uint8_t>();

    const uint32_t concat_dim = 1;
    ConcatParams<T, 2> dst = {
        {first_axis_size * (x1_sec_axis_size + x2_sec_axis_size), last_axis_size},
        {last_axis_size, 1},
        &l_y,
    };

    ConcatParams<T, 2> srcs[2] = {
        {{first_axis_size * x1_sec_axis_size, last_axis_size}, {x1_stride, 1}, &l_x1,},
        {{first_axis_size * x2_sec_axis_size, last_axis_size}, {x2_stride, 1}, &l_x2,},
    };

    GmToUb(l_x1, x1, srcs[0]);
    GmToUb(l_x2, x2, srcs[1]);

    auto gcd = Gcd(16U, x1_sec_axis_size * last_axis_size);
    gcd = Gcd(gcd, x2_sec_axis_size * last_axis_size);
    if constexpr (IsSameDim<ConcatContextType>()) {
      gcd = 1;
    }

    const uint32_t kScaleToB16 = sizeof(T) / sizeof(uint16_t);
    const uint32_t kEltNumPerBlock = kAddrListSize * kDataBlockSize / sizeof(T);
    ConcatTiling<2> concat_tiling{
        .gcd = gcd,
        .tmp_buf_size = 64 * 1024,
        .dst_dim_size = (x1_sec_axis_size + x2_sec_axis_size) * last_axis_size,
        .dst_row_num_unit = concat_tiling.dst_dim_size * kScaleToB16,
        .max_repeat_times = (concat_tiling.tmp_buf_size >> 10U) / (concat_tiling.dst_dim_size / concat_tiling.gcd),
        .max_element_num = concat_tiling.max_repeat_times * (concat_tiling.dst_dim_size / concat_tiling.gcd) * kEltNumPerBlock,
        .max_orig_row_num = concat_tiling.max_element_num / concat_tiling.dst_dim_size, // 非尾块, 每次loop的原始行数
        .src_dim_sizes = {x1_sec_axis_size * last_axis_size, x2_sec_axis_size * last_axis_size},
        .src_strides = {concat_tiling.max_orig_row_num * x1_stride, concat_tiling.max_orig_row_num * x2_stride},
        .src_buffer_offsets = {0, concat_tiling.max_repeat_times * (concat_tiling.src_dim_sizes[0] / concat_tiling.gcd) * kEltNumPerBlock},
        // strides[index] * ConcatContextType::kDataTypeSize / kDataBlockSize),
        .gather_mask_repeat_strides = {static_cast<uint16_t>(x1_stride * sizeof(T) / 32), static_cast<uint16_t>(x2_stride * sizeof(T) / 32)},
        .gather_mask_dim_sizes = {last_axis_size, last_axis_size}
    };

    for (size_t i = 0; i < padded.size(); ++i) {
      if (!padded[i]) {
        concat_tiling.gather_mask_repeat_strides[i] = 0;
      }
    }

    concat_tiling.first_copy_repeat_times =
        concat_tiling.max_repeat_times * kAddrListSize / kScaleToB16 / concat_tiling.gcd;
    concat_tiling.last_trans_repeat_times =
        concat_tiling.max_repeat_times * (concat_tiling.dst_dim_size / concat_tiling.gcd);
    concat_tiling.per_repeat_size = (concat_tiling.dst_dim_size / concat_tiling.gcd) * kEltNumPerBlock;

    ConcatInputList<T, ConcatContextType::kInputNum> input_list {
        .src_tensor_base_addrs = {(T *)l_x1.GetPhyAddr(), (T *)l_x2.GetPhyAddr()},
        .src_tensors = {&l_x1, &l_x2}
    };
    ConcatContextType context;
    context.total_row_num = first_axis_size;
    context.input_list = &input_list;
    ConcatExtendV2<ConcatContextType>(context, concat_tiling, l_y, l_tmp);

    UbToGm(y, l_y, dst);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(kernel, 1, input1_shape[0], input1_shape[1], input2_shape[1], input2_shape[2], x1, x2, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      for (int k = 0; k < input2_shape[2]; k++) {
        auto index = i * output_stride_0 + j * output_stride_1 + k;
        auto diff = (double) (y[index] - expect[index]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
          if (index < 32) {
            std::cout << index << ", " << y[index] << ", " << expect[index] << std::endl;
          }
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  std::vector<T *> to_free{x1, x2, y, expect};
  for (auto addr : to_free) {
    GmFree(addr);
  }
}

TEST(TestApiConcat, Test_concat_small_tail_not_padded) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 4}, {25, 100}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 3}, {25, 100}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 7}, {25, 200}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend<ConcatContextDiffDim<float, 2>>(input1_shape[idx],
                                                                 input2_shape[idx],
                                                                 output_shape[idx],
                                                                 false);
  }
}

TEST(TestApiConcat, Test_concat_small_tail_not_padded_s16) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 4}, {25, 100}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 3}, {25, 100}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 7}, {25, 200}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend<ConcatContextDiffDim<int16_t, 2>>(input1_shape[idx],
                                                                   input2_shape[idx],
                                                                   output_shape[idx],
                                                                   false);
  }
}

TEST(TestApiConcat, Test_concat_small_tail_padded) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 4}, {380, 2}, {192, 4}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 3}, {380, 5}, {192, 8}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 7}, {380, 7}, {192, 12}};
  for (size_t idx = 0; idx < input1_shape.size(); ++idx) {
    TestConcatSmallTailExtend<ConcatContextDiffDimPadded<int32_t, 2>>(input1_shape[idx],
                                                                         input2_shape[idx],
                                                                         output_shape[idx],
                                                                         true);
  }
}

TEST(TestApiConcat, Test_concat_small_tail_padded_s16) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 4}, {380, 2}, {192, 4}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 3}, {380, 5}, {192, 8}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 7}, {380, 7}, {192, 12}};
  for (size_t idx = 0; idx < input1_shape.size(); ++idx) {
    TestConcatSmallTailExtend<ConcatContextDiffDimPadded<int16_t, 2>>(input1_shape[idx],
                                                                      input2_shape[idx],
                                                                      output_shape[idx],
                                                                      true);
  }
}

TEST(TestApiConcat, Test_concat_same_tail_not_padded) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 1}, {4097, 1}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 1}, {4097, 1}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 2}, {4097, 2}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend<ConcatContextSameDim<int32_t, 2, 1>>(input1_shape[idx],
                                                                   input2_shape[idx],
                                                                   output_shape[idx],
                                                                   false);
  }
}

TEST(TestApiConcat, Test_concat_same_tail_padded) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 1}, {511, 1}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 1}, {511, 1}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 2}, {511, 2}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend<ConcatContextSameDimPadded<int32_t, 2, 1>>(input1_shape[idx],
                                                                         input2_shape[idx],
                                                                         output_shape[idx],
                                                                         true);
  }
}

TEST(TestApiConcat, Test_concat_same_tail_not_padded_s16) {
//  std::vector<std::vector<uint32_t>> input1_shape = {{2047, 1}, {1023, 2}, {257, 4}, {257, 8}, {255, 16}};
//  std::vector<std::vector<uint32_t>> input2_shape = {{2047, 1}, {1023, 2}, {257, 4}, {257, 8}, {255, 16}};
//  std::vector<std::vector<uint32_t>> output_shape = {{2047, 2}, {1023, 4}, {257, 8}, {257, 16}, {255, 32}};
  std::vector<std::vector<uint32_t>> input1_shape = {{127, 1}, {128, 1}, {129, 1}, {2047, 1}};
  std::vector<std::vector<uint32_t>> input2_shape = {{127, 1}, {128, 1}, {129, 1}, {2047, 1}};
  std::vector<std::vector<uint32_t>> output_shape = {{127, 2}, {128, 2}, {129, 2}, {2047, 2}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend<ConcatContextSameDim<int16_t, 2, 1>>(input1_shape[idx],
                                                                   input2_shape[idx],
                                                                   output_shape[idx],
                                                                   false);
  }
}

TEST(TestApiConcat, Test_concat_same_tail_not_padded_s16_3inputs) {
//  std::vector<std::vector<uint32_t>> input1_shape = {{128, 1}, {256, 2}, {4095, 2}, {257, 4}, {257, 8}, {255, 16}};
//  std::vector<std::vector<uint32_t>> input2_shape = {{128, 1}, {256, 2}, {4095, 2}, {257, 4}, {257, 8}, {255, 16}};
//  std::vector<std::vector<uint32_t>> input3_shape = {{128, 1}, {256, 2}, {4095, 2}, {257, 4}, {257, 8}, {255, 16}};
//  std::vector<std::vector<uint32_t>> output_shape = {{128, 3}, {256, 6}, {4095, 6}, {257, 12}, {257, 24}, {255, 48}};
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 1}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 1}};
  std::vector<std::vector<uint32_t>> input3_shape = {{128, 1}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 3}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend3Inputs<ConcatContextSameDim<int16_t, 3, 1>>(input1_shape[idx],
                                                                          input2_shape[idx],
                                                                          input3_shape[idx],
                                                                          output_shape[idx],
                                                                          false);
  }
}

TEST(TestApiConcat, Test_concat_same_tail_not_padded_s32_3inputs) {
//  std::vector<std::vector<uint32_t>> input1_shape = {{128, 1}, {256, 2}, {2049, 2}, {257, 4}, {257, 8}};
//  std::vector<std::vector<uint32_t>> input2_shape = {{128, 1}, {256, 2}, {2049, 2}, {257, 4}, {257, 8}};
//  std::vector<std::vector<uint32_t>> input3_shape = {{128, 1}, {256, 2}, {2049, 2}, {257, 4}, {257, 8}};
//  std::vector<std::vector<uint32_t>> output_shape = {{128, 3}, {256, 6}, {2049, 6}, {257, 12}, {257, 24}};
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 1}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 1}};
  std::vector<std::vector<uint32_t>> input3_shape = {{128, 1}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 3}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend3Inputs<ConcatContextSameDim<int32_t, 3, 1>>(input1_shape[idx],
                                                                          input2_shape[idx],
                                                                          input3_shape[idx],
                                                                          output_shape[idx],
                                                                          false);
  }
}

TEST(TestApiConcat, Test_concat_same_tail_padded_s16) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 1}, {513, 1}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 1}, {513, 1}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 2}, {513, 2}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend<ConcatContextSameDimPadded<int16_t, 2, 1>>(input1_shape[idx],
                                                                         input2_shape[idx],
                                                                         output_shape[idx],
                                                                         true);
  }
}

TEST(TestApiConcat, Test_concat_same_2nd_last_dim_not_padded_s16) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 5, 1}, {63, 5, 4}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 5, 1}, {63, 8, 4}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 10, 1}, {63, 13, 4}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend2ndLastDim<ConcatContextDiffDim<int16_t, 2>>(input1_shape[idx],
                                                                          input2_shape[idx],
                                                                          output_shape[idx],
                                                                          {false, false});
  }
}

TEST(TestApiConcat, Test_concat_same_2nd_last_dim_padded_s16) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 5, 1}, {63, 5, 4}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 5, 1}, {63, 8, 4}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 10, 1}, {63, 13, 4}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend2ndLastDim<ConcatContextDiffDimPadded<int16_t, 2>>(input1_shape[idx],
                                                                                input2_shape[idx],
                                                                                output_shape[idx],
                                                                                {true, true});
  }
}

TEST(TestApiConcat, Test_concat_same_2nd_last_dim_partial_padded_s16) {
  std::vector<std::vector<uint32_t>> input1_shape = {{128, 5, 1}, {63, 5, 4}};
  std::vector<std::vector<uint32_t>> input2_shape = {{128, 5, 1}, {63, 8, 4}};
  std::vector<std::vector<uint32_t>> output_shape = {{128, 10, 1}, {63, 13, 4}};
  for (auto idx = 0; idx < input1_shape.size(); idx++) {
    TestConcatSmallTailExtend2ndLastDim<ConcatContextDiffDimPadded<int16_t, 2>>(input1_shape[idx],
                                                                                input2_shape[idx],
                                                                                output_shape[idx],
                                                                                {false, true});
  }
}