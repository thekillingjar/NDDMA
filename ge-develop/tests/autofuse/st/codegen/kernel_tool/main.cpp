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
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
// kernel入口函数声明，需要结合实际情况进行修改
extern "C" __global__ __aicore__ void autofuse_pointwise_0__abs__add(GM_ADDR Add_out0_graph_Data_0, GM_ADDR Add_out0_graph_Data_1, GM_ADDR Add_out0_graph_Output_0, GM_ADDR workspace, GM_ADDR gm_tiling_data);

int main() {
  std::vector<int> test_shape{128, 7}; // 输入数据shape，需要结合实际情况进行修改
  int test_size = 1;
  for (auto s : test_shape) {
    test_size *= s;
  }
  // 申请输入输出数据内存
  float *x1 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *x2 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

   // 准备输入数据值和真值数据，需要结合实际情况进行修改
  srand(1);
  for (int i = 0; i < test_size; i++) {
    x1[i] = rand() / static_cast<double>(RAND_MAX);
    x2[i] = rand() / static_cast<double>(RAND_MAX);
    expect[i] = std::fabs(x1[i]) + x2[i];
  }
  // 设置tiling data的值，需要结合实际情况进行修改(具体值如何设置参考ATT定位手册)
  AutofuseTilingData tiling_data;
  tiling_data.block_dim = 1; // 在plog日志中grep -rwn "TilingSummary" | grep "The value of"进行查看值
  tiling_data.ub_size = 20512; // 在plog日志中grep -rwn "TilingSummary" | grep "The value of"进行查看值
  tiling_data.tiling_key = 0; // 在plog日志中grep "tiling_func" -nr | grep "Among the templates, tiling case"进行查看值
  tiling_data.s2 = 128; // 在plog日志中grep -rn "input params"进行查看值
  tiling_data.s3 = 7; // 在plog日志中grep -rn "input params"进行查看值
  tiling_data.z0z1t_size = 440; // 在plog日志中grep -rwn "TilingSummary" | grep "The value of"进行查看值
  tiling_data.z0z1Tb_size = 3; // 在plog日志中grep -rwn "TilingSummary" | grep "The value of"进行查看值
  tiling_data.q0_size = 1760; // 在plog日志中grep -rwn "TilingSummary" | grep "The value of"进行查看值
  tiling_data.q1_size = 1760; // 在plog日志中grep -rwn "TilingSummary" | grep "The value of"进行查看值
  tiling_data.q2_size = 1760; // 在plog日志中grep -rwn "TilingSummary" | grep "The value of"进行查看值
  tiling_data.b0_size = 1760; // 在plog日志中grep -rwn "TilingSummary" | grep "The value of"进行查看值

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  // 调用kernel，需要结合实际情况进行修改
  ICPU_RUN_KF(autofuse_pointwise_0__abs__add, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)x2, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // 对比计算结果与真值
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (std::fabs(y[i] - expect[i]) > 1e-6) {
      diff_count++;
    }
  }
  // 释放内存
  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
  if (diff_count == 0) {
    std::cout << "test kernel pass" << std::endl;
    return 0;
  } else {
    std::cout << "test kernel failed, test_size: " << test_size << ", diff_count: "
    << diff_count << std::endl;
    return -1;
  }
}
