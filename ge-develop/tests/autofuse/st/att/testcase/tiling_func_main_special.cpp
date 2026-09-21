/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "OpTest5_tiling_data.h"
using namespace optiling;
int main() {
  NpuKernel0TilingData tilingData;
  tilingData.set_s0(1024);
  tilingData.set_s1(65536);
  tilingData.set_block_dim(48);
  tilingData.set_ub_size(180 * 1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, -1);
  if ((status)) {
    std::cout << "z0t"<< " = " << tilingData.get_z0t_size() << std::endl;
    std::cout << "z1t"<< " = " << tilingData.get_z1t_size() << std::endl;
    std::cout << "z0Tz1Tb_size"<< " = " << tilingData.get_z0Tz1Tb_size() << std::endl;
    std::cout << "z0b_size"<< " = " << tilingData.get_z0b_size() << std::endl;
    std::cout << "block_dim"<< " = " << tilingData.get_block_dim() << std::endl;
    std::cout << "tiling_key"<< " = " << tilingData.get_tiling_key() << std::endl;
    std::cout << "tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Case0 tiling func execute failed." << std::endl;
  return -1;
}
