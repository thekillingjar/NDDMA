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
#include "OpTest2_tiling_data.h"
using namespace optiling;

template<typename T>
inline T Ceiling(T a)
{
    T value = static_cast<T>(static_cast<int64_t>(a));
    return (value == a) ? value : (value + 1);
}

int TestCase() {
  CeilingTilingData tilingData;
  tilingData.set_s1_size(1024);
  tilingData.set_s2_size(8192);
  tilingData.set_block_dim(48);
  tilingData.set_ub_size(184*1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, 0);
  if ((status)) {
    uint64_t s2t_size = tilingData.get_s2t_size();
    uint64_t s1s2Tb_size = tilingData.get_s1s2Tb_size();
    std::cout << "s2t"<< " = " << s2t_size << std::endl;
    std::cout << "s1s2Tb"<< " = " << s1s2Tb_size << std::endl;
    if (tilingData.get_block_dim() > 48) {
      std::cout << "Ceiling tiling func execute failed." << std::endl;
      return -1;
    }
    std::cout << "Ceiling tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Ceiling tiling func execute failed." << std::endl;
  return -1;
}

int main() {
  auto ret = TestCase();
  if (ret == 0) {
    return 0;
  }
  return -1;
}