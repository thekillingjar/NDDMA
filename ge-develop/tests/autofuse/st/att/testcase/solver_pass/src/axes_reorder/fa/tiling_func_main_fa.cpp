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
#include "FA_tiling_data.h"
using namespace optiling;
int main() {
  TilingData tilingData;
  tilingData.set_block_dim(48);
  tilingData.set_ub_size(184 * 1024);
  tilingData.set_hbm_size(180 * 1024 * 1024);
  tilingData.set_B(4);
  tilingData.set_N(10);
  tilingData.set_G(1);
  tilingData.set_S1(1024);
  tilingData.set_S2(1024);
  tilingData.set_D(128);
  tilingData.set_s1t_size(0);
  tilingData.set_s1tt2_size(0);
  tilingData.set_s1tt_size(0);
  tilingData.set_s2t_size(0);

  const auto status = GetTiling(tilingData, 0u);
  if ((status)) {
    std::cout << "get_s1t_size"<< " = " << tilingData.get_s1t_size() << std::endl;
    std::cout << "get_s1tt2_size"<< " = " << tilingData.get_s1tt2_size() << std::endl;
    std::cout << "get_s1tt_size"<< " = " << tilingData.get_s1tt_size() << std::endl;
    std::cout << "get_s2t_size"<< " = " << tilingData.get_s2t_size() << std::endl;
    if (tilingData.get_s1tt2_size() == 0u) {
      return -1;
    }
    return 0;
  }
  std::cout << "fa tiling func execute failed." << std::endl;
  return -1;
}