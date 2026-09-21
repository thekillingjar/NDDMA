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
 #include "AddLayerNorm_tiling_data.h"
 using namespace optiling;
 
 void PrintResult(graph_normalTilingData& tilingData) {
   std::cout << "====================================================" << std::endl;
   auto tiling_key = tilingData.get_graph0_tiling_key();
   std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
   std::cout << "====================================================" << std::endl;
 }
 
 int main() {
   graph_normalTilingData tilingData;
   tilingData.set_block_dim(64);
   tilingData.set_ub_size(245760);
   auto &schedule0_g0_tiling_data = tilingData.graph0_result0_g0_tiling_data;
   auto &schedule0_g1_tiling_data = tilingData.graph0_result0_g1_tiling_data;
   auto &schedule1_g0_tiling_data = tilingData.graph0_result1_g0_tiling_data;
   schedule0_g0_tiling_data.set_A(1536);
   schedule0_g0_tiling_data.set_R(128);
   schedule0_g0_tiling_data.set_BL(8);
   schedule0_g1_tiling_data.set_A(1536);
   schedule0_g1_tiling_data.set_R(128);
   schedule1_g0_tiling_data.set_A(1536);
   schedule1_g0_tiling_data.set_R(128);
 
   if (GetTiling(tilingData)) {
     PrintResult(tilingData);
   } else {
     std::cout << "addlayernorm tiling func execute failed." << std::endl;
     return -1;
   }
 
   return 0;
 }