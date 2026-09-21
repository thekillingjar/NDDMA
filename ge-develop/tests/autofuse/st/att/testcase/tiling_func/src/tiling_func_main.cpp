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
#include "kernel_context_holder_builder.h"
#include "stub_info.h"
#include "struct_info.h"
using namespace optiling;
using namespace att;
int case_index = 0;
uint64_t RESERVED_WORKSPACE_SIZE_910B = 16 * 1024 * 1024U;
bool RunCase(std::vector<std::vector<int64_t>> shape, uint64_t expect_tiling_key) {
  case_index++;
  KernelContextHolderBuilder builder;
  builder.AddInput(InOutput(ge::GeShape(shape[0]), ge::FORMAT_ND, ge::DT_FLOAT16)) // x1
         .AddInput(InOutput(ge::GeShape(shape[1]), ge::FORMAT_ND, ge::DT_FLOAT16)) // x2
         .AddInput(InOutput(ge::GeShape(shape[3]), ge::FORMAT_ND, ge::DT_FLOAT16)) // gamma
         .AddInput(InOutput(ge::GeShape(shape[4]), ge::FORMAT_ND, ge::DT_FLOAT16)) // beta
         .AddInput(InOutput(ge::GeShape(shape[2]), ge::FORMAT_ND, ge::DT_FLOAT16)); // bias
  std::vector<int64_t> output_shape = shape[0];
  output_shape[shape[0].size() - 1] = 1;
  auto tiling_context_holder = builder
                                   .AddOutput(InOutput(ge::GeShape(shape[0]), ge::FORMAT_ND, ge::DT_FLOAT16))    // y
                                   .AddOutput(InOutput(ge::GeShape(output_shape), ge::FORMAT_ND, ge::DT_FLOAT))  // mean
                                   .AddOutput(InOutput(ge::GeShape(output_shape), ge::FORMAT_ND, ge::DT_FLOAT))  // rtsd
                                   .AddOutput(InOutput(ge::GeShape(shape[0]), ge::FORMAT_ND, ge::DT_FLOAT16))    // x
                                   .SetTilingData(10240)
                                   .SetWorkSpace(1600)
                                   .SetCompileInfo(2)
                                   .SetPlatformInfo()
                                   .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
                                   .AddPrivateAtt({"additional_output", ge::AnyValue::CreateFrom<bool>(true)})
                                   .Build();
  gert::TilingContext *tiling_context = reinterpret_cast<gert::TilingContext *>(tiling_context_holder.context_);
  const auto status = GetTiling(tiling_context);
  if ((status) == ge::GRAPH_SUCCESS) {
    GET_TILING_DATA(tmpTiling, tiling_context->GetRawTilingData()->GetData());
    PrintTilingData(tmpTiling);
    if (tmpTiling.A == 0 && tmpTiling.R == 0) {
      std::cout << "Original axis is 0." << std::endl;
      return false;
    }
    if (tiling_context->GetTilingKey() == expect_tiling_key) {
      if (tiling_context->GetTilingKey() != 1151 ||
          tiling_context->GetWorkspaceSizes(1)[0] >= RESERVED_WORKSPACE_SIZE_910B + 4 * shape[0][0] * shape[0][1]) {
        std::cout << "Case " << case_index << " tiling func execute success." << std::endl;
        return true;
      }
    }
  }
  std::cout << "Case " << case_index << " tiling func execute failed." << std::endl;
  return false;
}

int main() {
  bool ret = true;
  ret &= RunCase({{1536, 128}, {1536, 128}, {1536, 128}, {128}, {128}}, 1101);  // 1
  std::cout << "case 1 -- " << ret << std::endl;
  ret &= RunCase({{1000, 1024}, {1000, 1024}, {1000, 1024}, {1024}, {1024}}, 1101);  // 2
  std::cout << "case 2 -- " << ret << std::endl;
  ret &= RunCase({{1000, 2048}, {1000, 2048}, {1000, 2048}, {2048}, {2048}}, 1101);  // 3
  std::cout << "case 3 -- " << ret << std::endl;
  ret &= RunCase({{1000, 4096}, {1000, 4096}, {1000, 4096}, {4096}, {4096}}, 1101);  // 4
  std::cout << "case 4 -- " << ret << std::endl;
  ret &= RunCase({{1000, 8192}, {1000, 8192}, {1000, 8192}, {8192}, {8192}}, 1101);  // 5
  std::cout << "case 5 -- " << ret << std::endl;
  ret &= RunCase({{1000, 16384}, {1000, 16384}, {1000, 16384}, {16384}, {16384}}, 1111);  // 6
  std::cout << "case 6 -- " << ret << std::endl;
  ret &= RunCase({{1000, 32768}, {1000, 32768}, {1000, 32768}, {32768}, {32768}}, 1151);  // 7
  std::cout << "case 7 -- " << ret << std::endl;
  ret &= RunCase({{200, 65536}, {200, 65536}, {200, 65536}, {65536}, {65536}}, 1151);  // 8
  std::cout << "case 8 -- " << ret << std::endl;
  ret &= RunCase({{200, 131072}, {200, 131072}, {200, 131072}, {131072}, {131072}}, 1151);  // 9
  std::cout << "case 9 -- " << ret << std::endl;
  ret &= RunCase({{80, 262144}, {80, 262144}, {80, 262144}, {262144}, {262144}}, 1151);  // 10
  std::cout << "case 10 -- " << ret << std::endl;
  ret &= RunCase({{40, 524288}, {40, 524288}, {40, 524288}, {524288}, {524288}}, 1151);  // 11
  std::cout << "case 11 -- " << ret << std::endl;
  ret &= RunCase({{40, 1048576}, {40, 1048576}, {40, 1048576}, {1048576}, {1048576}}, 1151);  // 12
  std::cout << "case 12 -- " << ret << std::endl;
  ret &= RunCase({{960, 1024}, {960, 1024}, {960, 1024}, {1024}, {1024}}, 1101);  // 13
  std::cout << "case 13 -- " << ret << std::endl;
  ret &= RunCase({{1000, 23456}, {1000, 23456}, {1000, 23456}, {23456}, {23456}}, 1111);  // 14
  std::cout << "case 14 -- " << ret << std::endl;
  ret &= RunCase({{10000, 61}, {10000, 61}, {10000, 61}, {61}, {61}}, 1101);  // 15
  std::cout << "case 15 -- " << ret << std::endl;
  ret &= RunCase({{500, 11000}, {500, 11000}, {500, 11000}, {11000}, {11000}}, 1101);  // 16
  std::cout << "case 16 -- " << ret << std::endl;
  ret &= RunCase({{8, 1234567}, {8, 1234567}, {8, 1234567}, {1234567}, {1234567}}, 1151);  // 17
  std::cout << "case 17 -- " << ret << std::endl;
  ret &= RunCase({{4567, 1567}, {4567, 1567}, {4567, 1567}, {1567}, {1567}}, 1101);  // 18
  std::cout << "case 18 -- " << ret << std::endl;
  ret &= RunCase({{4567, 2345}, {4567, 2345}, {4567, 2345}, {2345}, {2345}}, 1101);  // 19
  std::cout << "case 19 -- " << ret << std::endl;
  ret &= RunCase({{4321, 4567}, {4321, 4567}, {4321, 4567}, {4567}, {4567}}, 1101);  // 20
  std::cout << "case 20 -- " << ret << std::endl;
  ret &= RunCase({{64, 24, 128}, {64, 24, 128}, {64, 24, 128}, {128}, {128}}, 1101);  // 21
  std::cout << "case 21 -- " << ret << std::endl;
  ret &= RunCase({{1000, 1024}, {1000, 1024}, {1024}, {1024}, {1024}}, 1102);                 // 22
  std::cout << "case 22 -- " << ret << std::endl;
  ret &= RunCase({{1000, 16384}, {1000, 16384}, {16384}, {16384}, {16384}}, 1112);            // 23
  std::cout << "case 23 -- " << ret << std::endl;
  ret &= RunCase({{200, 65536}, {200, 65536}, {65536}, {65536}, {65536}}, 1152);              // 24
  std::cout << "case 24 -- " << ret << std::endl;
  return ret ? 0 : -1;
}