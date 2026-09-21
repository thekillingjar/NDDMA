/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/tiling_data.h"
#include <benchmark/benchmark.h>

namespace gert {
namespace {
struct TestData {
  int64_t a;
  int32_t b;
  int16_t c;
  int16_t d;
};
}
static void TilingData_AppendBasicType(benchmark::State &state) {
  auto data = TilingData::CreateCap(2 * 1024);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());

  for (auto _ : state) {
    tiling_data->Append(10);
    tiling_data->SetDataSize(0);
  }
}
BENCHMARK(TilingData_AppendBasicType);

static void TilingData_AppendStruct(benchmark::State &state) {
  auto data = TilingData::CreateCap(2048);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  TestData td {
      .a = 1024,
      .b = 512,
      .c = 256,
      .d = 128
  };

  for (auto _ : state) {
    tiling_data->Append(td);
    tiling_data->SetDataSize(0);
  }
}
BENCHMARK(TilingData_AppendStruct);

}

BENCHMARK_MAIN();
