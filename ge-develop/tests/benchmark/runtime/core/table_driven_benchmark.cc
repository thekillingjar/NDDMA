/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <benchmark/benchmark.h>
#include "common/table_driven.h"
#include "graph/types.h"
namespace gert {
int32_t Get(ge::Format key1, ge::Format key2) {
  static auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, int32_t>(-1)
                          .Add(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, 1)
                          .Add(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, 10)
                          .Add(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 11)
                          .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN, 100);

  return table.Find(key1, key2);
}

static auto g_table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, int32_t>(-1)
                        .Add(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, 1)
                        .Add(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, 10)
                        .Add(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, 11)
                        .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN, 100);
int32_t Get2(ge::Format key1, ge::Format key2) {
  return g_table.Find(key1, key2);
}

static void TableDriven2Find(benchmark::State &state) {
  Get(ge::FORMAT_ND, ge::FORMAT_ND);
  for (auto _ : state) {
    benchmark::DoNotOptimize(Get(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ));
  }
}
BENCHMARK(TableDriven2Find);
BENCHMARK(TableDriven2Find);

static void TableDriven2Find2(benchmark::State &state) {
  Get(ge::FORMAT_ND, ge::FORMAT_ND);
  for (auto _ : state) {
    benchmark::DoNotOptimize(Get2(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ));
  }
}
BENCHMARK(TableDriven2Find2);


static void TableDriven2Find2Inline(benchmark::State &state) {
  Get(ge::FORMAT_ND, ge::FORMAT_ND);
  for (auto _ : state) {
    benchmark::DoNotOptimize(g_table.Find(ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ));
  }
}
BENCHMARK(TableDriven2Find2Inline);

}  // namespace gert
