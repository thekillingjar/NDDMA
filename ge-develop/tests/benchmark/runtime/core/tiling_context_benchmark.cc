/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/tiling_context.h"
#include <benchmark/benchmark.h>
#include "faker/kernel_run_context_facker.h"

namespace gert {
namespace {
struct TestData {
  int64_t a;
  int32_t b;
  int16_t c;
  int16_t d;
};
}
static void TilingContext_GetTilingData(benchmark::State &state) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(context->GetTilingData<TestData>());
  }
}
BENCHMARK(TilingContext_GetTilingData);

static void TilingContext_GetRawTilingData(benchmark::State &state) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .TilingData(param.get())
                    .Build();

  auto context = holder.GetContext<TilingContext>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(context->GetRawTilingData());
  }
}
BENCHMARK(TilingContext_GetRawTilingData);
}
