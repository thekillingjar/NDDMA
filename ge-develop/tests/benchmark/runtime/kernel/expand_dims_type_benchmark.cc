/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <benchmark/benchmark.h>
#include "exe_graph/runtime/expand_dims_type.h"

namespace gert {
static void Expand1Dim(benchmark::State &state) {
  auto shape = Shape{2,2,256};
  Shape out_shape;
  ExpandDimsType edt("1");
  for (auto _ : state) {
    benchmark::DoNotOptimize(edt.Expand(shape, out_shape));
  }
}
BENCHMARK(Expand1Dim);

static void Expand1DimAtLast(benchmark::State &state) {
  auto shape = Shape{2,2,256};
  Shape out_shape;
  ExpandDimsType edt("0001");
  for (auto _ : state) {
    benchmark::DoNotOptimize(edt.Expand(shape, out_shape));
  }
}
BENCHMARK(Expand1DimAtLast);

static void Expand2Dim(benchmark::State &state) {
  auto shape = Shape{2,256};
  Shape out_shape;
  ExpandDimsType edt("11");
  for (auto _ : state) {
    benchmark::DoNotOptimize(edt.Expand(shape, out_shape));
  }
}
BENCHMARK(Expand2Dim);

static void Expand3Dim(benchmark::State &state) {
  auto shape = Shape{2};
  Shape out_shape;
  ExpandDimsType edt("111");
  for (auto _ : state) {
    benchmark::DoNotOptimize(edt.Expand(shape, out_shape));
  }
}
BENCHMARK(Expand3Dim);

static void Expand3Dim2(benchmark::State &state) {
  auto shape = Shape{2};
  Shape out_shape;
  ExpandDimsType edt("1110");
  for (auto _ : state) {
    benchmark::DoNotOptimize(edt.Expand(shape, out_shape));
  }
}
BENCHMARK(Expand3Dim2);

static void Expand3DimAtLast(benchmark::State &state) {
  auto shape = Shape{2};
  Shape out_shape;
  ExpandDimsType edt("0111");
  for (auto _ : state) {
    benchmark::DoNotOptimize(edt.Expand(shape, out_shape));
  }
}
BENCHMARK(Expand3DimAtLast);

/* todo
static void FractalNzTransformerPerformance(benchmark::State &state) {
  int64_t  dim_num = 0;
  auto dst_shape = Shape();
  auto shape = Shape{2,2,256};
  for (auto _ : state) {
    shape.Expend(Padding(0b00000001, 4));
    FractalNzTransformer().TransShapeTo(shape, 16, dst_shape);
  }
}


BENCHMARK(FractalNzTransformerPerformance);
 */
}
