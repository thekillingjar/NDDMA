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
#include "exe_graph/runtime/shape.h"
namespace gert {
namespace kernel {
int x = 2;
int y = 5;
int z = 10;
int ret = 0;
static void test_shape_inline(benchmark::State &state) {
  auto shape = Shape{2, x, 5};
  auto shape3 = Shape{2, 3, z};
  auto shape2 = Shape{2, 4, 5};
  for (auto _ : state) {
    for (int i = 0; i < 4; i++) {
      shape3.SetDim(2, shape3.GetDim(2) * (i + 1));
    }
    ret = shape3.GetDim(2) + shape2.GetDim(2) + shape3.GetDim(1);
  }
}
BENCHMARK(test_shape_inline);

static void ShapeGetDimInlineNotOptimize(benchmark::State &state) {
  auto shape = Shape{2, 4, 5};
  for (auto _ : state) {
    benchmark::DoNotOptimize(shape.GetDim(0));
  }
}
BENCHMARK(ShapeGetDimInlineNotOptimize);



//static void test_shape_not_inline(benchmark::State &state) {
//  for (auto _ : state) {
//    auto shape = Shape{2, x, 5};
//    auto shape2 = Shape{2, 4, 5};
//    auto shape3 = Shape{2, 3, z};
//    for(int i = 0; i < 4 ;i ++){
//      shape3.SetDimNotInline(2, shape3.GetDimNotInline(2) * (i+1));
//    }
//    ret = shape3.GetDimNotInline(2) +  shape2.GetDimNotInline(2) + shape3.GetDimNotInline(1);
//  }
//}
//
//BENCHMARK(test_shape_not_inline);

static void ShapeGetShapesizePerformance(benchmark::State &state) {
  auto shape = Shape{2, 6, 10, 40};
  int64_t shape_size;
  for (auto _ : state) {
    shape_size = shape.GetShapeSize();
  }
}
BENCHMARK(ShapeGetShapesizePerformance);

}
}
