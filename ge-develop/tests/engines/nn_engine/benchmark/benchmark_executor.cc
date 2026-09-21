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
#include "common/op_tensor_utils.h"
#include "graph/node.h"
#include "graph/op_desc.h"

namespace fe{

void SomeFunction(const ge::OpDescPtr &op, bool use_attr) {
  OpTensorUtils::IsUnKnownShapeOp(*op, use_attr);
}

void CreateOp(ge::OpDescPtr &op) {
  op = std::make_shared<ge::OpDesc>("test", "Test");
  ge::GeShape shape(vector<int64_t>({1, 2, 3, 4}));
  ge::GeTensorDesc tensor(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  op->AddInputDesc(tensor);
  op->AddInputDesc(tensor);
  op->AddInputDesc(tensor);
  op->AddInputDesc(tensor);
  op->AddOutputDesc(tensor);
  op->AddOutputDesc(tensor);
}

static void TestDynamicShapeCheck(benchmark::State &state) {
  ge::OpDescPtr op;
  CreateOp(op);

  for (auto _ : state) {
    SomeFunction(op, state.range(0)); /* Use range to get arguments */
  }
}

static void TestDynamicShapeCheck2(benchmark::State &state) {
  ge::OpDescPtr op;
  CreateOp(op);

  for (auto _ : state) {
    state.PauseTiming();
    /* All running time of the following operations will not be counted.
     * But the state.PauseTiming and state.ResumeTiming will cost*/
    for (int i = 0; i < state.range(0); i++) {
      string a = "a";
    }
    state.ResumeTiming();
    SomeFunction(op, false); /* Use range to get arguments */
  }
}



BENCHMARK(TestDynamicShapeCheck)->Arg(true)->Arg(false);
BENCHMARK(TestDynamicShapeCheck2)->Arg(10)->Arg(20)->Arg(30)->Arg(100);
}

BENCHMARK_MAIN();
