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
#include <memory>
#include <iostream>
#include <cstring>
#include "core/executor.h"

#include "faker/kernel_run_context_facker.h"

#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
namespace {
struct TestStruct {
  uint64_t a;
  uint64_t b;
};
}
static void GetInput(benchmark::State &state) {
  auto context_hold = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20]);
  memset(context_hold.get(), 0xff, sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20);
  auto context = reinterpret_cast<KernelRunContext *>(context_hold.get());
  context->input_size = 10;
  context->output_size = 10;
  context->output_start = &(context->values[context->input_size]);
  auto cpp_context = reinterpret_cast<KernelContext *>(context);
  for (auto _ : state) {
    benchmark::DoNotOptimize(cpp_context->GetInput(5));
  }
}
BENCHMARK(GetInput);
static void GetInlineInputPointer(benchmark::State &state) {
  auto context_hold = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20]);
  memset(context_hold.get(), 0xff, sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20);
  auto context = reinterpret_cast<KernelRunContext *>(context_hold.get());
  context->input_size = 10;
  context->output_size = 10;
  context->output_start = &(context->values[context->input_size]);
  auto cpp_context = reinterpret_cast<KernelContext *>(context);
  for (auto _ : state) {
    benchmark::DoNotOptimize(cpp_context->GetInputPointer<uint64_t>(5));
  }
}
BENCHMARK(GetInlineInputPointer);
static void GetAllocInputPointer(benchmark::State &state) {
  auto context_hold = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20]);
  memset(context_hold.get(), 0xff, sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20);
  auto context = reinterpret_cast<KernelRunContext *>(context_hold.get());
  std::vector<AsyncAnyValue> values(20);
  for (size_t i = 0; i < 20; ++i) {
    context->values[i] = &(values[i]);
  }
  context->input_size = 10;
  context->output_size = 10;
  context->output_start = &(context->values[context->input_size]);
  auto cpp_context = reinterpret_cast<KernelContext *>(context);
  for (auto _ : state) {
    benchmark::DoNotOptimize(cpp_context->GetInputPointer<TestStruct>(5));
  }
}
BENCHMARK(GetAllocInputPointer);

static void GetOutput(benchmark::State &state) {
  auto context_hold = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20]);
  memset(context_hold.get(), 0xff, sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20);
  auto context = reinterpret_cast<KernelRunContext *>(context_hold.get());
  context->input_size = 10;
  context->output_size = 10;
  context->output_start = &(context->values[context->input_size]);
  auto cpp_context = reinterpret_cast<KernelContext *>(context);

  for (auto _ : state) {
    benchmark::DoNotOptimize(cpp_context->GetOutput(5));
  }
}
BENCHMARK(GetOutput);
static void GetInlineOutputPointer(benchmark::State &state) {
  auto context_hold = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20]);
  memset(context_hold.get(), 0xff, sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20);
  auto context = reinterpret_cast<KernelRunContext *>(context_hold.get());
  context->input_size = 10;
  context->output_size = 10;
  context->output_start = &(context->values[context->input_size]);
  auto cpp_context = reinterpret_cast<KernelContext *>(context);
  for (auto _ : state) {
    benchmark::DoNotOptimize(cpp_context->GetOutputPointer<uint64_t>(5));
  }
}
BENCHMARK(GetInlineOutputPointer);
static void GetAllocOutputPointer(benchmark::State &state) {
  auto context_hold = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20]);
  memset(context_hold.get(), 0xff, sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20);
  auto context = reinterpret_cast<KernelRunContext *>(context_hold.get());
  std::vector<AsyncAnyValue> values(20);
  for (size_t i = 0; i < 20; ++i) {
    context->values[i] = &(values[i]);
  }
  context->input_size = 10;
  context->output_size = 10;
  context->output_start = &(context->values[context->input_size]);
  auto cpp_context = reinterpret_cast<KernelContext *>(context);
  for (auto _ : state) {
    benchmark::DoNotOptimize(cpp_context->GetOutputPointer<TestStruct>(5));
  }
}
BENCHMARK(GetAllocOutputPointer);

static void GetOutput2(benchmark::State &state) {
  auto context_hold = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20]);
  memset(context_hold.get(), 0xff, sizeof(KernelRunContext) + sizeof(AsyncAnyValue) * 20);
  auto context = reinterpret_cast<KernelRunContext *>(context_hold.get());
  context->input_size = 10;
  context->output_size = 10;
  context->output_start = &(context->values[context->input_size]);
  auto cpp_context = reinterpret_cast<KernelContext *>(context);

  for (auto _ : state) {
    benchmark::DoNotOptimize(cpp_context->GetOutput2(5));
  }
}
BENCHMARK(GetOutput2);

static void GetExtendInfo(benchmark::State &state) {
  StorageShape i1;
  StorageShape i2;
  StorageShape o1;
  auto holder = InferShapeContextFaker()
                    .IrInputNum(2)
                    .InputShapes({&i1, &i2})
                    .OutputShapes({&o1})
                    .NodeIoNum(2, 1)
                    .NodeAttrs({{"int", ge::AnyValue::CreateFrom<int64_t>(10)},
                                {"str", ge::AnyValue::CreateFrom<std::string>("Hello!")},
                                {"bool", ge::AnyValue::CreateFrom<bool>(true)},
                                {"float", ge::AnyValue::CreateFrom<float>(10.101)}})
                    .Build();

  auto context = holder.GetContext<ExtendedKernelContext>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(context->GetExtendInfo());
  }
}
BENCHMARK(GetExtendInfo);

static void GetAttrsObj(benchmark::State &state) {
  StorageShape i1;
  StorageShape i2;
  StorageShape o1;
  auto holder = InferShapeContextFaker()
                    .IrInputNum(2)
                    .InputShapes({&i1, &i2})
                    .OutputShapes({&o1})
                    .NodeIoNum(2, 1)
                    .NodeAttrs({{"int", ge::AnyValue::CreateFrom<int64_t>(10)},
                                {"str", ge::AnyValue::CreateFrom<std::string>("Hello!")},
                                {"bool", ge::AnyValue::CreateFrom<bool>(true)},
                                {"float", ge::AnyValue::CreateFrom<float>(10.101)}})
                    .Build();

  auto context = holder.GetContext<ExtendedKernelContext>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(context->GetAttrs());
  }
}
BENCHMARK(GetAttrsObj);

static void GetAttrs(benchmark::State &state) {
  StorageShape i1;
  StorageShape i2;
  StorageShape o1;
  auto holder = InferShapeContextFaker()
                    .IrInputNum(2)
                    .InputShapes({&i1, &i2})
                    .OutputShapes({&o1})
                    .NodeIoNum(2, 1)
                    .NodeAttrs({{"int", ge::AnyValue::CreateFrom<int64_t>(10)},
                                {"str", ge::AnyValue::CreateFrom<std::string>("Hello!")},
                                {"bool", ge::AnyValue::CreateFrom<bool>(true)},
                                {"float", ge::AnyValue::CreateFrom<float>(10.101)}})
                    .Build();

  auto context = holder.GetContext<ExtendedKernelContext>();

  for (auto _ : state) {
    auto attrs = context->GetAttrs();
    benchmark::DoNotOptimize(attrs->GetAttrPointer<char>(1));
  }
}
BENCHMARK(GetAttrs);

}
