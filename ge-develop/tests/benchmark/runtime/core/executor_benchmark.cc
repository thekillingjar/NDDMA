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
#include "faker/fake_value.h"
#include "common/share_graph.h"
#include "runtime/dev.h"
#include "stub/gert_runtime_stub.h"

#include "graph/types.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tensor.h"

#include "graph/utils/graph_utils.h"
#include "lowering/graph_converter.h"
#include "faker/global_data_faker.h"
#include "runtime/model_v2_executor.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_scheduler_config.h"
#include "lowering/model_converter.h"
#include "kernel/memory/caching_mem_allocator.h"

namespace gert {
namespace {
ge::graphStatus DefaultKernelFunc(KernelContext *run_context) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateOutput(const ge::Node *node, KernelContext *run_context) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus freeOutput(const ge::Node *node, KernelContext *run_context) {
  return ge::GRAPH_SUCCESS;
}
struct FakeKernelRegistry : KernelRegistry {
  const KernelFuncs *FindKernelFuncs(const std::string &kernel_type) const override {
    static KernelFuncs funcs = {DefaultKernelFunc, CreateOutput, freeOutput};
    return &funcs;
  }
} kernel_registry;

const std::string ReduceSumStubName = "ReduceSumStubBin";
const char *const TransDataStubName = "TransDataStubBin";
const char *const TransData13StubName = "TransData17StubBin";
const char *const DynamicAtomicStubName = "DynamicAtomicBin";
const char *const DynamicRnnv3StubName = "DynamicRNNV3StubBin";
const char *const AddStubName = "AddStubBin";
const char *const MulStubName = "MulStubBin";

ge::ComputeGraphPtr GenerateLstmpExeGraph() {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  GE_DUMP(graph, "LstmpST_ComputeGraph");
  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("TransData",
                      AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName).WithHandle())
          .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName))
          .BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  GE_ASSERT_NOTNULL(exe_graph);
  GE_DUMP(exe_graph, "LstmpST_ExecuteGraph1");
  return exe_graph;
}
}  // namespace
static void ExecutorRunForLstmpExeGraph(benchmark::State &state) {
  auto exe_graph = GenerateLstmpExeGraph();
  ge::GraphUtils::DumpGEGraphToOnnx(*exe_graph, "E2ELstmpUT");

  KernelRegistry::ReplaceKernelRegistry(std::make_shared<FakeKernelRegistry>());
  auto model_executor = ModelV2Executor::Create(exe_graph);
  model_executor->Load();
  auto outputs = FakeTensors({2}, 3);

  auto i0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto i1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto i2 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto i3 = FakeValue<uint64_t>(0);

  model_executor->Execute({i3.value}, std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(),
                          3, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size());
  for (auto _ : state) {
    model_executor->Execute({i3.value},
                            std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(), 3,
                            reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size());
  }
  model_executor->UnLoad();
  KernelRegistry::ReplaceKernelRegistry(nullptr);
}

BENCHMARK(ExecutorRunForLstmpExeGraph);

static void ExecutorWithKernelRunForLstmpExeGraph(benchmark::State &state) {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("TransData",
                      AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName).WithHandle())
          .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName))
          .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  auto model_executor = ModelV2Executor::Create(exe_graph);
  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048 * 2);
  memset_s(const_cast<void *>(mem_block->GetAddr()), mem_block->GetSize(), 0, 2048 * 2);

  model_executor->Load();
  auto outputs = FakeTensors({2048}, 3, const_cast<void *>(mem_block->GetAddr()));

  auto i0 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnHost,
                                     ge::DT_FLOAT16,
                                     const_cast<void *>(mem_block->GetAddr())});
  auto i1 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnHost,
                                     ge::DT_FLOAT16,
                                     const_cast<void *>(mem_block->GetAddr())});
  auto i2 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnHost,
                                     ge::DT_FLOAT16,
                                     const_cast<void *>(mem_block->GetAddr())});

  rtStream_t stream;
  rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT));
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  model_executor->Execute({i3.value}, std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(),
                          3, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size());
  for (auto _ : state) {
    model_executor->Execute({i3.value},
                            std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(), 3,
                            reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size());
  }
  model_executor->UnLoad();
  rtStreamDestroy(stream);
  mem_block->Free();
}

//BENCHMARK(ExecutorWithKernelRunForLstmpExeGraph)->Iterations(1);
BENCHMARK(ExecutorWithKernelRunForLstmpExeGraph);

static void ParallelExecutorWithKernelRunForLstmpExeGraph(benchmark::State &state) {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("TransData",
                      AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName).WithHandle())
          .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName))
          .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  auto option = MultiThreadExecutorOption(kLeastThreadNumber);
  auto model_executor = ModelV2Executor::Create(exe_graph, option);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048 * 2);
  memset_s(const_cast<void *>(mem_block->GetAddr()), mem_block->GetSize(), 0, 2048 * 2);
  model_executor->Load();
  auto outputs = FakeTensors({2048}, 3, const_cast<void *>(mem_block->GetAddr()));

  auto i0 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnHost,
                                     ge::DT_FLOAT16,
                                     const_cast<void *>(mem_block->GetAddr())});
  auto i1 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnHost,
                                     ge::DT_FLOAT16,
                                     const_cast<void *>(mem_block->GetAddr())});
  auto i2 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnHost,
                                     ge::DT_FLOAT16,
                                     const_cast<void *>(mem_block->GetAddr())});

  rtStream_t stream;
  rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT));
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  model_executor->Execute({i3.value},
                          std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(),
                          3, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size());
  for (auto _ : state) {
    model_executor->Execute({i3.value},
                            std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(), 3,
                            reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size());
  }
  model_executor->UnLoad();
  rtStreamDestroy(stream);
  mem_block->Free();
}

//BENCHMARK(ParallelExecutorWithKernelRunForLstmpExeGraph)->Iterations(2);
BENCHMARK(ParallelExecutorWithKernelRunForLstmpExeGraph);

/*
static void ExecutorWithKernelRunForLstmpExeGraph(benchmark::State &state) {
  auto exe_graph = GenerateLstmpExeGraph();
  ge::GraphUtils::DumpGEGraphToOnnx(*exe_graph, "E2ELstmpWithKernel");

  auto model_executor = ModelV2Executor::Create(exe_graph);
  model_executor->Load();

  void* device_block = (void*)0x01;
  auto outputs = FakeTensors({2}, 3, device_block);

  auto i0 = FakeValue<Tensor>(Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16,
device_block}); auto i1 = FakeValue<Tensor>(Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm,
ge::DT_FLOAT16, device_block}); auto i2 = FakeValue<Tensor>(Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}},
kOnDeviceHbm, ge::DT_FLOAT16, device_block}); auto i3 = FakeValue<uint64_t>(1);

  model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(),
3, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()); for (auto _ : state) {
    model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(),
3, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size());
  }
  model_executor->UnLoad();
}

BENCHMARK(ExecutorWithKernelRunForLstmpExeGraph);
 */
}  // namespace gert

BENCHMARK_MAIN();
