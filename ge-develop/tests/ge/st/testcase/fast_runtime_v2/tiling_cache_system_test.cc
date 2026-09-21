/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "faker/fake_value.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_scheduler_config.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data.h"
#include "common/bg_test.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "op_impl/data_flow_op_impl.h"
#include "lowering/model_converter.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "faker/magic_ops.h"
#include "check/executor_statistician.h"
#include "graph/utils/graph_dump_utils.h"
#include "kernel/tiling_cache.h"

namespace gert {
class TilingCacheSt : public testing::Test {
 public:
  void SetUp() override {}

  void TearDown() override {
    tiling_func_call_times.store(0UL);
  }

 public:
  static std::atomic_size_t tiling_func_call_times;
};
std::atomic_size_t TilingCacheSt::tiling_func_call_times(0UL);

namespace {
const std::string TransDataStubName = "TransDataStubBin";
const std::string DynamicAtomicStubName = "DynamicAtomicBin";
const std::string DynamicRnnv3StubName = "DynamicRNNV3StubBin";

UINT32 StubTilingFuncSuccEmpty(TilingContext *context) {
  (void)context;
  TilingCacheSt::tiling_func_call_times.fetch_add(1);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeFindTilingFunc(KernelContext *context) {
  auto tiling_fun_ptr = context->GetOutputPointer<OpImplKernelRegistry::TilingKernelFunc>(0);
  GE_ASSERT_NOTNULL(tiling_fun_ptr);
  *tiling_fun_ptr = StubTilingFuncSuccEmpty;
  return ge::GRAPH_SUCCESS;
}
}  // namespace

TEST_F(TilingCacheSt, MultiThreadExecutor_Ok_EnableTilingCache) {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("TransData",
                      AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName).WithHandle())
          .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName))
          .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  // runtime_stub.GetSlogStub().SetLevelInfo();
  runtime_stub.GetKernelStub().SetUp("FindTilingFunc", FakeFindTilingFunc);

  TaskProducerFactory::GetInstance().SetProducerType(TaskProducerType::KERNEL);
  ASSERT_EQ(TaskProducerFactory::GetInstance().GetProducerType(), TaskProducerType::KERNEL);
  auto option = MultiThreadExecutorOption(kLeastThreadNumber);
  auto model_executor = ModelV2Executor::Create(exe_graph, option, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048 * 2);
  memset_s(const_cast<void *>(mem_block->GetAddr()), mem_block->GetSize(), 0, 2048 * 2);

  auto multi_thread_ed = reinterpret_cast<const MultiThreadExecutionData *>(
      model_executor->GetExeGraphExecutor(SubExeGraphType::kMainExeGraph)->GetExecutionData());
  multi_thread_ed->scheduler->DumpBrief();
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
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
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto ess = StartExecutorStatistician(model_executor);
  // 第一次执行，无缓存，全部算子调用tiling_func
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(),
                                    3, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 13UL);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("TransData", "CacheableTiling"), 6);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("DynamicRNNV3", "CacheableTiling"), 1);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("DynamicAtomicAddrClean", "Tiling"), 6);
  // 第二次执行，走缓存，只有6个Atomic算子调用tiling_func
  ess->Clear();
  TilingCacheSt::tiling_func_call_times.store(0UL);
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(),
                                    3, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 6UL);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.CheckLaunchWhenStubTiling());
  multi_thread_ed->scheduler->Dump();
  runtime_stub.Clear();
  rtStreamDestroy(stream);
  mem_block->Free();
}

TEST_F(TilingCacheSt, PriorityTopologicalExecute_Ok_EnableTilingCache) {
  auto compute_graph = ShareGraph::IfGraph4();
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();
  GE_DUMP(compute_graph, "computegraph_IfGraph4");

  auto ge_root_model =
      GeModelBuilder(compute_graph).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "exe_graph_IfGraph4");

  GertRuntimeStub runtime_stub;
  // runtime_stub.GetSlogStub().SetLevelInfo();
  runtime_stub.GetKernelStub().SetUp("FindTilingFunc", FakeFindTilingFunc);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({0}).Build();
  auto input_holder1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
  auto input_holder2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
  auto input_holder3 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
  std::vector<Tensor *> inputs{pred_holder.GetTensor(), input_holder1.GetTensor(), input_holder2.GetTensor(),
                               input_holder3.GetTensor()};
  auto output_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  // 第一次执行, 全部走TilingFunc
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 3UL);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "CacheableTiling"), 3);

  // 第二次执行, 所有节点使用缓存，TilingFunc调用次数为0
  ess->Clear();
  TilingCacheSt::tiling_func_call_times.store(0UL);
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 0UL);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  runtime_stub.Clear();
  rtStreamDestroy(stream);
}

TEST_F(TilingCacheSt, PriorityTopologicalExecute_Ok_SameStorageShapeMissCache) {
  auto compute_graph = ShareGraph::IfGraph4();
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();
  GE_DUMP(compute_graph, "computegraph_IfGraph4");

  auto ge_root_model =
      GeModelBuilder(compute_graph).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "exe_graph_IfGraph4");

  GertRuntimeStub runtime_stub;
  // runtime_stub.GetSlogStub().SetLevelInfo();
  runtime_stub.GetKernelStub().SetUp("FindTilingFunc", FakeFindTilingFunc);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({0}).Build();
  auto input_holder1 = TensorFaker()
                           .Placement(kOnDeviceHbm)
                           .DataType(ge::DT_FLOAT)
                           .Shape({8, 3, 224, 224})
                           .StorageShape({8, 3, 1, 224, 224})
                           .Build();
  auto input_holder2 = TensorFaker()
                           .Placement(kOnDeviceHbm)
                           .DataType(ge::DT_FLOAT)
                           .Shape({8, 3, 224, 224})
                           .StorageShape({8, 3, 1, 224, 224})
                           .Build();
  auto input_holder3 = TensorFaker()
                           .Placement(kOnDeviceHbm)
                           .DataType(ge::DT_FLOAT)
                           .Shape({8, 3, 224, 224})
                           .StorageShape({8, 3, 1, 224, 224})
                           .Build();
  std::vector<Tensor *> inputs{pred_holder.GetTensor(), input_holder1.GetTensor(), input_holder2.GetTensor(),
                               input_holder3.GetTensor()};
  auto output_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};

  auto input_holder4 = TensorFaker()
                           .Placement(kOnDeviceHbm)
                           .DataType(ge::DT_FLOAT)
                           .Shape({8, 1, 224, 224})
                           .StorageShape({8, 3, 1, 224, 224})
                           .Build();
  auto input_holder5 = TensorFaker()
                           .Placement(kOnDeviceHbm)
                           .DataType(ge::DT_FLOAT)
                           .Shape({8, 1, 224, 224})
                           .StorageShape({8, 3, 1, 224, 224})
                           .Build();
  auto input_holder6 = TensorFaker()
                           .Placement(kOnDeviceHbm)
                           .DataType(ge::DT_FLOAT)
                           .Shape({8, 1, 224, 224})
                           .StorageShape({8, 3, 1, 224, 224})
                           .Build();
  std::vector<Tensor *> inputs2{pred_holder.GetTensor(), input_holder4.GetTensor(), input_holder5.GetTensor(),
                                input_holder6.GetTensor()};
  auto output_holder2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 1, 224, 224}).Build();
  std::vector<Tensor *> outputs2{output_holder2.GetTensor()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  // 第一次执行, 全部走TilingFunc
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 3UL);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "CacheableTiling"), 3);

  // 第二次执行, 未命中缓存, 全部走TilingFunc
  ess->Clear();
  TilingCacheSt::tiling_func_call_times.store(0UL);
  ASSERT_EQ(
      model_executor->Execute({stream_value.value}, inputs2.data(), inputs2.size(), outputs2.data(), outputs2.size()),
      ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 3UL);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  runtime_stub.Clear();
  rtStreamDestroy(stream);
}

TEST_F(TilingCacheSt, DataDependentNodes_Ok_EnableTilingCache) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  auto graph = ShareGraph::AicoreWithRtsDebugOverflowGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder.AddTaskDef("ReduceSum", AiCoreTaskDefFaker("ReduceSumStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);

  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EReduceSumGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("FindTilingFunc", FakeFindTilingFunc);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto input0 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT16).Format(ge::FORMAT_ND).Shape({}).Build();
  auto input1 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Format(ge::FORMAT_ND).Shape({0}).Build();
  auto input2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Format(ge::FORMAT_ND).Shape({8}).Build();
  auto input3 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Format(ge::FORMAT_ND).Shape({8}).Build();
  auto inputs = std::vector<Tensor *>({input0.GetTensor(), input1.GetTensor(), input2.GetTensor(), input3.GetTensor()});
  auto outputs = FakeTensors({}, 4);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  // 第一次执行, 全部走TilingFunc
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 1UL);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("ReduceSum", "CacheableTiling"), 1);

  // 第二次执行, 所有节点使用缓存，TilingFunc调用次数为0
  ess->Clear();
  TilingCacheSt::tiling_func_call_times.store(0UL);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 0UL);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  runtime_stub.Clear();
  rtStreamDestroy(stream);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(TilingCacheSt, DataDependentNodes_Ok_EnableTilingCacheFail) {
  auto graph = ShareGraph::AicoreWithRtsDebugOverflowGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder.AddTaskDef("ReduceSum", AiCoreTaskDefFaker("ReduceSumStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);

  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EReduceSumGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("FindTilingFunc", FakeFindTilingFunc);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto input0 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT16).Format(ge::FORMAT_ND).Shape({8192}).Build();
  auto input1 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Format(ge::FORMAT_ND).Shape({8192}).Build();
  auto input2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Format(ge::FORMAT_ND).Shape({8192}).Build();
  auto input3 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Format(ge::FORMAT_ND).Shape({8192}).Build();
  auto inputs = std::vector<Tensor *>({input0.GetTensor(), input1.GetTensor(), input2.GetTensor(), input3.GetTensor()});
  auto outputs = FakeTensors({}, 4);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  // 第一次执行, 全部走TilingFunc
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 1UL);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("ReduceSum", "CacheableTiling"), 1);

  // 第二次执行, 由于长度超过，没有缓存上，调用tiling func次数仍然为1
  ess->Clear();
  TilingCacheSt::tiling_func_call_times.store(0UL);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(TilingCacheSt::tiling_func_call_times, 1UL);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  runtime_stub.Clear();
  rtStreamDestroy(stream);
}
}