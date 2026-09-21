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
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_type.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_producer_config.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_package.h"
#include "core/executor_error_code.h"
#include "fake_execution_data.h"
#include "faker/global_data_faker.h"
#include "lowering/graph_converter.h"
#include "runtime/model_v2_executor.h"
#include "faker/fake_value.h"
#include "common/model_v2_executor_test_helper.h"
#include "exe_graph/runtime/tiling_context.h"
#include "stub/gert_runtime_stub.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "common/share_graph.h"
#include "faker/magic_ops.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "check/executor_statistician.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_task_producer.h"

using namespace gert;
namespace {
size_t kMaxProduceCount = 3U;
std::mutex g_fake_mtx;
uint32_t g_fake_flag = 0U;
ge::graphStatus CalcTensorSizeFromStorageFailedFake(gert::KernelContext *context) {
  std::unique_lock<std::mutex> lk(g_fake_mtx);
  if (g_fake_flag++ == 0U) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
}
class KernelTaskProducerUnitTest : public testing::Test {
  void SetUp() override {
    constexpr int32_t kEnvNoOverwrite = 0;
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_MAX_RUNTIME_CORE_NUMBER, "3", kEnvNoOverwrite, mmRet);
    (void) mmRet;
    KernelSpy::GetInstance().Clear();
  }
  void TearDown() override {
    Test::TearDown();
    int32_t mmRet = 0;
    MM_SYS_UNSET_ENV(MM_ENV_MAX_RUNTIME_CORE_NUMBER, mmRet);
    (void) mmRet;
  }
};

TEST_F(KernelTaskProducerUnitTest, should_run_all_kernel_for_two_nodes) {
  FakeExecutionData executionData(10);
  executionData.Chain({0, 1}).StartNodes({0});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::KERNEL, 10, kMaxProduceCount});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 2);
  TaskRun(package);
  KERNEL_RUN_EXPECT(0, 1);
  producer->Recycle(package);
  producer->Recycle(package);
  delete producer;
}

TEST_F(KernelTaskProducerUnitTest, should_run_all_kernel_for_two_nodes_failed) {
  FakeExecutionData executionData(10);
  executionData.Chain({0, 1}).StartNodes({0});
  executionData.FuncFailed(1, kStatusFailed);
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::KERNEL, 10, kMaxProduceCount});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 2);
  TaskRun(package);
  ASSERT_NE(producer->Recycle(package), ge::SUCCESS);
  producer->Dump();
  delete producer;
}

TEST_F(KernelTaskProducerUnitTest, run_two_node_end_of_sequence) {
  FakeExecutionData executionData(10);
  executionData.Chain({0, 1, 2}).StartNodes({0});
  executionData.FuncEndOfSequence(1, ge::END_OF_SEQUENCE);
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::KERNEL, 10, kMaxProduceCount});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 3);
  auto task = package.back();
  bool force_quit = true;
  task->SetForceQuit(&force_quit);
  TaskRun(package);
  ASSERT_EQ(producer->Recycle(package), ge::END_OF_SEQUENCE);
  producer->Dump();
  delete producer;
}

TEST_F(KernelTaskProducerUnitTest, should_run_all_kernel_for_two_chains) {
  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 9}).StartNodes({3, 5});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::KERNEL, 10, kMaxProduceCount});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), kMaxProduceCount);
  TaskRun(package);
  KERNEL_RUN_EXPECT(3, 7, 6);
  producer->Recycle(package);

  package = producer->Produce();
  ASSERT_EQ(package.size(), kMaxProduceCount);
  TaskRun(package);
  KERNEL_RUN_EXPECT(5, 8, 9);
  producer->Recycle(package);

  package = producer->Produce();
  ASSERT_EQ(package.size(), 0);
  delete producer;
}

TEST_F(KernelTaskProducerUnitTest, should_run_all_kernel_but_one_kernel_failed_for_three_chains) {
  FakeExecutionData executionData(10);
  executionData.Chain({1, 4, 7}).Chain({2, 5, 8}).Chain({3, 6, 9}).StartNodes({1, 2, 3});
  executionData.FuncFailed(4, kStatusFailed);
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::KERNEL, 10, kMaxProduceCount});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), kMaxProduceCount);
  TaskRun(package);
  ASSERT_NE(producer->Recycle(package), ge::GRAPH_SUCCESS);
  delete producer;
}

TEST_F(KernelTaskProducerUnitTest, kernel_while_graph_success) {
  GertRuntimeStub stub;
  MagicOpFakerBuilder("LessThan_5")
      .OutputPlacement(kOnHost)
      .KernelFunc([](KernelContext *ctx) {
        auto input = ctx->GetInputPointer<GertTensorData>(kFkiStart + 1)->GetAddr();
        auto output = ctx->GetOutputPointer<GertTensorData>(1)->GetAddr();
        *static_cast<bool *>(output) = (*static_cast<int32_t *>(input) < 5);
        return ge::GRAPH_SUCCESS;
      })
      .Build(stub);

  MagicOpFakerBuilder("Add_1")
      .KernelFunc([](KernelContext *ctx) {
        auto input = ctx->GetInputPointer<GertTensorData>(kFkiStart + 1);
        auto tensor_data = ctx->GetOutputPointer<GertTensorData>(1);
        *static_cast<int32_t *>(tensor_data->GetAddr()) = *static_cast<int32_t *>(input->GetAddr()) + 1;
        tensor_data->SetPlacement(input->GetPlacement());
        tensor_data->SetSize(input->GetSize());
        return ge::GRAPH_SUCCESS;
      })
      .Build(stub);

  for (auto &graph : {ShareGraph::WhileGraph2()}) {
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.BuildGeRootModel();

    bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    auto option = MultiThreadExecutorOption(3U);
    auto model_executor = ModelV2Executor::Create(exe_graph, option, ge_root_model);
    ASSERT_NE(model_executor, nullptr);

    int32_t output = 0;
    ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({}, 1, &output);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i1 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    auto inputs = FakeTensors({}, 1);
    *static_cast<int32_t *>(inputs.data()[0].GetAddr()) = 0;

    ASSERT_EQ(model_executor->Execute({i1.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    // Check output value
    EXPECT_EQ(*static_cast<int32_t *>(static_cast<gert::Tensor *>(outputs.GetAddrList()[0])->GetAddr()), 5);

    *static_cast<int32_t *>(inputs.data()[0].GetAddr()) = 8;
    ASSERT_EQ(model_executor->Execute({i1.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(*static_cast<int32_t *>(static_cast<gert::Tensor *>(outputs.GetAddrList()[0])->GetAddr()), 8);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
}

TEST_F(KernelTaskProducerUnitTest, kerne_Lstmp_graph_success) {
  const std::string TransDataStubName = "TransDataStubBin";
  const std::string DynamicAtomicStubName = "DynamicAtomicBin";
  const std::string DynamicRnnv3StubName = "DynamicRNNV3StubBin";
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data =
      GlobalDataFaker(root_model)
          .FakeWithHandleAiCore("Add", false)
          .AddTaskDef("TransData",
                      AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName).WithHandle())
          .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName))
          .Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(graph, global_data);

  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto option = MultiThreadExecutorOption(3U);
  auto model_executor = ModelV2Executor::Create(exe_graph, option, root_model);
  ASSERT_NE(model_executor, nullptr);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048 * 2);
  memset_s(const_cast<void *>(mem_block->GetAddr()), mem_block->GetSize(), 0, 2048 * 2);

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
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(),
                                    3, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->Execute({i3.value},
                                    std::vector<Tensor *>({i0.holder.get(), i1.holder.get(), i2.holder.get()}).data(),
                                    3, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.CheckLaunchWhenStubTiling());
  rtStreamDestroy(stream);
  mem_block->Free();
}

TEST_F(KernelTaskProducerUnitTest, kerne_fail_then_success_graph_success) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto graph = ShareGraph::BuildTwoAddNodeGraph();
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("CalcTensorSizeFromStorage", CalcTensorSizeFromStorageFailedFake);

  auto option = MultiThreadExecutorOption(3U);
  auto model_executor = ModelV2Executor::Create(exe_graph, option, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  ess->Clear();

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({1, 2, 3, 4}, 2);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Load({i3.value}), ge::GRAPH_SUCCESS);
  ASSERT_NE(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  model_executor->GetSubscribers().Clear();
  g_fake_flag = 0U;
  ASSERT_NE(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);


  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ge::RuntimeStub::Reset();
}

TEST_F(KernelTaskProducerUnitTest, all_kernel_in_start_task) {
  FakeExecutionData executionData(10, true);
  executionData.Chain({1, 4, 7}).Chain({2, 5, 8}).Chain({3, 6, 9}).StartNodes({1, 2, 3});
  auto producer = reinterpret_cast<KernelTaskProducer *>(
      TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::KERNEL, 10, kMaxProduceCount}));
  producer->Prepare(executionData.Data());
  ASSERT_EQ(producer->start_tasks_.size(), 9);
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), kMaxProduceCount);
  TaskRun(package);
  KERNEL_RUN_EXPECT(1, 2, 3);
  producer->Recycle(package);

  package = producer->Produce();
  ASSERT_EQ(package.size(), kMaxProduceCount);
  TaskRun(package);
  KERNEL_RUN_EXPECT(4, 5, 6);
  producer->Recycle(package);

  package = producer->Produce();
  ASSERT_EQ(package.size(), kMaxProduceCount);
  TaskRun(package);
  KERNEL_RUN_EXPECT(7, 8, 9);
  producer->Recycle(package);

  package = producer->Produce();
  ASSERT_EQ(package.size(), 0);
  delete producer;
}

TEST_F(KernelTaskProducerUnitTest, SwitchNotify_in_start_task) {
  FakeExecutionData executionData(10, true);
  executionData
      .KernelAttr({{1, {"conv2d", "AllocMemHbm"}},
                   {2, {"conv2d", "Tiling"}},
                   {3, {"conv2d", "Launch"}},
                   {4, {"conv2d", "CalcSize"}},
                   {5, {"if", "SwitchNotify"}},
                   {6, {"transdata", "InferShape"}},
                   {7, {"transdata", "Tiling"}},
                   {8, {"transdata", "Launch"}},
                   {9, {"Netoutput", "Output"}}})
      .Chain({1, 2, 3, 4, 5, 6, 7, 8 ,9})
      .StartNodes({1});
  auto producer = reinterpret_cast<KernelTaskProducer *>(
      TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::KERNEL, 10, kMaxProduceCount}));
  producer->Prepare(executionData.Data());
  ASSERT_EQ(producer->start_tasks_.size(), 5);
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), kMaxProduceCount);
  TaskRun(package);
  KERNEL_RUN_EXPECT(1, 2, 3);
  producer->Recycle(package);

  package = producer->Produce();
  ASSERT_EQ(package.size(), 2);
  TaskRun(package);
  KERNEL_RUN_EXPECT(4, 5);
  producer->Recycle(package);

  package = producer->Produce();
  ASSERT_EQ(package.size(), kMaxProduceCount);
  TaskRun(package);
  KERNEL_RUN_EXPECT(6, 7, 8);
  producer->Recycle(package);

  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(9);
  producer->Recycle(package);
  delete producer;
}