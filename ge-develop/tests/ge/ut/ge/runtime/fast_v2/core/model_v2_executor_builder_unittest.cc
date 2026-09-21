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
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "lowering/graph_converter.h"
#include "graph/utils/graph_utils.h"
#include "runtime/model_v2_executor.h"
#include "faker/fake_value.h"
#include "common/bg_test.h"
#include "common/model_v2_executor_test_helper.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "exe_graph/runtime/tiling_context.h"
#include "stub/gert_runtime_stub.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "graph/utils/graph_dump_utils.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "graph/debug/ge_attr_define.h"

namespace gert {
class ModelV2ExecutorBuilderUT : public bg::BgTest {
 public:
  static void CheckSingleNodeGraphModelDesc(ModelV2Executor *model_executor) {
    auto &model_desc = model_executor->GetModelDesc();
    size_t count;
    auto descs = model_desc.GetAllInputsDesc(count);
    ASSERT_EQ(count, 2);
    ASSERT_NE(descs, nullptr);
    EXPECT_STREQ(descs[0].GetName(), "data1");
    EXPECT_EQ(descs[0].GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(descs[0].GetOriginShape(), Shape({-1, 2, 3, 4}));
    EXPECT_EQ(descs[0].GetOriginFormat(), ge::FORMAT_ND);
    EXPECT_EQ(descs[0].GetOriginShapeRange().GetMin(), Shape({1, 2, 3, 4}));
    EXPECT_EQ(descs[0].GetOriginShapeRange().GetMax(), Shape({100, 2, 3, 4}));

    EXPECT_STREQ(descs[1].GetName(), "data2");
    EXPECT_EQ(descs[1].GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(descs[1].GetOriginShape(), Shape({1, -1, 3, 4}));
    EXPECT_EQ(descs[1].GetOriginFormat(), ge::FORMAT_ND);
    EXPECT_EQ(descs[1].GetOriginShapeRange().GetMin(), Shape({1, 1, 3, 4}));
    EXPECT_EQ(descs[1].GetOriginShapeRange().GetMax(), Shape({1, 100, 3, 4}));

    descs = model_desc.GetAllOutputsDesc(count);
    ASSERT_EQ(count, 1);
    ASSERT_NE(descs, nullptr);
    EXPECT_EQ(descs[0].GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(descs[0].GetOriginShape(), Shape({-1, -1, 3, 4}));
    EXPECT_EQ(descs[0].GetOriginFormat(), ge::FORMAT_ND);
    EXPECT_EQ(descs[0].GetOriginShapeRange().GetMin(), Shape({1, 1, 3, 4}));
    EXPECT_EQ(descs[0].GetOriginShapeRange().GetMax(), Shape({100, 100, 3, 4}));
  }
};
TEST_F(ModelV2ExecutorBuilderUT, BuildFromSingleNodeGraph) {
  auto compute_graph =
      ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1, -1, 3, 4}, {-1, -1, 3, 4}},
                                       {{1, 2, 3, 4}, {1, 1, 3, 4}, {1, 1, 3, 4}, {1, 1, 3, 4}},
                                       {{100, 2, 3, 4}, {1, 100, 3, 4}, {100, 100, 3, 4}, {100, 100, 3, 4}});
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  CheckSingleNodeGraphModelDesc(model_executor.get());
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);

  auto input0 = FakeValue<Tensor>(
      Tensor{{{100, 2, 3, 4}, {100, 2, 3, 4}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 = FakeValue<Tensor>(
      Tensor{{{1, 100, 3, 4}, {1, 100, 3, 4}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->ExecuteSync(std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(), 2,
                                        reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(ModelV2ExecutorBuilderUT, RefsHasTheSameAddr) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  auto execution_data = ModelV2ExecutorTestHelper::GetExecutionData(model_executor.get(), kMainExeGraph);
  ASSERT_NE(execution_data, nullptr);

  auto alloc_nodes = ModelV2ExecutorTestHelper::GetNodesByKernelType(model_executor.get(), "AllocLaunchArg");
  ASSERT_EQ(alloc_nodes.size(), 1);
  auto alloc_node = alloc_nodes[0];

  auto tiling_nodes = ModelV2ExecutorTestHelper::GetNodesByKernelType(execution_data, "CacheableTiling");
  ASSERT_EQ(tiling_nodes.size(), 1);
  auto tiling_node = tiling_nodes[0];

  auto launch_nodes = ModelV2ExecutorTestHelper::GetNodesByKernelType(execution_data, "LaunchKernelWithHandle");
  ASSERT_EQ(launch_nodes.size(), 1);

  // zero copy tiling-data
  EXPECT_NE(ModelV2ExecutorTestHelper::GetOutChain(tiling_node, TilingContext::kOutputTilingData), nullptr);
  EXPECT_EQ(
      ModelV2ExecutorTestHelper::GetOutChain(tiling_node, TilingContext::kOutputTilingData)->GetValue<void *>(),
      ModelV2ExecutorTestHelper::GetOutChain(alloc_node, static_cast<size_t>(AllocLaunchArgOutputs::kTilingDataBase))
          ->GetValue<void *>());

  // zero copy args
  EXPECT_NE(
      ModelV2ExecutorTestHelper::GetOutChain(tiling_node, static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)),
      nullptr);
  EXPECT_EQ(
      ModelV2ExecutorTestHelper::GetOutChain(tiling_node, static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg))
          ->GetValue<void *>(),
      ModelV2ExecutorTestHelper::GetOutChain(alloc_node, static_cast<size_t>(AllocLaunchArgOutputs::kRtArg))
          ->GetValue<void *>());
}

TEST_F(ModelV2ExecutorBuilderUT, BuildFromFrameworkOpSingleNodeGraph) {
  auto compute_graph = ShareGraph::FrameworkOPGraph("Add");
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(ModelV2ExecutorBuilderUT, BuildVarialbeGraph) {
  auto compute_graph = ShareGraph::VariableOPGraph("Add");
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  RtSession session;
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model, &session);
  ASSERT_NE(model_executor, nullptr);
  ModelExecuteArg executor_args;
  ModelLoadArg load_args(&session, {});
  ASSERT_EQ(model_executor->Load(executor_args, load_args), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(ModelV2ExecutorBuilderUT, BuildVarialbeGraph_Failed_SessionIdNotEqual) {
  auto compute_graph = ShareGraph::VariableOPGraph("Add");
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  RtSession session_tmp;
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model, &session_tmp);
  ASSERT_NE(model_executor, nullptr);
  RtSession session(10010);
  ModelExecuteArg executor_args;
  ModelLoadArg load_args(&session, {});
  ASSERT_NE(model_executor->Load(executor_args, load_args), ge::GRAPH_SUCCESS);
}

TEST_F(ModelV2ExecutorBuilderUT, BuildVarialbeGraph_ModelV2ExecutorCreateWithRtSession) {
  auto compute_graph = ShareGraph::VariableOPGraph("Add");
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  RtSession session;
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model, &session);
  ASSERT_NE(model_executor, nullptr);
  ModelExecuteArg executor_args;
  ModelLoadArg load_args(&session, {});
  ASSERT_EQ(model_executor->Load(executor_args, load_args), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}
TEST_F(ModelV2ExecutorBuilderUT, MultiThreadExecutor_NewThreadNum_lessthan_LeastThreadNum_build_failed) {
  auto compute_graph =
      ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1, -1, 3, 4}, {-1, -1, 3, 4}},
                                       {{1, 2, 3, 4}, {1, 1, 3, 4}, {1, 1, 3, 4}, {1, 1, 3, 4}},
                                       {{100, 2, 3, 4}, {1, 100, 3, 4}, {100, 100, 3, 4}, {100, 100, 3, 4}});
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  MultiThreadExecutorOption option(1U);
  auto model_executor = ModelV2Executor::Create(exe_graph, option, root_model);
  ASSERT_EQ(model_executor, nullptr);
}

/**
 * 场景: CheckIoReuseAddrs - 空配置时直接返回成功
 *
 * 说明：
 * - io_same_addr_pairs_ 为空
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 GRAPH_SUCCESS
 */
TEST_F(ModelV2ExecutorBuilderUT, CheckIoReuseAddrs_EmptyPairs_Success) {
  dlog_setlevel(0, 0, 0);
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  // io_same_addr_pairs_ 为空，校验应通过
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}},
        {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                       kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto output0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}},
        {ge::FORMAT_ND, ge::FORMAT_ND, {}},
        kOnDeviceHbm, ge::DT_FLOAT16, 0});

  Tensor *inputs[] = {input0.holder.get()};
  Tensor *outputs[] = {output0.holder.get()};

  EXPECT_EQ(ModelV2ExecutorTestHelper::CheckIoReuseAddrs(model_executor.get(), inputs, 1, outputs, 1),
            ge::GRAPH_SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 地址相同时校验通过
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {0, 0}
 * - input[0] 和 output[0] 地址相同
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 GRAPH_SUCCESS
 */
TEST_F(ModelV2ExecutorBuilderUT, CheckIoReuseAddrs_SameAddress_Success) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  dlog_setlevel(0, 0, 0);
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();

  // 设置属性到模型
  for (auto &iter : root_model->GetSubgraphInstanceNameToModel()) {
    ge::AttrUtils::SetStr(iter.second, ge::ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");
    break;
  }

  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);


  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  // 使用相同地址
  auto input0 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                         {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                         kOnDeviceHbm,
                                         ge::DT_FLOAT16,
                                         reinterpret_cast<void *>(0x1000)});
  auto output0 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                          {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                          kOnDeviceHbm,
                                          ge::DT_FLOAT16,
                                          reinterpret_cast<void *>(0x1000)});

  Tensor *inputs[] = {input0.holder.get()};
  Tensor *outputs[] = {output0.holder.get()};

  EXPECT_EQ(ModelV2ExecutorTestHelper::CheckIoReuseAddrs(model_executor.get(), inputs, 1, outputs, 1),
            ge::GRAPH_SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 地址不同时校验失败
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {0, 0}
 * - input[0] 和 output[0] 地址不同
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 PARAM_INVALID
 */
TEST_F(ModelV2ExecutorBuilderUT, CheckIoReuseAddrs_DifferentAddress_Fail) {
  dlog_setlevel(0, 0, 0);
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();

  // 设置属性到模型
  for (auto &iter : root_model->GetSubgraphInstanceNameToModel()) {
    ge::AttrUtils::SetStr(iter.second, ge::ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");
    break;
  }

  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  // 使用不同地址
  auto input0 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                         {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                         kOnDeviceHbm,
                                         ge::DT_FLOAT16,
                                         reinterpret_cast<void *>(0x1000)});
  auto output0 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                          {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                          kOnDeviceHbm,
                                          ge::DT_FLOAT16,
                                          reinterpret_cast<void *>(0x2000)});

  Tensor *inputs[] = {input0.holder.get()};
  Tensor *outputs[] = {output0.holder.get()};

  EXPECT_EQ(ModelV2ExecutorTestHelper::CheckIoReuseAddrs(model_executor.get(), inputs, 1, outputs, 1),
            ge::PARAM_INVALID);
}

/**
 * 场景: CheckIoReuseAddrs - 输出索引越界
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {0, 5}，但只有 1 个输出
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 PARAM_INVALID
 */
TEST_F(ModelV2ExecutorBuilderUT, CheckIoReuseAddrs_OutputIndexOutOfRange_Fail) {
  dlog_setlevel(0, 0, 0);
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();

  // 设置越界的输出索引
  for (auto &iter : root_model->GetSubgraphInstanceNameToModel()) {
    ge::AttrUtils::SetStr(iter.second, ge::ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,5");
    break;
  }

  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  auto input0 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                         {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                         kOnDeviceHbm,
                                         ge::DT_FLOAT16,
                                         reinterpret_cast<void *>(0x1000)});
  auto output0 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                          {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                          kOnDeviceHbm,
                                          ge::DT_FLOAT16, reinterpret_cast<void *>(0x1000)});

  Tensor *inputs[] = {input0.holder.get()};
  Tensor *outputs[] = {output0.holder.get()};  // 只有1个输出，但配置索引为5

  EXPECT_EQ(ModelV2ExecutorTestHelper::CheckIoReuseAddrs(model_executor.get(), inputs, 1, outputs, 1),
            ge::PARAM_INVALID);
}

/**
 * 场景: CheckIoReuseAddrs - 多对索引全部匹配
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {0, 0}, {1, 1}
 * - 所有配对的输入输出地址都相同
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 GRAPH_SUCCESS
 */
TEST_F(ModelV2ExecutorBuilderUT, CheckIoReuseAddrs_MultiplePairs_AllMatch_Success) {
  dlog_setlevel(0, 0, 0);
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();

  // 设置多对索引
  for (auto &iter : root_model->GetSubgraphInstanceNameToModel()) {
    ge::AttrUtils::SetStr(iter.second, ge::ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0|1,1");
    break;
  }

  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  auto input0 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                         {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                         kOnDeviceHbm,
                                         ge::DT_FLOAT16,
                                         reinterpret_cast<void *>(0x1000)});
  auto input1 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                         {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                         kOnDeviceHbm,
                                         ge::DT_FLOAT16,
                                         reinterpret_cast<void *>(0x2000)});
  auto output0 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                          {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                          kOnDeviceHbm,
                                          ge::DT_FLOAT16,
                                          reinterpret_cast<void *>(0x1000)});
  auto output1 = FakeValue<Tensor>(Tensor{{{256}, {256}},
                                          {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                          kOnDeviceHbm,
                                          ge::DT_FLOAT16,
                                          reinterpret_cast<void *>(0x2000)});

  Tensor *inputs[] = {input0.holder.get(), input1.holder.get()};
  Tensor *outputs[] = {output0.holder.get(), output1.holder.get()};

  EXPECT_EQ(ModelV2ExecutorTestHelper::CheckIoReuseAddrs(model_executor.get(), inputs, 2, outputs, 2),
            ge::GRAPH_SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 空输入/输出
 *
 * 说明：
 * - inputs 或 outputs 为 nullptr 或数量为 0
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 GRAPH_SUCCESS（跳过校验）
 */
TEST_F(ModelV2ExecutorBuilderUT, CheckIoReuseAddrs_NullInputsOutputs_Success) {
  dlog_setlevel(0, 0, 0);
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();

  for (auto &iter : root_model->GetSubgraphInstanceNameToModel()) {
    ge::AttrUtils::SetStr(iter.second, ge::ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");
    break;
  }

  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  // 测试 nullptr 输入
  EXPECT_EQ(ModelV2ExecutorTestHelper::CheckIoReuseAddrs(model_executor.get(), nullptr, 0, nullptr, 0),
            ge::GRAPH_SUCCESS);

  // 测试数量为 0
  auto input0 = FakeValue<Tensor>(
      Tensor{{{256}, {256}}, {ge::FORMAT_ND,
        ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, reinterpret_cast<void *>(0x1000)});
  Tensor *inputs[] = {input0.holder.get()};

  EXPECT_EQ(ModelV2ExecutorTestHelper::CheckIoReuseAddrs(model_executor.get(), inputs, 0, nullptr, 0),
            ge::GRAPH_SUCCESS);
}

}  // namespace gert