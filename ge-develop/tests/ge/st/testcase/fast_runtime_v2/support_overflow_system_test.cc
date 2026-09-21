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
#include <iostream>
#include "faker/fake_value.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/share_graph.h"
#include "lowering/graph_converter.h"
#include "faker/global_data_faker.h"
#include "runtime/model_v2_executor.h"
#include "common/bg_test.h"
#include "runtime/dev.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "op_impl/less_important_op_impl.h"
#include "op_impl/transdata/trans_data_positive_source_tc_1010.h"
#include "op_impl/dynamicatomicaddrclean/dynamic_atomic_addr_clean_impl.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling/op_tiling_constants.h"

using namespace ge;
namespace gert {
class GraphExecutorWithOverflowKernelUnitTest : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
};
const std::string DynamicAtomicStubName = "DynamicAtomicBin";

TEST_F(GraphExecutorWithOverflowKernelUnitTest, SingleNodeAiCore_SupportOverflow_WithSingleOpScene_ExecuteSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_node = graph->FindNode("add1");
  AttrUtils::SetBool(add_node->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(add_node->GetOpDesc(), "globalworkspace_size", 32U);
  AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model)
                         .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").AtomicStubNum(DynamicAtomicStubName))
                         .Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  void *device_block = (void *)0x01;
  auto outputs = FakeTensors({2048}, 1, device_block);

  auto i0 = FakeValue<Tensor>(
      Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, device_block});
  auto i1 = FakeValue<Tensor>(
      Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, device_block});
  auto inputs = std::vector<Tensor *>({i0.holder.get(), i1.holder.get()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, 184);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, 152);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, 168);
  auto args_host_buffer = reinterpret_cast<TensorAddress *>(add_launch_args->GetArgsEx()->args);
  ASSERT_NE(args_host_buffer, nullptr);
  EXPECT_NE(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U], nullptr);
  EXPECT_EQ(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U],
            args_host_buffer[add_launch_args->GetArgsEx()->tilingDataOffset / 8 - 1U]);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithOverflowKernelUnitTest, SingleNodeAiCore_SupportOverflow_ExecuteSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_node = graph->FindNode("add1");
  AttrUtils::SetBool(add_node->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(add_node->GetOpDesc(), "globalworkspace_size", 32U);
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model)
                         .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").AtomicStubNum(DynamicAtomicStubName))
                         .Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  void *device_block = (void *)0x01;
  auto outputs = FakeTensors({2048}, 1, device_block);

  auto i0 = FakeValue<Tensor>(
      Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, device_block});
  auto i1 = FakeValue<Tensor>(
      Tensor{{{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, device_block});
  auto inputs = std::vector<Tensor *>({i0.holder.get(), i1.holder.get()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, 184);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, 152);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, 168);
  auto args_host_buffer = reinterpret_cast<TensorAddress *>(add_launch_args->GetArgsEx()->args);
  ASSERT_NE(args_host_buffer, nullptr);
  EXPECT_NE(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U], nullptr);
  EXPECT_EQ(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U],
            args_host_buffer[add_launch_args->GetArgsEx()->tilingDataOffset / 8 - 1U]);
  rtStreamDestroy(stream);
}
}
