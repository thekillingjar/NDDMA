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
#include "common/share_graph.h"
#include "lowering/graph_converter.h"
#include "faker/global_data_faker.h"
#include "runtime/model_v2_executor.h"
#include "common/bg_test.h"
#include "runtime/dev.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "op_impl/dynamic_rnn_impl.h"
#include "op_impl/data_flow_op_impl.h"
#include "lowering/model_converter.h"
#include "mmpa/mmpa_api.h"
#include "securec.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"

#include "faker/aicpu_taskdef_faker.h"
#include "faker/model_data_faker.h"
#include "framework/common/ge_types.h"
#include "runtime/gert_api.h"
#include "stub/hostcpu_mmpa_stub.h"
#include "check/executor_statistician.h"
#include "graph_builder/bg_infer_shape_range.h"
#include "common/helper/model_parser_base.h"
#include "register/kernel_registry.h"

using namespace ge;
namespace gert {
class AicpuE2ESt : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
  void TearDown() override {
    Test::TearDown();
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
  }
};

namespace {
void CheckAndExecute(ComputeGraphPtr &graph, TaskDefFaker &task_def) {
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", task_def).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048, 1, 1, 1}, 1);
  auto inputs = FakeTensors({2048, 1, 1, 1}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

void CheckAndExecute3rdOp(ComputeGraphPtr &graph, TaskDefFaker &task_def) {
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", task_def)
                              .AddTaskDef("NonZero", task_def)
                              .BuildGeRootModel();
  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2}, 1);
  auto inputs = FakeTensors({2048, 2, 3, 4}, 2);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto ess = StartExecutorStatistician(model_executor);
  ess->Clear();

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("NonZero", "InferShapeRange"), 1);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  bg::ShapeRangeInferenceResult::ErrorResult();
  rtStreamDestroy(stream);
}
}  // namespace

class AicpuTfLaunchStub : public RuntimeStub {
 public:
  rtError_t rtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char *opName, uint32_t blockDim,
                                          const rtAicpuArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                          rtStream_t stream, uint32_t flags) override {
    EXPECT_EQ(argsInfo->kernelOffsetInfoNum, 3);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].addrOffset, 80);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].dataOffset, 112);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[1].addrOffset, 88);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[1].dataOffset, 294);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[2].addrOffset, 104);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[2].dataOffset, 318);
    return RT_ERROR_NONE;
  }
};

TEST_F(AicpuE2ESt, SingleNodeAiCpuTf_ExecuteSuccess) {
   auto aicpu_launch_stub = std::make_shared<AicpuTfLaunchStub>();
   RuntimeStub::Install(aicpu_launch_stub.get());

   auto graph = ShareGraph::BuildSingleNodeGraph();
   auto add_desc = graph->FindNode("add1")->GetOpDesc();
   ge::AttrUtils::SetStr(add_desc, "ops_json_path", "tf_kernel.json");
   graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
   AiCpuTfTaskDefFaker aicpu_task_def_faker;
   CheckAndExecute(graph, aicpu_task_def_faker);

   RuntimeStub::UnInstall(nullptr);
 }

class AicpuCCLaunchStub : public RuntimeStub {
 public:
  rtError_t rtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char *opName, uint32_t blockDim,
                                          const rtAicpuArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                          rtStream_t stream, uint32_t flags) override {
    EXPECT_EQ(argsInfo->kernelOffsetInfoNum, 1);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].addrOffset, 12);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].dataOffset, 98);
    EXPECT_EQ(argsInfo->kernelNameAddrOffset, 92);
    EXPECT_EQ(argsInfo->soNameAddrOffset, 97);
    return RT_ERROR_NONE;
  }
};

TEST_F(AicpuE2ESt, SingleNodeAiCpuCC_ExecuteSuccess) {
  auto aicpu_launch_stub = std::make_shared<AicpuCCLaunchStub>();
  RuntimeStub::Install(aicpu_launch_stub.get());

  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_desc = graph->FindNode("add1")->GetOpDesc();
  ge::AttrUtils::SetStr(add_desc, "ops_json_path", "aicpu_kernel.json");
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  CheckAndExecute(graph, aicpu_task_def_faker);

  RuntimeStub::UnInstall(nullptr);
}

class ThirdAicpuLaunchStub : public RuntimeStub {
 public:
  rtError_t rtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char *opName, uint32_t blockDim,
                                          const rtAicpuArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                          rtStream_t stream, uint32_t flags) override {
    if (opName == string("nonzero")) {
      EXPECT_EQ(argsInfo->kernelOffsetInfoNum, 2);
      EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].addrOffset, 80);
      EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].dataOffset, 112);
      EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[1].addrOffset, 88);
      EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[1].dataOffset, 350);
    }
    return RT_ERROR_NONE;
  }
};

// TEST_F(AicpuE2ESt, ThirdAicpuOp_ExecuteSuccess) {
//   auto aicpu_launch_stub = std::make_shared<ThirdAicpuLaunchStub>();
//   RuntimeStub::Install(aicpu_launch_stub.get());

//   auto graph = ShareGraph::ThirdAicpuOpGraph();
//   graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
//   graph->FindNode("nonzero")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
//   AiCpuTfTaskDefFaker aicpu_task_def_faker;
//   CheckAndExecute3rdOp(graph, aicpu_task_def_faker);
//   RuntimeStub::UnInstall(nullptr);
// }

TEST_F(AicpuE2ESt, TensorListOp_ExecuteSuccess) {
  auto graph = ShareGraph::TensorListGraph();
  AiCpuTfTaskDefFaker task_def;
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("EmptyTensorList", task_def)
                              .AddTaskDef("TensorListPushBack", task_def)
                              .AddTaskDef("TensorListPopBack", task_def)
                              .BuildGeRootModel();
  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2}, 1);
  auto inputs = FakeTensors({2048, 2, 3, 4}, 4);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(AicpuE2ESt, ThirdAicpuOpHostCpuEngine_ExecuteSuccess) {
  char_t opp_path[MMPA_MAX_PATH] = "./";
  mmSetEnv("ASCEND_OPP_PATH", &opp_path[0U], MMPA_MAX_PATH);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<HostcpuMockMmpa>());

  auto graph = ShareGraph::ThirdAicpuOpGraph();
  graph->FindNode("nonzero")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu.c_str());
  AttrUtils::SetInt(graph->FindNode("nonzero")->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  CheckAndExecute3rdOp(graph, aicpu_task_def_faker);
}

TEST_F(AicpuE2ESt, AicpuOpHostCpuEngine_RefSmallShape_ExecuteSuccess) {
  char_t opp_path[MMPA_MAX_PATH] = "./";
  mmSetEnv("ASCEND_OPP_PATH", &opp_path[0U], MMPA_MAX_PATH);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<HostcpuMockMmpa>());

  auto graph = ShareGraph::BuildRefnodeGraph();
  auto refnode_desc = graph->FindNode("refnode")->GetOpDesc();
  ge::AttrUtils::SetBool(refnode_desc, "reference", true);
  refnode_desc->SetOpKernelLibName(ge::kEngineNameHostCpu.c_str());

  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  CheckAndExecute(graph, aicpu_task_def_faker);
}
} // namespace
