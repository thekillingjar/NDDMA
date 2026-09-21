/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/model_v2_executor.h"
#include <gtest/gtest.h>
#include "lowering/model_converter.h"

// stub and faker
#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "faker/fake_value.h"
#include "faker/magic_ops.h"
#include "faker/aicore_taskdef_faker.h"
#include "stub/gert_runtime_stub.h"
#include "check/executor_statistician.h"
#include "graph/operator_reg.h"
#include "graph/utils/graph_dump_utils.h"
#include "faker/space_registry_faker.h"

using namespace ge;

REG_OP(If)
    .INPUT(cond, TensorType::ALL())
        .DYNAMIC_INPUT(input, TensorType::ALL())
        .DYNAMIC_OUTPUT(output, TensorType::ALL())
        .GRAPH(then_branch)
        .GRAPH(else_branch)
        .OP_END_FACTORY_REG(If)

namespace gert {
class CtrlOpST : public testing::Test {
 public:
  void RunGraph_ChainConflict(bool branch) {
    TensorHolder pred_holder;
    StorageShape output0_expect_shape = {{8, 3, 16, 16}, {8, 3, 16, 16}};
    StorageShape output1_expect_shape;
    if (branch) {
      pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({1}).Build();
      output1_expect_shape = {{8, 3, 16, 16}, {8, 3, 16, 16}};
    } else {
      pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({0}).Build();
      output1_expect_shape = {{2, 4 * 3 * 16 * 16}, {2, 4 * 3 * 16 * 16}};
    }

    auto compute_graph = ShareGraph::IfGraphShapeChangedOneBranch();
    ASSERT_NE(compute_graph, nullptr);
    compute_graph->TopologicalSorting();
    ge::GraphUtils::DumpGEGraphToOnnx(*compute_graph, "ComputeGraphChainConflict");
    GeModelBuilder builder(compute_graph);
    auto ge_root_model = builder.BuildGeRootModel();

    GertRuntimeStub rtstub;
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "ExecuteGraphChainConflict");
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
    ASSERT_NE(model_executor, nullptr);

    auto input_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 16, 16}).Build();
    std::vector<Tensor *> inputs{pred_holder.GetTensor(), input_holder.GetTensor()};
    auto output0_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 24, 16}).Build();
    auto output1_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 24, 16}).Build();
    std::vector<Tensor *> outputs{output0_holder.GetTensor(), output1_holder.GetTensor()};

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

    ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    ASSERT_EQ(model_executor->Execute({stream}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(output0_holder.GetTensor()->GetShape(), output0_expect_shape);
    ASSERT_EQ(output1_holder.GetTensor()->GetShape(), output1_expect_shape);

    output0_holder.GetTensor()->GetShape().MutableStorageShape() = {8, 3, 24, 16};
    output0_holder.GetTensor()->GetShape().MutableOriginShape() = {8, 3, 24, 16};
    output1_holder.GetTensor()->GetShape().MutableStorageShape() = {8, 3, 24, 16};
    output1_holder.GetTensor()->GetShape().MutableOriginShape() = {8, 3, 24, 16};
    ASSERT_EQ(model_executor->Execute({stream}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(output0_holder.GetTensor()->GetShape(), output0_expect_shape);
    ASSERT_EQ(output1_holder.GetTensor()->GetShape(), output1_expect_shape);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  void RunIfGraph2(TensorHolder &pred_tensor, bool expect_branch) {
    auto compute_graph = ShareGraph::IfGraph2();
    ASSERT_NE(compute_graph, nullptr);
    compute_graph->TopologicalSorting();
    GeModelBuilder builder(compute_graph);
    auto ge_root_model = builder.BuildGeRootModel();

    GertRuntimeStub rtstub;
    rtstub.StubByNodeTypes({"Shape", "Rank"});

    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
    ASSERT_NE(exe_graph, nullptr);
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    auto input_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
    std::vector<Tensor *> inputs{pred_tensor.GetTensor(), input_holder.GetTensor()};
    auto output_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT64).Shape({8}).Build();
    std::vector<Tensor *> outputs{output_holder.GetTensor()};
    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    auto shape_count = expect_branch ? 1 : 0;
    auto rank_count = expect_branch ? 0 : 1;
    auto output_shape = expect_branch ? Shape({4}) : Shape({});
    auto output_data = expect_branch ? Shape({8, 3, 224, 224}) : Shape({4});

    ess->Clear();
    ASSERT_EQ(
        model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
        ge::GRAPH_SUCCESS);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Shape", "FakeExecuteNode"), shape_count);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Rank", "FakeExecuteNode"), rank_count);
    EXPECT_EQ(output_holder.GetTensor()->GetShape().GetStorageShape(), output_shape);
    EXPECT_EQ(output_holder.GetTensor()->GetShape().GetOriginShape(), output_shape);
    EXPECT_EQ(memcmp(output_holder.GetTensor()->GetAddr(), &output_data[0], output_data.GetDimNum() * 8), 0);

    ess->Clear();
    ASSERT_EQ(
        model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
        ge::GRAPH_SUCCESS);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Shape", "FakeExecuteNode"), shape_count);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Rank", "FakeExecuteNode"), rank_count);
    EXPECT_EQ(output_holder.GetTensor()->GetShape().GetStorageShape(), output_shape);
    EXPECT_EQ(output_holder.GetTensor()->GetShape().GetOriginShape(), output_shape);
    EXPECT_EQ(memcmp(output_holder.GetTensor()->GetAddr(), &output_data[0], output_data.GetDimNum() * 8), 0);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  void RunIfGraph3(TensorHolder &pred_tensor, bool expect_branch) {
    auto compute_graph = ShareGraph::IfGraph3();
    ASSERT_NE(compute_graph, nullptr);
    compute_graph->TopologicalSorting();
    auto ge_root_model = GeModelBuilder(compute_graph).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

    GertRuntimeStub stub;

    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "IfDiffPlacement");
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    auto input_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
    std::vector<Tensor *> inputs{pred_tensor.GetTensor(), input_holder.GetTensor()};
    auto output_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
    std::vector<Tensor *> outputs{output_holder.GetTensor()};
    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(
        model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
        ge::GRAPH_SUCCESS);

    auto expect = gert::Shape({8,3,224,224});
    EXPECT_EQ(output_holder.GetTensor()->GetShape().GetOriginShape(), expect);
    // todo: rts被打桩，所以无法观察输出tensor的内容
    // todo: 当前stub实现中没有device和host的概念，暂时验证不了launch时都是device的地址

    ess->Clear();
    ASSERT_EQ(
        model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
        ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_holder.GetTensor()->GetShape().GetOriginShape(), gert::Shape({8,3,224,224}));

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  void RunCaseGraph(TensorHolder &index_tensor, int32_t expect_branch) {
    auto compute_graph = ShareGraph::CaseGraph();
    ASSERT_NE(compute_graph, nullptr);
    compute_graph->TopologicalSorting();
    GeModelBuilder builder(compute_graph);
    auto ge_root_model = builder.BuildGeRootModel();

    GertRuntimeStub rtstub;
    rtstub.StubByNodeTypes({"Shape", "Rank", "Size"});

    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
    ASSERT_NE(exe_graph, nullptr);
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    auto input_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8, 3, 224, 224}).Build();
    std::vector<Tensor *> inputs{index_tensor.GetTensor(), input_holder.GetTensor()};
    auto output_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT64).Shape({8}).Build();
    std::vector<Tensor *> outputs{output_holder.GetTensor()};
    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    size_t shape_count = 0, rank_count = 0, size_count = 0;
    switch (expect_branch) {
      case 0:
        shape_count = 1;
        break;
      case 1:
        rank_count = 1;
        break;
      case 2:
        size_count = 1;
        break;
      default:
        break;
    }

    ess->Clear();
    ASSERT_EQ(
        model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
        ge::GRAPH_SUCCESS);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Shape", "FakeExecuteNode"), shape_count);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Rank", "FakeExecuteNode"), rank_count);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Size", "FakeExecuteNode"), size_count);

    ess->Clear();
    ASSERT_EQ(
        model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
        ge::GRAPH_SUCCESS);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Shape", "FakeExecuteNode"), shape_count);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Rank", "FakeExecuteNode"), rank_count);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Size", "FakeExecuteNode"), size_count);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
};
TEST_F(CtrlOpST, IfOp_ExecuteCorrectBranch_WhenTrue) {
  auto pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({1}).Build();
  RunIfGraph2(pred_holder, true);

  pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Shape({1, 1}).Value<int32_t>({1}).Build();
  RunIfGraph2(pred_holder, true);
}
TEST_F(CtrlOpST, IfOp_ExecuteCorrectBranch_WhenFalse) {
  auto pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({0}).Build();
  RunIfGraph2(pred_holder, false);
  pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Shape({1, 0, 1}).Build();
  RunIfGraph2(pred_holder, false);
}

TEST_F(CtrlOpST, IfOp_CopyH2D_DifferentPlacemenetBranchs) {
  // 只能加载libgert_op_impl.so而且在CreateDefaultSpaceRegistry里要先加载
  SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl();
  auto pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({1}).Build();
  RunIfGraph3(pred_holder, true);
}

TEST_F(CtrlOpST, CaseOp_ExecuteCorrectBranch_DefaultBranch) {
  auto pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({-1}).Build();
  RunCaseGraph(pred_holder, 2);
  pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({3}).Build();
  RunCaseGraph(pred_holder, 2);
}
TEST_F(CtrlOpST, CaseOp_ExecuteCorrectBranch_Branch0) {
  auto pred_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({0}).Build();
  RunCaseGraph(pred_holder, 0);
}

TEST_F(CtrlOpST, GraphWithUnsupportedV1CtrlOp) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph("Enter");
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();
  GeModelBuilder builder(compute_graph);
  auto ge_root_model = builder.BuildGeRootModel();
  EXPECT_EQ(ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model), nullptr);
  bg::ValueHolder::PopGraphFrame();
}

TEST_F(CtrlOpST, WhileOp) {
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

  for (auto &graph : {ShareGraph::WhileGraph2(), ShareGraph::WhileGraph2(true)}) {
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.BuildGeRootModel();

    bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
    ASSERT_NE(model_executor, nullptr);

    int32_t output = 0;
    ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({}, 1, &output);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i1 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    auto inputs = FakeTensors({}, 1);
    *static_cast<int32_t *>(inputs.data()[0].GetAddr()) = 0;

    ASSERT_EQ(model_executor->Execute({i1.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()),
              ge::GRAPH_SUCCESS);
    // Check output value
    EXPECT_EQ(*static_cast<int32_t *>(static_cast<gert::Tensor *>(outputs.GetAddrList()[0])->GetAddr()), 5);

    *static_cast<int32_t *>(inputs.data()[0].GetAddr()) = 8;
    ASSERT_EQ(model_executor->Execute({i1.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(*static_cast<int32_t *>(static_cast<gert::Tensor *>(outputs.GetAddrList()[0])->GetAddr()), 8);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
}

TEST_F(CtrlOpST, WhileOpWithXBody) {
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

  for (auto &graph : {ShareGraph::WhileGraphXBody()}) {
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.BuildGeRootModel();

    bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
    ASSERT_NE(model_executor, nullptr);

    std::vector<TensorHolder> input_holders;
    std::vector<Tensor *> inputs;
    input_holders.push_back(TensorFaker().Build());
    input_holders.push_back(TensorFaker().Build());
    inputs.push_back(input_holders[0].GetTensor());
    inputs.push_back(input_holders[1].GetTensor());

    *static_cast<int32_t *>(inputs[0]->GetAddr()) = 1;
    *static_cast<int32_t *>(inputs[1]->GetAddr()) = 5;

    std::vector<TensorHolder> output_holders;
    std::vector<Tensor *> outputs;
    output_holders.push_back(TensorFaker().Build());
    output_holders.push_back(TensorFaker().Build());
    outputs.push_back(output_holders[0].GetTensor());
    outputs.push_back(output_holders[1].GetTensor());
    ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto i1 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_EQ(model_executor->Execute({i1.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
              ge::GRAPH_SUCCESS);
    // Check output value
    EXPECT_EQ(*static_cast<int32_t *>(outputs[0]->GetAddr()), 5);
    EXPECT_EQ(*static_cast<int32_t *>(outputs[1]->GetAddr()), 1);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
}

/**
 * 用例描述：构造while算子级联的图，并多次构造执行器执行
 * 预置条件：
 * 1. 构图包含while控制算子,并且两个while算子直接相连
 * 2. 构造LessThan_5、LargerThan_1桩函数，用于构造while算子正常循环执行一次
 * 3. 输入value构造
 *
 * 预期结果
 * 1. 执行结果正常，执行输出结果和输入一致
 * 2. 多次构造执行器，执行正常，结果正确
 */
TEST_F(CtrlOpST, WhileOpCascadeExecuteMultiTimes) {
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
  MagicOpFakerBuilder("LargerThan_1")
      .OutputPlacement(kOnHost)
      .KernelFunc([](KernelContext *ctx) {
        auto input = ctx->GetInputPointer<GertTensorData>(kFkiStart + 1)->GetAddr();
        auto output = ctx->GetOutputPointer<GertTensorData>(1)->GetAddr();
        *static_cast<bool *>(output) = (*static_cast<int32_t *>(input) > 1);
        return ge::GRAPH_SUCCESS;
      })
      .Build(stub);

  for (auto &graph : {ShareGraph::WhileGraphCascade()}) {
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    ge::DumpGraph(graph, "WhileOpCascadeComputeGraph");
    auto ge_root_model = builder.BuildGeRootModel();

    bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

    ge::DumpGraph(exe_graph.get(), "WhileOpCascadeExeGraph");
    for (int32_t i = 0; i < 3; ++i) {
      auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
      ASSERT_NE(model_executor, nullptr);

      std::vector<TensorHolder> input_holders;
      std::vector<Tensor *> inputs;
      input_holders.push_back(TensorFaker().Build());
      input_holders.push_back(TensorFaker().Build());
      inputs.push_back(input_holders[0].GetTensor());
      inputs.push_back(input_holders[1].GetTensor());

      *static_cast<int32_t *>(inputs[0]->GetAddr()) = 1;
      *static_cast<int32_t *>(inputs[1]->GetAddr()) = 5;

      std::vector<TensorHolder> output_holders;
      std::vector<Tensor *> outputs;
      output_holders.push_back(TensorFaker().Build());
      output_holders.push_back(TensorFaker().Build());
      outputs.push_back(output_holders[0].GetTensor());
      outputs.push_back(output_holders[1].GetTensor());
      ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

      rtStream_t stream;
      ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
      auto i1 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

      ASSERT_EQ(model_executor->Execute({i1.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
                ge::GRAPH_SUCCESS);
      // Check output value
      EXPECT_EQ(*static_cast<int32_t *>(outputs[0]->GetAddr()), 1);
      EXPECT_EQ(*static_cast<int32_t *>(outputs[1]->GetAddr()), 5);

      ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
      rtStreamDestroy(stream);
    }
  }
}
/*
 *                                 +--------------+
 *                                 | Else Graph   |
 *                 +------------+  |              |
 *                 | Then graph |  |   NetOutput  |
 *                 |            |  |      |       |
 *    NetOutput    | NetOutput  |  |   ReShape    |
 *     |     \     |    |       |  |     / \      |
 *     |     If -- |  Data      |  | Data  Const  |
 *     \    /      +------------+  +--------------+
 *      data
 */
TEST_F(CtrlOpST, IfOp_OuterNotChanged_WhenChainConflict) {
  RunGraph_ChainConflict(true);
  RunGraph_ChainConflict(false);
}
}  // namespace gert