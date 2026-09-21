/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/gert_api.h"
#include <gtest/gtest.h>
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator_factory_impl.h"
#include "graph/ge_local_context.h"

#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/model_data_faker.h"
#include "faker/fake_value.h"
#include "stub/gert_runtime_stub.h"
#include "depends/runtime/src/runtime_stub.h"
#include "faker/space_registry_faker.h"

namespace gert {
class StreamExecutorST : public testing::Test {
  void SetUp() override {
    CreateVersionInfo();
    SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  }

  void TearDown() override {
    DestroyVersionInfo();
  }
};
TEST_F(StreamExecutorST, OneStream_Success_ExecuteTwoTimes) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

  auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
TEST_F(StreamExecutorST, OneStream_ExecuteSuccess_GetTwoTimes) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  auto add = graph->FindNode("add1");
  ASSERT_NE(add, nullptr);
  auto op_desc = add->GetOpDesc();
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);

  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>(ge::TILING_SCHEDULE_OPTIMIZE, "1"));
  ge::GetThreadLocalContext().SetGlobalOption(options);
  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_DEV_CAP_SUPPORT);

  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  options.clear();
  ge::GetThreadLocalContext().SetGlobalOption(options);
}

TEST_F(StreamExecutorST, OneStream_ExecuteSuccess_Memcheck) {
  auto graph = ShareGraph::BuildSingleNodeGraph("AddMemcheck", {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                {{100, 2, 3, 4}, {1, 100, 3, 4}, {100}, {100, 100, 3, 4}});
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  auto add = graph->FindNode("add1");
  ASSERT_NE(add, nullptr);
  auto op_desc = add->GetOpDesc();
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3);
  (void)ge::AttrUtils::SetStr(op_desc, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_outputs_indexes", {{0}}); 
  add->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x20", 1}};
  add->GetOpDesc()->MutableAllOutputName() = {{"output0", 0}};
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputDynamic);
  op_desc->AppendIrOutput("output", ge::kIrOutputDynamic);
  op_desc->SetOpKernelLibName(ge::kEngineNameAiCore);

  auto infer_fun = [](ge::Operator &op) -> ge::graphStatus {
    const char_t *name = "output";
    op.UpdateOutputDesc(name, op.GetInputDesc(0));
    return ge::GRAPH_SUCCESS;
  };
  ge::OperatorFactoryImpl::RegisterInferShapeFunc("AddMemcheck", infer_fun);

  graph->TopologicalSorting();
  auto ge_root_model =
      GeModelBuilder(graph)
          .AddTaskDef(
              "AddMemcheck",
              AiCoreTaskDefFaker("AddStubBin").WithHandle().ArgsFormat("{i0}{i_desc1}{o_desc0}{ws0}{overflow_addr}"))
          .FakeTbeBin({"AddMemcheck"})
          .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(StreamExecutorST, OneStream_ExecuteSuccess_Memcheck_instance) {
  auto graph = ShareGraph::BuildSingleNodeGraph("AddMemcheck", {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                {{100, 2, 3, 4}, {1, 100, 3, 4}, {100}, {100, 100, 3, 4}});
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  auto add = graph->FindNode("add1");
  ASSERT_NE(add, nullptr);
  auto op_desc = add->GetOpDesc();
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3);
  (void)ge::AttrUtils::SetStr(op_desc, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_outputs_indexes", {{0}});
  add->GetOpDesc()->MutableAllInputName() = {{"x1", 0}, {"x20", 1}};
  add->GetOpDesc()->MutableAllOutputName() = {{"output0", 0}};
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputDynamic);
  op_desc->AppendIrOutput("output", ge::kIrOutputDynamic);
  op_desc->SetOpKernelLibName(ge::kEngineNameAiCore);

  auto infer_fun = [](ge::Operator &op) -> ge::graphStatus {
    const char_t *name = "output";
    op.UpdateOutputDesc(name, op.GetInputDesc(0));
    return ge::GRAPH_SUCCESS;
  };
  ge::OperatorFactoryImpl::RegisterInferShapeFunc("AddMemcheck", infer_fun);

  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
      .AddTaskDef("AddMemcheck", AiCoreTaskDefFaker("AddStubBin").WithHandle().ArgsFormat("{i_instance0}{i_instance1}{o_instance0}{ws0}{overflow_addr}"))
      .FakeTbeBin({"AddMemcheck"})
      .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(StreamExecutorST, OneStream_ExecuteSuccess_AfterErase) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(stream_executor->Erase(stream), ge::GRAPH_SUCCESS);

  model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
TEST_F(StreamExecutorST, TwoStream_ExecuteSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  auto add = graph->FindNode("add1");
  ASSERT_NE(add, nullptr);
  auto op_desc = add->GetOpDesc();
  op_desc->SetWorkspaceBytes({4, 8});
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  ge::AttrUtils::SetInt(op_desc, "ori_op_para_size", 512);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  rtStream_t stream1, stream2;
  ASSERT_EQ(rtStreamCreate(&stream1, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  ASSERT_EQ(rtStreamCreate(&stream2, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream1));
  auto model_executor = stream_executor->GetOrCreateLoaded(stream1, {stream1, nullptr});
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream2));
  model_executor = stream_executor->GetOrCreateLoaded(stream2, {stream2, nullptr});
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  dlog_setlevel(0, 3, 0);
  rtStreamDestroy(stream1);
  rtStreamDestroy(stream2);
}

/**
 * 用例描述：确保LoweringOptions在StreamExecutor是生效的
 * 本用例使用StreamExecutor执行一个基础场景的LoweringOption用例：打开trust_output_tensor开关时，输出节点的InferShape被消除
 *
 * 预置条件：
 * 1. Fake Add算子的lowering和对应InferShape、Tiling等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含Add算子的单算子图
 * 2. lowering、加载这张图
 * 3. 挂接ess（kernel执行监控）
 * 4. fake输入、输出tensor，origin shape为8，3，224，224；storage shape为8，1，224，224，16
 * 5. 将option trust_output_tensor设置为true，执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出tensor的storage shape未变化
 * 3. 通过ess观察，没有执行InferShape kernel
 */
TEST_F(StreamExecutorST, LoweringOption_TakeEffect) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();

  ge::graphStatus error_code = ge::GRAPH_FAILED;
  LoweringOption opiton{.trust_shape_on_out_tensor = true};
  auto stream_executor =
      LoadStreamExecutorFromModelData(model_data_holder.Get(), opiton, error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  auto model_executor = stream_executor->GetOrCreateLoaded(stream, {stream, nullptr});
  ASSERT_NE(model_executor, nullptr);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8, 1, 224, 224, 16}));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8, 1, 224, 224, 16}));

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
}  // namespace gert