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
#include "op_impl/dynamic_rnn_impl.h"
#include "op_impl/data_flow_op_impl.h"
#include "lowering/model_converter.h"
#include "mmpa/mmpa_api.h"
#include "securec.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_context.h"
#include "graph_builder/bg_memory.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "faker/aicpu_taskdef_faker.h"
#include "faker/model_data_faker.h"
#include "framework/common/ge_types.h"
#include "runtime/gert_api.h"
#include "stub/hostcpu_mmpa_stub.h"
#include "check/executor_statistician.h"
#include "graph_builder/bg_infer_shape_range.h"
#include "common/helper/model_parser_base.h"
#include "register/kernel_registry.h"
#include "common/topo_checker.h"
#include "graph/ge_local_context.h"

using namespace ge;
namespace gert {
void CheckModelOutputManagerIsNull(ComputeGraphPtr graph, size_t input_num, bool is_need_fake_add = true) {
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  GeRootModelPtr ge_root_model;
  if (is_need_fake_add) {
    ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  } else {
    ge_root_model = builder.BuildGeRootModel();
  }

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  Tensor out1, out2;
  std::vector<Tensor *> outputs = {&out1, &out2};
  auto inputs = FakeTensors({2048}, input_num);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  TensorAddrManager out1_manager, out2_manager;
  auto out1_addr = out1.MutableTensorData().Release(out1_manager);
  auto out2_addr = out2.MutableTensorData().Release(out2_manager);
  ASSERT_EQ(out1_addr, out2_addr);
  ASSERT_NE(out1_manager, nullptr);
  ASSERT_NE(out2_manager, nullptr);
  out1.MutableTensorData().SetAddr(out1_addr, out1_manager);
  out2.MutableTensorData().SetAddr(out2_addr, out2_manager);
  out1.MutableTensorData().Free();
  out2.MutableTensorData().Free();
  ASSERT_EQ(out1.GetAddr(), nullptr);
  ASSERT_EQ(out2.GetAddr(), nullptr);
  rtStreamDestroy(stream);
}
class GraphExecutorWithKernelUnitTest : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
    std::string opp_path = "./";
    std::string opp_version = "version.info";
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
    setenv("GE_USE_STATIC_MEMORY", "3", 1);
    system(("touch " + opp_version).c_str());
    system(("echo 'Version=3.20.T100.0.B356' > " + opp_version).c_str());
    memory::RtsCachingMemAllocator::GetAllocator(0, RT_MEMORYINFO_HBM)->Recycle();
    memory::RtsCachingMemAllocator::device_id_to_allocators_.clear();
  }

  void TearDown() override {
    Test::TearDown();
    system("rm -f ./version.info");
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("GE_USE_STATIC_MEMORY");
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
  }
};

graphStatus StubHostKernel(KernelContext *context) {
  return GRAPH_SUCCESS;
}

const std::string TransDataStubName = "TransDataStubBin";
const std::string TransData13StubName = "TransData17StubBin";
const std::string DynamicAtomicStubName = "DynamicAtomicBin";
const std::string DynamicRnnv3StubName = "DynamicRNNV3StubBin";
const std::string AddStubName = "AddStubBin";
const std::string MulStubName = "MulStubBin";
const std::string ReduceSumStubName = "ReduceSumStubBin";

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeReshape_LoadUnloadSuccess) {
  auto graph = ShareGraph::BuildReshapeGraph();
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EReshapeGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeReshape_LoadUnloadSuccess2) {
  auto graph = ShareGraph::BuildReshapeGraph2();
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  EXPECT_EQ(model_executor->Load({stream}), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeAiCore_LoadUnloadSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(GraphExecutorWithKernelUnitTest, VirtualNodeAiCore_LoadUnloadSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  for (const NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == "Add") {
      ge::AttrUtils::SetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, true);
    }
  }
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(GraphExecutorWithKernelUnitTest, ThirdClassNodeAiCore_LoadUnloadSuccess) {
  auto graph = ShareGraph::BuildAiCoreThirdClassNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                              .AddTaskDef("NonZero", AiCoreTaskDefFaker("NonZeroStubBin").WithHandle())
                              .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(GraphExecutorWithKernelUnitTest, Lstmp_With_Null_Compile_Result_Test) {
  auto graph = ShareGraph::LstmpGraph();
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data =
      GlobalDataFaker(root_model)
          .AddTaskDef("transdata_13", AiCoreTaskDefFaker(TransData13StubName).AtomicStubNum(DynamicAtomicStubName))
          .AddTaskDef("TransData", AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName))
          .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName))
          .BuildEmptyGlobalData();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  EXPECT_EQ(exe_graph, nullptr);
}

const std::string save_om_file_name = "file_name_prefix.om";
TEST_F(GraphExecutorWithKernelUnitTest, save_and_load_static_model) {
  GeModelBuilder builder(ShareGraph::LstmpGraph());
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data = gert::ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();

  ge::graphStatus error_code;
  auto executor_model = gert::LoadExecutorFromModelData(model_data.Get(), error_code);
  ASSERT_NE(error_code, ge::GRAPH_SUCCESS);
  ASSERT_EQ(executor_model, nullptr);

  executor_model = gert::LoadExecutorFromFile(save_om_file_name.c_str(), error_code);
  ASSERT_NE(error_code, ge::GRAPH_SUCCESS);
  ASSERT_EQ(executor_model, nullptr);
}

LowerResult LoweringAdd(const ge::NodePtr &node, const LowerInput &lower_input) {
  size_t output_size = 8U;
  gert::StorageShape shape;
  auto size_holder = bg::ValueHolder::CreateConst(&output_size, sizeof(output_size));
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, {size_holder}, *(lower_input.global_data));
  auto compute_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>(
      "LaunchKernelWithHandle", {lower_input.input_addrs[1], output_addrs[0], lower_input.global_data->GetStream()});

  return {HyperStatus::Success(), {compute_holder}, {lower_input.input_shapes[0]}, output_addrs};
}
REGISTER_NODE_CONVERTER("_lower_add", LoweringAdd);

TEST_F(GraphExecutorWithKernelUnitTest, LoadExecutorFromModelDataWithRtSession_exec_successfully) {
  auto graph = ShareGraph::SimpleVariableAddGraph();
  graph->TopologicalSorting();

  auto add = graph->FindNode("add");
  ASSERT_NE(add, nullptr);
  ge::AttrUtils::SetStr(add->GetOpDesc(), "_ge_attr_lowering_func", "_lower_add");

  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder.AddTaskDef("Add", AiCoreTaskDefFaker("AdsStubName")).FakeTbeBin({"Add"}).BuildGeRootModel();

  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  GertStreamStub rts_stub;
  ge::graphStatus error_code;
  RtSession session(999);
  auto executor = gert::LoadExecutorFromModelDataWithRtSession(model_data.Get(), &session, error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(executor, nullptr);

  gert::ModelLoadArg load_arg(&session);
  EXPECT_EQ(executor->Load({}, load_arg), ge::GRAPH_SUCCESS);

  auto executor1 = gert::LoadExecutorFromModelDataWithRtSession(model_data.Get(), &session, error_code);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
  ASSERT_NE(executor1, nullptr);
  // session id mis match
  RtSession session1(888);
  gert::ModelLoadArg load_arg1(&session1);
  EXPECT_NE(executor1->Load({}, load_arg1), ge::GRAPH_SUCCESS);
}

TEST_F(GraphExecutorWithKernelUnitTest, check_is_dynamic_model)
{
  GeModelBuilder builder(ShareGraph::LstmpGraph());
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShapeFile(save_om_file_name.c_str());
  bool is_dynamic_model = true;
  ASSERT_EQ(IsDynamicModel(save_om_file_name.c_str(), is_dynamic_model), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(is_dynamic_model == true);
}

TEST_F(GraphExecutorWithKernelUnitTest, check_is_dynamic_model_with_model)
{
  GeModelBuilder builder(ShareGraph::LstmpGraph());
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  bool is_dynamic_model = true;
  ASSERT_EQ(IsDynamicModel(model_data_holder.Get().model_data, model_data_holder.Get().model_len, is_dynamic_model), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(is_dynamic_model == true);
  ASSERT_EQ(IsDynamicModel(model_data_holder.Get().model_data, 0UL, is_dynamic_model), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeReshape_ExecuteSuccess) {
  auto graph = ShareGraph::BuildReshapeGraph();
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EReshapeGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->ExecuteSync(inputs.GetTensorList(), inputs.size(),
                                        reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeAiCore_ExecuteSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
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

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeAiCore_with_frozen_input_ExecuteSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto op_desc = graph->FindNode("data2")->GetOpDesc();
  EXPECT_TRUE(AttrUtils::SetBool(op_desc, "frozen_input", true));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "addr", 1111));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "size", 2048 * sizeof(float)));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "placement", ge::Placement::kPlacementDevice));
  std::vector<int64_t> input_shape = {2048};
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "storage_shape", input_shape));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "origin_shape", input_shape));
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
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

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeAiCore_GetOutNodeNameSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  EXPECT_EQ(ge_root_model->GetSubgraphInstanceNameToModel().size(), 1);
  AttrUtils::SetListStr(ge_root_model->GetSubgraphInstanceNameToModel().begin()->second, ge::ATTR_MODEL_OUT_NODES_NAME, {"add"});
  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_NE(model_executor->GetModelDesc().GetOutputDesc(0), nullptr);
  EXPECT_STREQ(model_executor->GetModelDesc().GetOutputDesc(0)->GetName(), "add:0");
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(GraphExecutorWithKernelUnitTest, ExecuteModel_HostInput) {
  const auto back_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = back_options;
  new_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
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
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
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
  runtime_stub.Clear();
  rtStreamDestroy(stream);
  mem_block->Free();
  ge::GetThreadLocalContext().SetGlobalOption(back_options);
}

TEST_F(GraphExecutorWithKernelUnitTest, ExecuteModel_BinaryKernel) {
  auto graph = ShareGraph::BinaryKernelTypicalGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo" || node->GetType() == "Bar") {
      MockLessImportantNodeKernel(node);
    } else if (node->GetType() == "ConditionCalc") {
      ge::AttrUtils::SetStr(node->GetOpDesc(), "cond_func", "st_cond_func");
      KernelRegistry::GetInstance().RegisterKernel("st_cond_func", {{.run_func =
                                                                         [](KernelContext *context) {
                                                                           *context->GetOutputPointer<int32_t>(0) = 1;
                                                                           return ge::GRAPH_SUCCESS;
                                                                         }}, ""});
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"if"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(sizeof(uint32_t));
  memset_s(const_cast<void *>(mem_block->GetAddr()), mem_block->GetSize(), 0, 2048 * 2);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({}, 1, const_cast<void *>(mem_block->GetAddr()));

  auto i0 = FakeValue<Tensor>(Tensor{
      {{}, {}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnHost, ge::DT_FLOAT16, const_cast<void *>(mem_block->GetAddr())});
  auto i1 = FakeValue<Tensor>(Tensor{
      {{}, {}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnHost, ge::DT_FLOAT16, const_cast<void *>(mem_block->GetAddr())});
  auto i2 = FakeValue<Tensor>(Tensor{
      {{}, {}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnHost, ge::DT_FLOAT16, const_cast<void *>(mem_block->GetAddr())});

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
  rtStreamDestroy(stream);
  mem_block->Free();
}

TEST_F(GraphExecutorWithKernelUnitTest, Lowering_Execute_Model_On_UB_fusion_node) {
  auto graph = ShareGraph::BuildGraphWithUBFusionNode();
  graph->TopologicalSorting();

  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker(AddStubName).WithHandle())
                           .AddTaskDef("Mul", AiCoreTaskDefFaker(MulStubName))
                           .BuildGeRootModel();
  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  ge::DumpGraph(exe_graph.get(), "E2EUBFusionGraph");
  // ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(1024UL);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2048}, 1, mem_block->GetAddr());

  auto i0 = FakeValue<Tensor>(
      Tensor{{{2}, {2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(
      Tensor{{{2}, {2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, mem_block->GetAddr()});
  auto i2 = FakeValue<Tensor>(
      Tensor{{{3}, {3}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, mem_block->GetAddr()});

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

  // input_0 shape [2]  input_1 shape [2]
  // input_2 shape [3]
  // expect output shape[(2+2)*3]
  ASSERT_EQ(outputs.at(0).GetShape().GetStorageShape().GetDim(0), 6);  // check output shape
  rtStreamDestroy(stream);
  mem_block->Free();
}

TEST_F(GraphExecutorWithKernelUnitTest, StaticGraph_LoadUnloadSuccess) {
  auto graph = ShareGraph::AicoreStaticGraph();
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(GraphExecutorWithKernelUnitTest, StaticGraphWithAtomicClean_LoadUnloadSuccess) {
  auto graph = ShareGraph::AicoreStaticGraph(true);
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin"))
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin"))
                           .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

/**
 * 用例描述：构造包含纯const的静态子图且静态子图的输出连接到placement为device的节点，不会产生H2D拷贝
 * 预置条件：
 * 1. 构造包含纯const的静态子图且静态子图，静态子图中有1个const节点，动态图中有1个const节点
 * 2. lowering、加载计算图
 *
 * 预期结果
 * 1、lowering时纯const静态子图会展开，不会产生CopyH2D拷贝
 * 2、const节点会通过CreateMemAssigner申请1次内存，执行两次SinkWeightData拷贝内存到device
 */
TEST_F(GraphExecutorWithKernelUnitTest, AllConstKnownSubgraph_LoadUnloadSuccess) {
  auto compute_graph = ShareGraph::BuildWithAllConstKnownSubgraph2();
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();

  auto ge_root_model = GeModelBuilder(compute_graph).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  ess->Clear();
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  ASSERT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Const", "CopyH2D"), 0);
  ASSERT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Const", "CreateMemAssigner"), 1);
  ASSERT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Const", "SinkWeightData"), 2);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeAiCpuTf_ExecuteSuccess) {
  auto graph = ShareGraph::Aicpu4thGraph();
  (void)ge::AttrUtils::SetBool(graph, "_single_op_scene", true);
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  graph->FindNode("add2")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048, 1, 1, 1}, 1);
  auto inputs = FakeTensors({2048, 1, 1, 1}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
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

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeAiCpuCC_ExecuteSuccess) {
  auto graph = ShareGraph::Aicpu4thGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048, 1, 1, 1}, 1);
  auto inputs = FakeTensors({2048, 1, 1, 1}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeHostAiCpu_ExecuteSuccess) {
  char_t opp_path[MMPA_MAX_PATH] = "./";
  mmSetEnv("ASCEND_OPP_PATH", &opp_path[0U], MMPA_MAX_PATH);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<HostcpuMockMmpa>());

  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048);
  auto outputs = FakeTensors({2048}, 1, mem_block->GetAddr());

  auto i0 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     mem_block->GetAddr()});
  auto inputs = std::vector<Tensor *>({i0.holder.get(), i1.holder.get()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  mem_block->Free();
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeHostKernel_ExecuteSuccess) {
  char_t opp_path[MMPA_MAX_PATH] = "./";
  mmSetEnv("ASCEND_OPP_PATH", &opp_path[0U], MMPA_MAX_PATH);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<HostcpuMockMmpa>());

  REGISTER_KERNEL(AddHostKernel).RunFunc(StubHostKernel);
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048);
  auto outputs = FakeTensors({2048}, 1, mem_block->GetAddr());

  auto i0 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(Tensor{{{2048}, {2048}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     mem_block->GetAddr()});
  auto inputs = std::vector<Tensor *>({i0.holder.get(), i1.holder.get()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  mem_block->Free();
}

void ConstructStringTensor(Tensor *input) {
  auto elem_cnt = input->GetShapeSize();
  uint64_t single_str_size = 64U;
  uint64_t total_size = elem_cnt * (single_str_size + sizeof(ge::StringHead) + 1U);
  uint8_t *addr = reinterpret_cast<uint8_t *>(input->GetAddr());
  uint64_t offset = elem_cnt * sizeof(ge::StringHead);

  for (int64_t i = 0; i < elem_cnt; i++) {
    ge::StringHead *string_head = reinterpret_cast<ge::StringHead *>(addr);
    string_head->len = single_str_size;
    string_head->addr = offset;
    offset += single_str_size + 1U;
    addr += sizeof(ge::StringHead);
  }
  input->SetSize(total_size);
  return;
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleStringNodeAiCpu_ExecuteSuccess) {
  char_t opp_path[MMPA_MAX_PATH] = "./";
  mmSetEnv("ASCEND_OPP_PATH", &opp_path[0U], MMPA_MAX_PATH);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<HostcpuMockMmpa>());

  auto graph = ShareGraph::BuildStringNodeGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2 * 64 * (16 + 64 + 1));
  auto outputs = FakeTensors({64, 2}, 1, mem_block->GetAddr());

  auto i0 = FakeValue<Tensor>(Tensor{
      {{64, 2}, {64, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_STRING, mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(Tensor{
      {{64, 2}, {64, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_STRING, mem_block->GetAddr()});
  auto inputs = std::vector<Tensor *>({i0.holder.get(), i1.holder.get()});

  ConstructStringTensor(i0.holder.get());
  ConstructStringTensor(i1.holder.get());

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
                                    ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  mem_block->Free();
}

TEST_F(GraphExecutorWithKernelUnitTest, StringNodeAiCore_LoadUnloadSuccess) {
  auto graph = ShareGraph::BuildStringNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeDataDependency_ExecuteSuccess) {
  auto graph = ShareGraph::BuildDataDependencySingleOpNodeGraph();
  GertRuntimeStub fakeRuntime;

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  fakeRuntime.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  int32_t shape_value = 24;
  std::unique_ptr<uint8_t[]> host_address(new (std::nothrow) uint8_t[512]);
  memcpy_s((void *)(host_address.get()), 512, (void *)(&shape_value), 4);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs0 = FakeTensors({1, 2, 3, 4}, 1);
  auto inputs1 = FakeTensors({1}, 1, (void *)(host_address.get()));
  auto inputs = std::vector<Tensor *>({inputs0.GetTensorList()[0], inputs1.GetTensorList()[0]});

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
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeIdentityN_MixPlacement_ExecuteSuccess) {
  // build compute graph
  auto graph = ShareGraph::BuildIdentityNGraph();
  graph->TopologicalSorting();

  // convert to exe_graph
  // bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EIdentityGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  // load exe_graph
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  // prepare model inputs
  auto outputs = FakeTensors({2048}, 2);
  outputs.GetTensorList()[0]->MutableTensorData().SetPlacement(kOnHost);
  outputs.GetTensorList()[1]->MutableTensorData().SetPlacement(kOnDeviceHbm);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto host_mem_block = allocator->Malloc(sizeof(uint32_t));
  memset_s(const_cast<void *>(host_mem_block->GetAddr()), host_mem_block->GetSize(), 0, 1024 * 2);

  auto inputs = FakeTensors({2048}, 2, const_cast<void *>(host_mem_block->GetAddr()));

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  // execute exe_graph
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->ExecuteSync(inputs.GetTensorList(), inputs.size(),
                                        reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(outputs.at(0).GetShape().GetStorageShape().GetDim(0), 1);  // check output shape
  ASSERT_EQ(outputs.at(1).GetShape().GetStorageShape().GetDim(0),
            2048);  // todo 原来是1024，但是从流程上看应该是2048，根鑫鑫确认下原因
  ASSERT_EQ(outputs.at(0).GetPlacement(), kOnHost);       // check output placement
  ASSERT_EQ(outputs.at(1).GetPlacement(), kOnDeviceHbm);  // check output placement
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  host_mem_block->Free();
}

TEST_F(GraphExecutorWithKernelUnitTest, Test_Control_Edge_Execute_Order_Success) {
  // build compute graph
  auto graph = ShareGraph::BuildNoOpGraph();
  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();
  // convert to exe_graph
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "NoOpGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  // load exe_graph
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  // prepare model inputs
  auto outputs = FakeTensors({2048}, 2);
  outputs.GetTensorList()[0]->MutableTensorData().SetPlacement(kOnHost);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto host_mem_block = allocator->Malloc(sizeof(uint32_t));
  memset_s(const_cast<void *>(host_mem_block->GetAddr()), host_mem_block->GetSize(), 0, 1024 * 2);

  auto inputs = FakeTensors({2048}, 4, const_cast<void *>(host_mem_block->GetAddr()));
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  // execute exe graph
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  std::string kernel_type = "LaunchKernelWithHandle";
  EXPECT_GT(ess->GetExecuteIndexByNodeNameAndKernelType("add1", kernel_type),
            ess->GetExecuteIndexByNodeNameAndKernelType("add2", kernel_type));
  ASSERT_EQ(model_executor->ExecuteSync(inputs.GetTensorList(), inputs.size(),
                                        reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_GT(ess->GetExecuteIndexByNodeNameAndKernelType("add1", kernel_type),
            ess->GetExecuteIndexByNodeNameAndKernelType("add2", kernel_type));

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  host_mem_block->Free();
}

TEST_F(GraphExecutorWithKernelUnitTest, ThirdAicpuOp_ExecuteSuccess) {
  auto graph = ShareGraph::ThirdAicpuOpGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  graph->FindNode("nonzero")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true))
                           .AddTaskDef("NonZero", aicpu_task_def_faker.SetNeedMemcpy(true))
                           .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  void *device_block = (void *)0x01;
  auto outputs = FakeTensors({2}, 1, device_block);

  auto i0 = FakeValue<Tensor>(Tensor{{{2048, 2, 3, 4}, {2048, 2, 3, 4}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     device_block});
  auto i1 = FakeValue<Tensor>(Tensor{{{2048, 2, 3, 4}, {2048, 2, 3, 4}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     device_block});
  auto inputs = std::vector<Tensor *>({i0.holder.get(), i1.holder.get()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("NonZero", "InferShapeRange"), 1);
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("NonZero", "CreateTensorRangesAndShapeRanges"), 1);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  bg::ShapeRangeInferenceResult::ErrorResult();
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, DataFlowOnMainRootLast_ExecuteSuccess) {
  char_t opp_path[MMPA_MAX_PATH] = "./";
  mmSetEnv("ASCEND_OPP_PATH", &opp_path[0U], MMPA_MAX_PATH);
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<HostcpuMockMmpa>());

  auto graph = ShareGraph::BuildHostCpuDataFlowGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameHostCpu.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker)
                              .AddTaskDef("SequenceStub", aicpu_task_def_faker)
                              .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  // check the topo order is correct
  auto ensure_node =
    ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "EnsureTensorAtOutMemory");
  ASSERT_EQ(ensure_node.size(), 1U);
  auto clear_node =
    ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "ClearTestStepContainer");
  ASSERT_EQ(clear_node.size(), 1U);
  EXPECT_TRUE(FastNodeTopoChecker(ensure_node.at(0)).BeforeInTopoOrder(clear_node.at(0)));

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048UL);
  auto outputs = FakeTensors({2}, 1, mem_block->GetAddr());

  auto i0 = FakeValue<Tensor>(Tensor{{{1, 2, 3, 4}, {1, 2, 3, 4}},
                              {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                              kOnDeviceHbm,
                              ge::DT_FLOAT,
                              mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(Tensor{{{1, 2, 3, 4}, {1, 2, 3, 4}},
                              {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                              kOnDeviceHbm,
                              ge::DT_FLOAT,
                              mem_block->GetAddr()});
  auto inputs = std::vector<Tensor *>({i0.holder.get(), i1.holder.get()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  uint32_t model_exec_count = 0;
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
                                    ge::GRAPH_SUCCESS);
  model_exec_count++;
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("SequenceStub", "SequenceStub"), model_exec_count * 2);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
                                    ge::GRAPH_SUCCESS);
  model_exec_count++;
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("SequenceStub", "SequenceStub"), model_exec_count * 2);
  std::vector<std::string> stat_exec_nodes_count = GetStatExecNodesCount();
  // ClearTestStepContainer execute once when main graph execute
  EXPECT_EQ(std::count(stat_exec_nodes_count.begin(), stat_exec_nodes_count.end(), "ClearTestStepContainer"), model_exec_count);
  EXPECT_EQ(std::count(stat_exec_nodes_count.begin(), stat_exec_nodes_count.end(), "CreateTestStepContainer"), 1);
  EXPECT_EQ(std::count(stat_exec_nodes_count.begin(), stat_exec_nodes_count.end(), "DestroyTestStepContainer"), 0);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  stat_exec_nodes_count = GetStatExecNodesCount();
  // DestroyTestStepContainer execute once for each model
  EXPECT_EQ(std::count(stat_exec_nodes_count.begin(), stat_exec_nodes_count.end(), "DestroyTestStepContainer"), 1);
  stat_exec_nodes_count.clear();
  rtStreamDestroy(stream);
  mem_block->Free();
  FreeDataFlowMemory();
}

TEST_F(GraphExecutorWithKernelUnitTest, ZeroTensorInput_ExecuteSuccess) {
  auto graph = ShareGraph::BuildZeroInputAicoreGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("ReduceSum", AiCoreTaskDefFaker("ReduceSumStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);

  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto input0 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT16).Format(ge::FORMAT_ND).Shape({}).Build();
  auto input1 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Format(ge::FORMAT_ND).Shape({0}).Build();
  auto inputs = std::vector<Tensor *>({input0.GetTensor(), input1.GetTensor()});
  auto outputs = FakeTensors({}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, RtsOverflowDetection_ExecuteSuccess) {
  auto graph = ShareGraph::AicoreWithRtsOverflowGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("ReduceSum", AiCoreTaskDefFaker("ReduceSumStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);

  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EReduceSumGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

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
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
  reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, RtsDebugOverflowDetection_ExecuteSuccess) {
  auto graph = ShareGraph::AicoreWithRtsDebugOverflowGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model =
      builder.AddTaskDef("ReduceSum", AiCoreTaskDefFaker("ReduceSumStubBin").WithHandle()).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);

  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EReduceSumGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

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
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, Cmo_ExecuteSuccess) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  auto graph = ShareGraph::AicoreWithCmoGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("ReduceSum", AiCoreTaskDefFaker("ReduceSumStubBin").WithHandle())
                           .BuildGeRootModel();
  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);

  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2ECmoGraph");

  EXPECT_EQ(ge_root_model->GetNodesToTaskDef().size(), 1UL);
  ge_root_model->GetNodesToTaskDef().size();
  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto input0 =
      TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT16).Format(ge::FORMAT_ND).Shape({32, 64}).Build();
  auto input1 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Format(ge::FORMAT_ND).Shape({0}).Build();
  auto inputs = std::vector<Tensor *>({input0.GetTensor(), input1.GetTensor()});
  auto outputs = FakeTensors({2048}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/**
* 用例描述：同一个model可以加载多次
*
* 预置条件：NA
*
* 测试步骤：
* 1. 构造带静态子图和动态子图的计算图及root model
* 2. lowering、加载计算图
* 3. 构造输入Tensor执行
* 4. 再次加载执行该ge root model
*
* 预期结果：
* 1. 第二次加载执行执行成功
*/
TEST_F(GraphExecutorWithKernelUnitTest, OnolineLoadModelWithSameRootGraphTwice) {
  auto graph = ShareGraph::BuildDynamicAndStaticGraph();
  graph->TopologicalSorting();
  auto ge_root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();

  // malloc device mem
  size_t weight_size = 10U;  // require 8
  void *weight_mem = nullptr;
  auto rt_ret = rtMalloc(&weight_mem, weight_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  ASSERT_EQ(rt_ret, RT_ERROR_NONE);
  ge::ModelData model_data = model_data_holder.Get();
  GELOGI("alloc weight size[%zu], addr[%p]", weight_size, weight_mem);

  // load and execute first time
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  auto executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(executor, nullptr);

  uint64_t session_id = 100U;
  auto session = gert::RtSession(session_id);
  gert::OuterWeightMem weight = {weight_mem, weight_size};
  ModelExecuteArg arg = {(void *)2};
  ModelLoadArg load_arg(&session, weight);
  ge::graphStatus ret = executor->Load(&arg, load_arg);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2, 2}, 2);
  auto inputs = FakeTensors({2, 2}, 2);

  ret = executor->ExecuteSync(inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size());
  executor->UnLoad();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  // load and execute twice
  auto exe_graph2 = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph2, nullptr);
  auto executor2 = ModelV2Executor::Create(exe_graph2, ge_root_model);
  ASSERT_NE(executor2, nullptr);

  ret = executor2->Load(&arg, load_arg);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ret = executor2->ExecuteSync(inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size());
  executor2->UnLoad();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  rt_ret = rtFree(weight_mem);
  ASSERT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(GraphExecutorWithKernelUnitTest, BlockAiCpuTf_ExecuteSuccess) {
  auto graph = ShareGraph::BuildBlockGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EBlockAddOpGraph");
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({1, 1}, 1);
  auto inputs = FakeTensors({1, 1}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex("Launch aicpu inputs/outputs sizes") != -1);
  runtime_stub.Clear();
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, LowerFromGeModelWithNullTaskDef) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  ge_root_model->SetFlattenGraph(nullptr);
  auto name_to_model = ge_root_model->GetSubgraphInstanceNameToModel();
  name_to_model["test"]->SetModelTaskDef(nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_EQ(exe_graph, nullptr);
}

TEST_F(GraphExecutorWithKernelUnitTest, SingleNodeReshape_rtStreamSynchronize_timeout_Failed) {
  const char_t *const kTimeoutEnvPath = "TIMEOUT";
  char_t record_path[MMPA_MAX_PATH] = "timeout";
  mmSetEnv(kTimeoutEnvPath, &record_path[0U], MMPA_MAX_PATH);
  ge::GetContext().SetStreamSyncTimeout(15000);

  auto graph = ShareGraph::BuildReshapeGraph();
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EReshapeGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto inputs = FakeTensors({2048}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_NE(model_executor->ExecuteSync(inputs.GetTensorList(), inputs.size(),
                                        reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  unsetenv(kTimeoutEnvPath);
}
uint32_t g_launch_flag = 0U;
graphStatus LaunchKernelFailedByLaunchFlagFake(gert::KernelContext *context) {
  if (g_launch_flag++ == 0U) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
/**
 * 用例描述：构造Topological执行器, 第一次执行时有kernel执行失败但第二次执行时kernel执行成功的场景
 * 预置条件：
 * 执行时走Topological执行器的条件是：执行图中有控制结点且launch节点数小于5
 * 1. 构图包含if控制算子，且launch节点总数小于5
 * 2. aicore算子执行桩，第一次执行时执行失败后续执行成功
 * 3. 输入value构造
 *
 * 预期结果
 * 1. 第一次执行失败，第二次执行成功
 */
TEST_F(GraphExecutorWithKernelUnitTest, TopologicalExecuteFailThenSuccess) {
  auto graph = ShareGraph::IfCondByShapeGraph();
  graph->TopologicalSorting();
  const char* const Cast = "Cast";
  auto ge_root_model = GeModelBuilder(graph).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .AddTaskDef(Cast, AiCoreTaskDefFaker(Cast).WithHandle()).FakeTbeBin({Cast}).BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "IfCondByShapeGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("LaunchKernelWithHandle", LaunchKernelFailedByLaunchFlagFake);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT)
      .Format(ge::FORMAT_ND)
      .Shape({2, 3, 4, 6})
      .Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};

  auto i0 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Value<int32_t>({1}).Build();
  auto i1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({2, 3, 4, 6}).Build();
  auto inputs0 = std::vector<Tensor *>({i0.GetTensor(), i1.GetTensor()});

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  // 第一次执行失败
  g_launch_flag = 0U;
  ess->Clear();
  ASSERT_NE(model_executor->Execute({i3.value}, inputs0.data(), inputs0.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  // 执行失败，和失败结点关联的后续launch结点都不会执行
  ASSERT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("add0", "LaunchKernelWithHandle"), 1);
  ASSERT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("cast0", "LaunchKernelWithFlag"), 0);

  // 第二次执行成功
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs0.data(), inputs0.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  // 执行成功，所有launch结点都正常执行
  ASSERT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("add0", "LaunchKernelWithHandle"), 1);
  ASSERT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("cast0", "LaunchKernelWithFlag"), 1);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：构造PriorityTopological执行器,第一次执行时有kernel执行失败第二次执行时kernel执行成功的场景
 * 执行时走PriorityTopological执行器的条件是：执行图中有控制结点且launch节点数大于等于5
 * 预置条件：
 * 1. 构图包含If控制算子,launch结点的数量大于等于5
 * 2. aicore算子执行桩，第一次执行时执行失败后续执行成功
 * 3. 输入value构造
 *
 * 预期结果
 * 1. 第一次执行失败，第二次执行成功
 */
TEST_F(GraphExecutorWithKernelUnitTest, PriorityTopologicalExecuteFailThenSuccess) {
  auto compute_graph = ShareGraph::IfGraph4();
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();
  GE_DUMP(compute_graph, "computegraph_IfGraph4");

  auto ge_root_model = GeModelBuilder(compute_graph).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "exe_graph_IfGraph4");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  g_launch_flag = 0U;
  runtime_stub.GetKernelStub().SetUp("LaunchKernelWithFlag", LaunchKernelFailedByLaunchFlagFake);

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

  // 第一次执行失败
  g_launch_flag = 0U;
  ess->Clear();
  ASSERT_NE(
      model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
      ge::GRAPH_SUCCESS);
  // 执行失败，和失败结点关联的后续launch结点都不会执行
  ASSERT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("add4", "LaunchKernelWithFlag"), 1);
  ASSERT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("add2", "LaunchKernelWithHandle"), 0);
  ASSERT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithHandle"), 2);
  ASSERT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithFlag"), 1);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "KernelTrace") >= 0);
  ASSERT_EQ(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "KernelTrace"),
            runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "LaunchKernelWithFlag"));

  // 第二次执行成功
  ess->Clear();
  ASSERT_EQ(
      model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
      ge::GRAPH_SUCCESS);
  // 执行成功，所有launch结点都正常执行
  ASSERT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("add4", "LaunchKernelWithFlag"), 1);
  ASSERT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("add2", "LaunchKernelWithHandle"), 1);
  ASSERT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithHandle"), 3);
  ASSERT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "LaunchKernelWithFlag"), 1);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex("TilingData: ") != -1);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex("Input/Output sizes:") != -1);
  runtime_stub.Clear();
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithKernelUnitTest, check_model_output_manager_not_null_success) {
  auto graph = ShareGraph::BuildAddUnSqueezeGraph();
  CheckModelOutputManagerIsNull(graph, 2);

  graph = ShareGraph::BuildAddAsTwoOutputGraph();
  CheckModelOutputManagerIsNull(graph, 2);

  graph = ShareGraph::BuildUnsqueezeAsTwoOutputGraph();
  CheckModelOutputManagerIsNull(graph, 1);

  graph = ShareGraph::BuildInputAsTwoOutputGraph();
  CheckModelOutputManagerIsNull(graph, 1);

  graph = ShareGraph::BuildAssignAsTwoOutputGraph();
  CheckModelOutputManagerIsNull(graph, 2, false);

  graph = ShareGraph::BuildAddToUnSqueezeGraph();
  CheckModelOutputManagerIsNull(graph, 2);
}

TEST_F(GraphExecutorWithKernelUnitTest, Test_With_Empty_Tensor_Success) {
  // build compute graph
  auto graph = ShareGraph::BuildNoOpGraph();
  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();
  // convert to exe_graph
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "NoOpGraphWithBgIf");
  auto main_node = ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main");
  ASSERT_NE(main_node, nullptr);
  auto main_graph = FastNodeUtils::GetSubgraphFromNode(main_node, 0);
  ASSERT_NE(main_graph, nullptr);
  FastNode * check_shape_node = ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "CheckOutputShapesEmpty");
  ASSERT_NE(check_shape_node, nullptr);
}

TEST_F(GraphExecutorWithKernelUnitTest, Test_No_Empty_Tensor_Success) {
  const auto back_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  auto new_options = back_options;
  new_options[ge::OPTION_ALL_TENSOR_NOT_EMPTY] = "1";
  ge::GetThreadLocalContext().SetGraphOption(new_options);

  // build compute graph
  auto graph = ShareGraph::BuildNoOpGraph();
  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();
  // convert to exe_graph
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "NoOpGraphWithBgIf");
  auto main_node = ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main");
  ASSERT_NE(main_node, nullptr);
  auto main_graph = FastNodeUtils::GetSubgraphFromNode(main_node, 0);
  ASSERT_NE(main_graph, nullptr);
  FastNode * check_shape_node = ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "CheckOutputShapesEmpty");
  ASSERT_EQ(check_shape_node, nullptr);
}
}  // namespace gert
