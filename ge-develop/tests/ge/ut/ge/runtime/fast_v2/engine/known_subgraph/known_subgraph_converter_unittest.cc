/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/static_compiled_graph_converter.h"

#include <gtest/gtest.h>

#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "lowering/graph_converter.h"
#include "common/topo_checker.h"
#include "register/op_impl_registry.h"
#include "aicore/converter/bg_kernel_launch.h"
#include "common/ge_inner_attrs.h"
#include "common/const_data_helper.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_local_context.h"
#include "stub/gert_runtime_stub.h"
#include "graph/utils/graph_dump_utils.h"

using namespace ge;
using namespace gert::bg;

namespace gert {
namespace {
ge::Status SerializeGeModel(const ge::GeModelPtr &ge_model, ge::ModelBufferData &model_buffer) {
  ge::ModelHelper model_helper;
  std::string dummy_file = "dummy_file";

  model_helper.SetSaveMode(false);
  return model_helper.SaveToOmModel(ge_model, dummy_file, model_buffer);
}
ge::GeModelPtr SerializeAndDeserializeGeModel(ge::GeModel *ge_model) {
  ge::ModelBufferData model_buffer;
  auto ret = SerializeGeModel(ge_model->shared_from_this(), model_buffer);
  if (ret != ge::SUCCESS) {
    return nullptr;
  }

  const auto file_header = reinterpret_cast<ModelFileHeader *>(model_buffer.data.get());
  uint8_t *model_data_addr = reinterpret_cast<uint8_t *>(model_buffer.data.get()) + sizeof(ModelFileHeader);
  uint32_t model_data_size = file_header->model_length;

  ge::OmFileLoadHelper om_load_helper;
  auto status = om_load_helper.Init(model_data_addr, model_data_size, file_header);
  if (status != ge::SUCCESS) {
    GELOGE(status, "om load helper init failed");
    return nullptr;
  }

  ge::GeModelPtr ge_model_ptr;
  ge::GeModelPtr first_ge_model = nullptr;
  ge::ModelHelper model_helper;
  status = model_helper.GenerateGeModel(om_load_helper, ge_model_ptr, first_ge_model, 0U, false);
  if (status != ge::SUCCESS) {
    GELOGE(status, "generate GeModel failed");
    return nullptr;
  }
  return ge_model_ptr;
}

LowerResult LoweringKnownSubgraph(const ge::NodePtr &node, const LowerInput &input) {
  auto graph_name = node->GetOpDesc()->GetSubgraphInstanceName(0U);
  auto root = ge::GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  auto model_graph = root->GetSubgraph(graph_name);
  auto model = input.global_data->GetGraphStaticCompiledModel(graph_name);
  return LoweringStaticModel(model_graph, static_cast<ge::GeModel *>(model), input.input_addrs, *input.global_data);
}

void *FindKnownSubgraphModel(const ge::NodePtr &node, LoweringGlobalData &global_data) {
  auto graph_name = node->GetOpDesc()->GetSubgraphInstanceName(0U);
  return global_data.GetGraphStaticCompiledModel(graph_name);
}

void AddKnownSubgraphModel(const ge::NodePtr &node, LoweringGlobalData &global_data, void *model) {
  auto graph_name = node->GetOpDesc()->GetSubgraphInstanceName(0U);
  global_data.AddStaticCompiledGraphModel(graph_name, model);
}
}  // namespace
class KnownSubgraphNodeConverterUT : public bg::BgTestAutoCreateFrame {};

TEST_F(KnownSubgraphNodeConverterUT, ConvertSuccess) {
  Create3StageFrames();
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);

  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  ASSERT_NE(sub_graph, nullptr);
  auto net_output = sub_graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_output, nullptr);
  ASSERT_NE(net_output->GetOpDesc(), nullptr);

  const auto &tensor_desc0 = net_output->GetOpDesc()->MutableInputDesc(1U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc0, 36);

  const auto &tensor_desc1 = net_output->GetOpDesc()->MutableInputDesc(2U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc1, 72);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("Conv2d", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]}, {data_a_ret.out_addrs[0]}, &global_data};

  auto conv_ret = LoweringKnownSubgraph(parent_node, conv2d_input);
  EXPECT_TRUE(conv_ret.result.IsSuccess());
  ASSERT_EQ(conv_ret.out_addrs.size(), 3);
  EXPECT_EQ(conv_ret.out_shapes.size(), 3);
  EXPECT_EQ(conv_ret.order_holders.size(), 1);

  // output[1] reuse input[0]
  EXPECT_EQ(conv_ret.out_addrs[1], data_a_ret.out_addrs[0]);
}

TEST_F(KnownSubgraphNodeConverterUT, ConvertSuccess_ZeroOutput) {
  Create3StageFrames();
  auto graph = ShareGraph::BuildWithKnownSubgraph(true);
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("Conv2d", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  EXPECT_TRUE(data_a_ret.result.IsSuccess());
  EXPECT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]}, {data_a_ret.out_addrs[0]}, &global_data};

  auto conv_ret = LoweringKnownSubgraph(parent_node, conv2d_input);
  EXPECT_TRUE(conv_ret.result.IsSuccess());
  EXPECT_EQ(conv_ret.out_addrs.size(), 0);
  EXPECT_EQ(conv_ret.out_shapes.size(), 0);
  EXPECT_EQ(conv_ret.order_holders.size(), 1);
}

TEST_F(KnownSubgraphNodeConverterUT, ConvertSuccess_MallocWorkspaceMemorySuccess_WithP2pFeatureMemory) {
  Create3StageFrames();
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("Conv2d", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]}, {data_a_ret.out_addrs[0]}, &global_data};

  // GeModel with not supported memory type
  auto ge_model = reinterpret_cast<ge::GeModel *>(FindKnownSubgraphModel(parent_node, global_data));

  ge::AttrUtils::SetInt(*ge_model, ge::ATTR_MODEL_P2P_MEMORY_SIZE, 2);

  auto conv_ret = LoweringKnownSubgraph(parent_node, conv2d_input);
  EXPECT_TRUE(conv_ret.result.IsSuccess());

  // clear memory type not supported
  ge::AttrUtils::SetInt(*ge_model, ge::ATTR_MODEL_P2P_MEMORY_SIZE, 0);
}

TEST_F(KnownSubgraphNodeConverterUT, SerializeAndDeserializeGeModelTwiceLoweringSuccess) {
  Create3StageFrames();
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("Conv2d", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]}, {data_a_ret.out_addrs[0]}, &global_data};

  auto ge_model = reinterpret_cast<ge::GeModel *>(FindKnownSubgraphModel(parent_node, global_data));
  ASSERT_NE(ge_model->GetTBEKernelStore().Data(), nullptr);

  // after serialize and deserialize, tbe kernel data is null
  auto ge_model_ptr = SerializeAndDeserializeGeModel(ge_model);
  ASSERT_EQ(ge_model_ptr->GetTBEKernelStore().Data(), nullptr);

  AddKnownSubgraphModel(parent_node, global_data, &(*ge_model_ptr));

  auto conv_ret = LoweringKnownSubgraph(parent_node, conv2d_input);
  EXPECT_TRUE(conv_ret.result.IsSuccess());
}

TEST_F(KnownSubgraphNodeConverterUT, SetSessionToGeModelSuccess) {
  Create3StageFrames();
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("Conv2d", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]}, {data_a_ret.out_addrs[0]}, &global_data};
  auto conv_ret = LoweringKnownSubgraph(parent_node, conv2d_input);
  EXPECT_TRUE(conv_ret.result.IsSuccess());
  auto main_exe_graph = bg::ValueHolder::PopGraphFrame()->GetExecuteGraph();
  auto root_graph = bg::ValueHolder::PopGraphFrame()->GetExecuteGraph();
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto create_davinci_model = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreate");
  ASSERT_NE(create_davinci_model, nullptr);
  auto session_input_node = create_davinci_model->GetInDataNodes().at(1);
  ASSERT_EQ(session_input_node->GetType(), "GetSessionId");  // GetSessionId在init图
}

TEST_F(KnownSubgraphNodeConverterUT, LowerGraphWithStaticPartitonedCall) {
  auto graph = ShareGraph::SimpleStaticPartitionedCallGraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("StaticFoo", false).Build();
  graph->TopologicalSorting();
  bg::ValueHolder::PopGraphFrame();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "static_graph");

  auto sub_graphs = exe_graph->GetAllSubgraphs();
  EXPECT_EQ(sub_graphs.size(), 3U);
  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ASSERT_NE(main_graph, nullptr);

  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreate"), nullptr);

  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "DavinciModelUpdateWorkspaces"), nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "AllocModelOutTensor"), nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "EnsureTensorAtOutMemory"), nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "DavinciModelExecute"), nullptr);
}

TEST_F(KnownSubgraphNodeConverterUT, LoweringStaticCompiledComputeGraph) {
  auto graph1 = ShareGraph::SimpleStaticPartitionedCallGraph();
  auto root_model = GeModelBuilder(graph1).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("StaticFoo", false).Build();
  auto graph = ShareGraph::SimpleStaticGraph();
  graph->SetName("static");
  graph->TopologicalSorting();
  bg::ValueHolder::PopGraphFrame();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "static_graph");
  auto sub_graphs = exe_graph->GetAllSubgraphs();
  EXPECT_EQ(sub_graphs.size(), 3U);
  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ASSERT_NE(main_graph, nullptr);

  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreate"), nullptr);

  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "DavinciModelUpdateWorkspaces"), nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "AllocModelOutTensor"), nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "EnsureTensorAtOutMemory"), nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "DavinciModelExecute"), nullptr);
}
TEST_F(KnownSubgraphNodeConverterUT, UpdateNodeCemToInit_KnownShapeInIfSubgraph) {
  auto graph = ShareGraph::IfWithKnownShapeSubGraph();
  graph->SetName("static");
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("StaticFoo", false).Build();

  bg::ValueHolder::PopGraphFrame();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph =
      GraphConverter().SetModelDescHolder(&model_desc_holder).ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "KnownSubgraphInIfUT");
  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ASSERT_NE(main_graph, nullptr);

  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreate"), nullptr);

  // todo 当前还不支持到if里面去做零拷贝融合，因此网络的结尾是 EnsureTensorAtOutMemory
  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "EnsureTensorAtOutMemory"), nullptr);
}

TEST_F(KnownSubgraphNodeConverterUT, KnownSubgraph_WithGlobalWorkspace_ConvertSuccess) {
  ge::GetThreadLocalContext().SetGraphOption({{"ge.exec.static_model_addr_fixed", "1"}});
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ASSERT_NE(main_graph, nullptr);

  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreateV2"), nullptr);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "DavinciModelUpdateWorkspaces"), nullptr);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  ge::GetThreadLocalContext().SetGraphOption({{}});
}

TEST_F(KnownSubgraphNodeConverterUT, KnownSubgraph_WithFixedFeatureMem_ConvertSuccess) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  uint64_t mem;
  ge_root_model->MutableFixedFeatureMemory().insert(
      {RT_MEMORY_HBM, {RT_MEMORY_HBM, &mem, sizeof(mem), true, false, false, 0U, nullptr}});

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ASSERT_NE(main_graph, nullptr);

  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreate"), nullptr);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  ge::GetThreadLocalContext().SetGraphOption({{}});
}

size_t GetNodeTypeCount(const ge::ExecuteGraph *const graph, std::string str) {
  size_t cnt = 0UL;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == str) {
      ++cnt;
    }
  }
  return cnt;
}
/*
 * 1 需要p2p fixed内存，用户指定nullptr, size 0， 不需要在init图中申请
 * 2 hbm没有这个类型的fixed内存， 不需要在init图中申请
 */
TEST_F(KnownSubgraphNodeConverterUT, KnownSubgraph_WithFixedFeatureMemory_AddrNullptrSizeZero) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  ge_root_model->MutableFixedFeatureMemory().insert(
      {RT_MEMORY_P2P_DDR, {RT_MEMORY_P2P_DDR, nullptr, 0U, true, false, false, 0U, nullptr}});

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ASSERT_NE(main_graph, nullptr);

  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreate"), nullptr);
  EXPECT_EQ(GetNodeTypeCount(init_graph, "EmptyTensorData"), 2U);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  ge::GetThreadLocalContext().SetGraphOption({{}});
}

/*
 * 1 需要p2p fixed内存，用户没有指定，需要在init图中申请
 * 2 hbm没有这个类型的fixed内存， 不需要在init图中申请
 */
TEST_F(KnownSubgraphNodeConverterUT, KnownSubgraph_WithFixedFeatureMemory_AddrNullptrSizeNotZero) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  for (const auto &name_to_model : ge_root_model->GetSubgraphInstanceNameToModel()) {
    const auto &ge_model_temp = name_to_model.second;

    size_t p2p_size = 64;
    (void)AttrUtils::SetInt(ge_model_temp, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_size);
  }

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ASSERT_NE(main_graph, nullptr);

  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreate"), nullptr);
  EXPECT_EQ(GetNodeTypeCount(init_graph, "EmptyTensorData"), 1U);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  ge::GetThreadLocalContext().SetGraphOption({{}});
}

/*
 * 1 需要p2p fixed内存，用户没有指定，需要在init图中申请
 * 2 hbm没有这个类型的fixed内存， 不需要在init图中申请
 */
TEST_F(KnownSubgraphNodeConverterUT, KnownSubgraph_WithFixedFeatureMemory_UseFixedBaseExpandableAllocator) {
  mmSetEnv("GE_USE_STATIC_MEMORY", "4", 1);
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  for (const auto &name_to_model : ge_root_model->GetSubgraphInstanceNameToModel()) {
    const auto &ge_model_temp = name_to_model.second;

    size_t p2p_size = 64;
    (void)AttrUtils::SetInt(ge_model_temp, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_size);
  }

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto deinit_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "DeInit"), 0);
  ASSERT_NE(deinit_graph, nullptr);

  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "GetUserAllocatorOrFixedBaseAllocator"), nullptr);
  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "AllocFixedFeatureMemory"), nullptr);
  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(deinit_graph, "FreeFixedFeatureMemory"), nullptr);
  EXPECT_EQ(GetNodeTypeCount(init_graph, "EmptyTensorData"), 1U);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  ge::GetThreadLocalContext().SetGraphOption({{}});
  mmSetEnv("GE_USE_STATIC_MEMORY", "0", 1);
}

/*
 * 1 hbm 用户指定了地址， 不需要在init图中申请
 * 2 p2p 没有这个类型的fixed内存， 不需要在init图中申请
 */
TEST_F(KnownSubgraphNodeConverterUT, KnownSubgraph_WithFixedFeatureMemory_AddrNotNullptrSizeNotZero) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  int8_t mem[64];
  ge_root_model->MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, &mem, 64, true, false, false, 0U, nullptr}});

  for (const auto &name_to_model : ge_root_model->GetSubgraphInstanceNameToModel()) {
    const auto &ge_model_temp = name_to_model.second;
    uint64_t mem = 0UL;
    std::vector<std::vector<int64_t>> sub_mem_infos;
    std::vector<int64_t> sub_mem_offset;
    sub_mem_offset.emplace_back(0x2U);             // mem_type RT_MEMORY_HBM 0x2U
    sub_mem_offset.emplace_back((int64_t)(&mem));  // mem_offset_base
    sub_mem_offset.emplace_back(64);               // mem_size
    sub_mem_offset.emplace_back(1UL);              // is_fixed_addr_prior
    sub_mem_infos.emplace_back(sub_mem_offset);
    AttrUtils::SetListListInt(ge_model_temp, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_infos);
  }

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ASSERT_NE(main_graph, nullptr);

  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreate"), nullptr);
  EXPECT_EQ(GetNodeTypeCount(init_graph, "EmptyTensorData"), 2U);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  ge::GetThreadLocalContext().SetGraphOption({{}});
}

TEST_F(KnownSubgraphNodeConverterUT, KnownSubgraph_UserSetFileConstantDeviceMem) {
  auto graph = ShareGraph::BuildWithKnownSubgraph(false, true);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  int8_t mem[64];
  ge_root_model->MutableFixedFeatureMemory().insert({RT_MEMORY_HBM, {RT_MEMORY_HBM, &mem, 64, true, false, false, 0U, nullptr}});

  for (const auto &name_to_model : ge_root_model->GetSubgraphInstanceNameToModel()) {
    const auto &ge_model_temp = name_to_model.second;
    uint64_t mem = 0UL;
    std::vector<std::vector<int64_t>> sub_mem_infos;
    std::vector<int64_t> sub_mem_offset;
    sub_mem_offset.emplace_back(0x2U);             // mem_type RT_MEMORY_HBM 0x2U
    sub_mem_offset.emplace_back((int64_t)(&mem));  // mem_offset_base
    sub_mem_offset.emplace_back(64);               // mem_size
    sub_mem_offset.emplace_back(1UL);              // is_fixed_addr_prior
    sub_mem_infos.emplace_back(sub_mem_offset);
    AttrUtils::SetListListInt(ge_model_temp, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_infos);
  }

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  std::vector<ge::FileConstantMem> file_constant_mems;
  file_constant_mems.emplace_back(ge::FileConstantMem{"file1", (void *)0x100, 2566});
  file_constant_mems.emplace_back(ge::FileConstantMem{"file1.bin", (void *)0x100, 2566});
  gert::ModelConverter::Args args({}, nullptr, nullptr, nullptr, &file_constant_mems);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevel(DLOG_INFO);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  auto log_ret = runtime_stub.GetSlogStub().FindLog(DLOG_INFO, "FileConstant memory vector size: 2");
  EXPECT_NE(log_ret, -1);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreate"), nullptr);

  runtime_stub.GetSlogStub().SetLevel(DLOG_ERROR);
  runtime_stub.GetKernelStub().StubTiling();

  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  ge::GetThreadLocalContext().SetGraphOption({{}});
}

TEST_F(KnownSubgraphNodeConverterUT, KnownSubgraph_NoNeedMallocOutputMem_RefNodeInsideSubgraph) {
  ge::GetThreadLocalContext().SetGraphOption({{"ge.exec.static_model_addr_fixed", "1"}});
  auto graph = ShareGraph::BuildWithKnownSubgraphWithRefNode();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Assign", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  ASSERT_NE(main_graph, nullptr);

  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "DavinciModelCreateV2"), nullptr);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "DavinciModelUpdateWorkspaces"), nullptr);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "AllocModelOutTensor"), nullptr);
  auto davinci_model_execute_kernel = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "DavinciModelExecute");
  EXPECT_NE(davinci_model_execute_kernel, nullptr);

  EXPECT_EQ(FastNodeTopoChecker(davinci_model_execute_kernel)
                .StrictConnectFrom({{"InnerData", 0},
                                    {"SplitRtStreams", 0},
                                    {"InnerData", 0},
                                    {"InnerData", 0},
                                    {"MakeSureTensorAtDevice", 0},
                                    {"MakeSureTensorAtDevice", 0}}),  // output is from input
            "success");
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  ge::GetThreadLocalContext().SetGraphOption({{}});
}
}  // namespace gert