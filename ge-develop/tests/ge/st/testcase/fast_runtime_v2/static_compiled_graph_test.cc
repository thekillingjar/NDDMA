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
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_dump_utils.h"
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
#include "register/op_impl_registry.h"

#include "securec.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"

#include "check/executor_statistician.h"
#include "graph_builder/bg_infer_shape_range.h"
#include "common/helper/model_parser_base.h"
#include "graph/manager/graph_var_manager.h"
#include "faker/rt2_var_manager_faker.h"
#include "graph/manager/memory_manager.h"
#include "graph/manager/mem_manager.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_graph_dsl/graph_dsl.h"

using namespace ge;
namespace gert {
class StaticCompiledGraphTest : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
  void TearDown() override {
    Test::TearDown();
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
  }

  GeTensor CreateVecTorGeTensor(std::vector<int64_t> shape, DataType type) {
    uint32_t shape_size = 1;
    for (auto dim : shape) {
      shape_size *= dim;
    }
    const auto data_size = ge::GetSizeInBytes(shape_size, type);
    return GeTensor(GeTensorDesc(GeShape(shape), FORMAT_ND, type), reinterpret_cast<uint8_t *>(shape.data()), data_size);
  }

  void SetSubGraph(ComputeGraphPtr &parent_graph, NodePtr &parent_node, ComputeGraphPtr &sub_graph) {
    parent_node->GetOpDesc()->RegisterSubgraphIrName("f", SubgraphType::kStatic);

    size_t index = parent_node->GetOpDesc()->GetSubgraphInstanceNames().size();
    parent_node->GetOpDesc()->AddSubgraphName(sub_graph->GetName());
    parent_node->GetOpDesc()->SetSubgraphInstanceName(index, sub_graph->GetName());

    sub_graph->SetParentNode(parent_node);
    sub_graph->SetParentGraph(parent_graph);
    parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  }

  ge::ComputeGraphPtr BuildWithKnownSubgraphWithAddNode() {
    std::vector<int64_t> shape = {2, 2};           // NCHW
    std::vector<int64_t> unknown_shape = {2, -1};  // NCHW

    int64_t shape_size;
    TensorUtils::CalcTensorMemSize(GeShape(shape), FORMAT_NCHW, DT_FLOAT, shape_size);
    auto data1 = OP_CFG("Data")
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                     .InCnt(1)
                     .OutCnt(1)
                     .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                     .Attr(ge::ATTR_NAME_INDEX, (int32_t)0)
                     .Build("data1");
    data1->SetOutputOffset({0});
    TensorUtils::SetSize(*data1->MutableOutputDesc(0), shape_size);

    auto const1 = OP_CFG("Const")
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("const1");
    ge::AttrUtils::SetTensor(const1, "value", CreateVecTorGeTensor(shape, DT_FLOAT16));
    const1->SetOutputOffset({1024});
    TensorUtils::SetSize(*const1->MutableOutputDesc(0), shape_size);

    vector<int64_t> test_int64_list_attr = {1, 2, 3};
    auto conv2d = OP_CFG("Conv2d")
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr("string_attr", "test")
                      .Attr("int32_attr", (int32_t)1)
                      .Attr("uint32_attr", (uint32_t)1)
                      .Attr("data_format", "NHWC")  // attr on operator
                      .Attr("dilations", test_int64_list_attr)
                      .Attr("groups", (int32_t)1)
                      .Attr("offset_x", (int32_t)1)
                      .Attr("_kernel_bin_id", "te_conv2d_12345")
                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                      .Attr(ATTR_NAME_KERNEL_BIN_ID, "conv2d_fake_id")
                      .Build("conv2d");
    conv2d->SetOpEngineName("AIcoreEngine");
    conv2d->SetOpKernelLibName("AIcoreEngine");
    conv2d->SetInputOffset({0});
    conv2d->SetOutputOffset({16});

    auto relu = OP_CFG("Relu").TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape).InCnt(1).OutCnt(1).Build("relu");
    relu->SetInputOffset({16});
    relu->SetOutputOffset({48});
    TensorUtils::SetSize(*relu->MutableOutputDesc(0), shape_size);

    relu->SetOpEngineName("AIcoreEngine");
    relu->SetOpKernelLibName("AIcoreEngine");

    auto netoutput_sub = OP_CFG("NetOutput")
                             .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                             .InCnt(3)
                             .OutCnt(1)
                             .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                             .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                             .Build("netoutput_sub");
    netoutput_sub->SetSrcName({"relu", "data1", "const1"});
    netoutput_sub->SetSrcIndex({0, 0, 0});
    netoutput_sub->SetInputOffset({48, 0, 1024});

    DEF_GRAPH(g2) {
      CHAIN(NODE(data1)->NODE(conv2d)->NODE(relu)->EDGE(0, 0)->NODE(netoutput_sub));
      CHAIN(NODE(data1)->EDGE(0, 1)->NODE(netoutput_sub));
      CHAIN(NODE(const1)->EDGE(0, 2)->NODE(netoutput_sub));
    };

    auto data_a = OP_CFG("Data")
                      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                      .InCnt(1)
                      .OutCnt(1)
                      .Attr(ATTR_NAME_INDEX, 0)
                      .Build("data_a");

    auto known_subgraph = OP_CFG(ge::PARTITIONEDCALL)
                              .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                              .InCnt(1)
                              .OutCnt(3)
                              .Build(ge::PARTITIONEDCALL);

    auto add = OP_CFG("Add")
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape)
                   .InCnt(3)
                   .OutCnt(1)
                   .Build("add");
    add->SetOpEngineName("AIcoreEngine");
    add->SetOpKernelLibName("AIcoreEngine");
    auto netoutput = OP_CFG("NetOutput").TensorDesc(FORMAT_NCHW, DT_FLOAT16, shape).InCnt(1).OutCnt(1).Build("netoutput");
    netoutput->SetSrcName({"add"});
    netoutput->SetSrcIndex({0});

    DEF_GRAPH(g1) {
      CHAIN(NODE(data_a)->NODE(known_subgraph)->NODE(add)->NODE(netoutput));
      CHAIN(NODE(known_subgraph)->EDGE(1, 1)->NODE(add));
      CHAIN(NODE(known_subgraph)->EDGE(2, 2)->NODE(add));
    };

    auto graph = ToGeGraph(g1);
    auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
    compute_graph->SetGraphUnknownFlag(true);
    auto parent_node = compute_graph->FindNode(ge::PARTITIONEDCALL);
    auto sub_graph = ToGeGraph(g2);
    auto sub_compute_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);
    sub_compute_graph->SetGraphUnknownFlag(false);
    auto conv2d_node = sub_compute_graph->FindNode("conv2d");
    const char kernel_bin[] = "kernel_bin";
    vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
    conv2d_node->GetOpDesc()->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);

    // set sub graph
    SetSubGraph(compute_graph, parent_node, sub_compute_graph);
    AddCompileResult(parent_node, false);
    auto add_node = compute_graph->FindNode("add");
    AddCompileResult(add_node, false);
    return compute_graph;
  }
};
namespace {
UINT32 StubTilingFuncSuccEmpty(TilingContext *context) {
  EXPECT_EQ(context->GetInputDesc(0)->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(context->GetInputTensor(0)->GetDataType(), ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeFindTilingFunc(KernelContext *context) {
  auto tiling_fun_ptr = context->GetOutputPointer<OpImplKernelRegistry::TilingKernelFunc>(0);
  GE_ASSERT_NOTNULL(tiling_fun_ptr);
  *tiling_fun_ptr = StubTilingFuncSuccEmpty;
  return ge::GRAPH_SUCCESS;
}
}

TEST_F(StaticCompiledGraphTest, KnownSubgraph_ExecuteSuccess) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  GertRuntimeStub stub;
  const auto back_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = back_options;
  new_options["ge.exec.frozenInputIndexes"] = "0;1";
  ge::GetThreadLocalContext().SetGlobalOption(new_options);  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EKnownSubgraphGraph");

  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  // pre alloc 1G memory
  auto mem_block = allocator->Malloc(2 * 2048);
  memset_s(const_cast<void *>(mem_block->GetAddr()), mem_block->GetSize(), 0, 2048 * 2);

  auto outputs = FakeTensors({2, 2}, 3, mem_block->GetAddr());
  auto inputs = FakeTensors({2, 2}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  stub.GetSlogStub().SetLevel(DLOG_INFO);
  stub.GetSlogStub().Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  stub.GetSlogStub().FindLogEndswith(1, "CopyOutputData:base");
  ASSERT_TRUE(stub.GetSlogStub().FindInfoLogRegex("inputs address and size") != -1);
  ASSERT_TRUE(stub.GetSlogStub().FindInfoLogRegex("outputs address and size") != -1);

  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "DavinciModelExecute"), 1);
  ASSERT_EQ(model_executor->ExecuteSync(inputs.GetTensorList(), inputs.size(),
                                        reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  rtStreamDestroy(stream);
  mem_block->Free();
  ge::GetThreadLocalContext().SetGlobalOption(back_options);
}

TEST_F(StaticCompiledGraphTest, KnownSubgraph_WithVariable_And_SubMemInfo_ExecuteSuccess) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();

  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);
  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  ASSERT_NE(sub_graph, nullptr);
  // const --> Variable
  auto const1 = sub_graph->FindNode("const1");
  auto op_desc = const1->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, "Variable");
  ge::TensorUtils::SetSize(*op_desc->MutableOutputDesc(0), 64);
  op_desc->SetOutputOffset({137438959572U});

  UtestRt2VarManager manager;
  Rt2VarManagerFaker rt2_var_manager_faker(manager);
  rt2_var_manager_faker.AddVar("const1", {2, 2});

  GertRuntimeStub fakeRuntime;
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).EnableSubMemInfo().Build();
  const uint64_t session_id = 1U;

  ge::VarManager::Instance(session_id)->Init(0, session_id, 0, 0);
  ge::VarManager::Instance(session_id)->SetVarMemLogicBase(137438959572U);
  VarManager::Instance(session_id)->SetAllocatedGraphId("const1", sub_graph->GetGraphID());

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EKnownSubgraphGraph");

  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto rt_session = gert::RtSession(session_id);
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model, &rt_session);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);

  gert::OuterWeightMem weight = {nullptr, 0U};
  rt_session.SetVarManager(static_cast<RtVarManager *>(&manager));
  ModelLoadArg load_arg(&rt_session, weight);
  EXPECT_EQ(model_executor->Load(nullptr, load_arg), ge::GRAPH_SUCCESS);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048 * 2);
  memset_s(const_cast<void *>(mem_block->GetAddr()), mem_block->GetSize(), 0, 2048 * 2);

  auto outputs = FakeTensors({2, 2}, 3, mem_block->GetAddr());
  auto inputs = FakeTensors({2, 2}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "DavinciModelExecute"), 1);
  ASSERT_EQ(model_executor->ExecuteSync(inputs.GetTensorList(), inputs.size(),
                                        reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  rtStreamDestroy(stream);
  mem_block->Free();
  VarManager::Instance(session_id)->FreeVarMemory();
}

/**
 * 用例描述: partitioned_call子图中的嵌套子图，内部子图父节点直连NetOutput
 *
 * 预置条件：
 * 1. 构造包含partitioned_call子图嵌套静态子图的计算图
 *
 * 测试步骤：
 * 1. 构造包含静态子图嵌套静态子图的计算图
 * 2. 转换为执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 根据日志校验静态子图的输出内存复用情况
 */
TEST_F(StaticCompiledGraphTest, KnownSubgraph_ExecuteSuccess_WithNestingSubGraph) {
  auto graph = ShareGraph::BuildWithNestingKnownSubgraph();
  GertRuntimeStub stub;
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();

  stub.GetSlogStub().Clear();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  stub.GetSlogStub().FindErrorLogEndsWith("Reuse info of static graph g2: outputs num: 3, no reuse outputs: [0, ], reuse outputs: [(1->0), (2->0), ], reuse inputs: [], reference outputs: []");
}

TEST_F(StaticCompiledGraphTest, MultiKnownSubgraph_ReuseStream_LoadSuccess) {
  g_runtime_stub_mock = "";
  auto graph = ShareGraph::BuildWithMulitKnownSubgraphs();
  GertRuntimeStub stub;
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  (void)StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  g_runtime_stub_mock = "";
}

TEST_F(StaticCompiledGraphTest, KnownSubgraph_WithGlobalWorkspace_ExecuteSuccess) {
  ge::GetThreadLocalContext().SetGraphOption({{"ge.exec.static_model_addr_fixed", "1"}});
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  GertRuntimeStub stub;
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();

  ModelDescHolder model_desc_holder;
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  global_data.SetStaicModelWsSize(0U);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "DavinciModelCreateV2"), 1);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  // pre alloc 1G memory
  auto mem_block = allocator->Malloc(2048 * 2);
  memset_s(const_cast<void *>(mem_block->GetAddr()), mem_block->GetSize(), 0, 2048 * 2);

  auto outputs = FakeTensors({2, 2}, 3, mem_block->GetAddr());
  auto inputs = FakeTensors({2, 2}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  stub.GetSlogStub().SetLevel(DLOG_INFO);
  stub.GetSlogStub().Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  stub.GetSlogStub().FindLogEndswith(1, "CopyOutputData:base");
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "DavinciModelExecute"), 1);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "DavinciModelUpdateWorkspaces"), 0);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  rtStreamDestroy(stream);
  mem_block->Free();
  ge::GetThreadLocalContext().SetGraphOption({{}});
}

TEST_F(StaticCompiledGraphTest, KnownSubgraphWithAddNode_FakeTiling_ExecuteSuccess) {
  auto graph = BuildWithKnownSubgraphWithAddNode();
  GertRuntimeStub stub;
  const auto back_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = back_options;
  new_options["ge.exec.frozenInputIndexes"] = "0;1";
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  graph->TopologicalSorting();
  auto ge_model_builder = GeModelBuilder(graph);
  ge_model_builder.AddTbeKernelStore();
  auto root_model = ge_model_builder.BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).FakeWithHandleAiCore("Add", false, false).Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EKnownSubgraphGraph");

  GertRuntimeStub runtime_stub;
  // runtime_stub.GetSlogStub().SetLevelInfo();
  runtime_stub.GetKernelStub().SetUp("FindTilingFunc", FakeFindTilingFunc);

  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  // pre alloc 1G memory
  auto mem_block = allocator->Malloc(2 * 2048);
  memset_s(const_cast<void *>(mem_block->GetAddr()), mem_block->GetSize(), 0, 2048 * 2);

  auto outputs = FakeTensors({2, 2}, 1, mem_block->GetAddr());
  auto inputs = FakeTensors({2, 2}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  stub.GetSlogStub().SetLevel(DLOG_INFO);
  stub.GetSlogStub().Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  rtStreamDestroy(stream);
  mem_block->Free();
  ge::GetThreadLocalContext().SetGlobalOption(back_options);
}
}  // namespace gert
