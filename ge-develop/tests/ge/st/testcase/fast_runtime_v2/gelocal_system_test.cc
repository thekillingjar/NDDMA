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
#include "faker/model_data_faker.h"
#include "faker/global_data_faker.h"
#include "faker/rt2_var_manager_faker.h"
#include "runtime/model_v2_executor.h"
#include "common/bg_test.h"
#include "runtime/dev.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "faker/aicpu_taskdef_faker.h"
#include "op_impl/less_important_op_impl.h"
#include "op_impl/dynamic_rnn_impl.h"
#include "securec.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "check/executor_statistician.h"
#include "graph_builder/bg_infer_shape_range.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "lowering/placement/placed_lowering_result.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "graph_metadef/common/ge_common/util.h"

using namespace ge;
namespace gert {
namespace {
ge::graphStatus CreateFakeOMAndFileConstantWeightFile(const std::string &om_dir, const std::string &weight_file_name,
                                                      const size_t weight_data_length) {
  // fileconstant weight file
  std::string weight_dir = om_dir + "/weight/";
  GE_ASSERT_TRUE(CreateDirectory(weight_dir) == 0);

  std::string weight_file_path = weight_dir.append(weight_file_name);
  vector<int32_t> data(weight_data_length);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = i;
  }
  std::ofstream out1(weight_file_path, std::ios::binary);
  GE_ASSERT_TRUE(out1.is_open());
  out1.write(reinterpret_cast<char *>(data.data()), sizeof(int32_t) * data.size());
  out1.close();

  // test_resnet50.om
  std::string fake_om_path = om_dir + "/test_resnet50.om";
  std::ofstream out2(fake_om_path, std::ios::binary);
  GE_ASSERT_TRUE(out2.is_open());
  out2.write(reinterpret_cast<char *>(data.data()), sizeof(int32_t) * data.size());
  out2.close();
  return ge::GRAPH_SUCCESS;
}

void TestFileConstantLoadStatus(const std::string &om_dir, const std::string &om_path,
                                const ge::graphStatus load_status, bool is_weight_dir_concat_expected,
                                bool is_from_location_attr_expected,
                                bool use_user_device_mem = false,
                                bool size_too_small = false) {
  gert::CreateVersionInfo();
  const std::string weight_file_name = "test_weight_xxxxxx.bin";
  const size_t weight_data_length = 25U;
  ASSERT_EQ(CreateFakeOMAndFileConstantWeightFile(om_dir, weight_file_name, weight_data_length), ge::GRAPH_SUCCESS);

  auto graph = ShareGraph::BuildFileConstantGraph();
  auto FileConstant = graph->FindNode("FileConstant");
  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "location", weight_file_name);
  ge::AttrUtils::SetBool(FileConstant->GetOpDesc(), "OwnerGraphIsUnknown", true);

  LoadExecutorArgs args{nullptr, {}};
  int32_t user_mem[weight_data_length];
  if (use_user_device_mem) {
    for (int32_t i = 0; static_cast<size_t>(i) < weight_data_length; i++) {
      user_mem[i] = i;
    }
    ge::FileConstantMem file_constant_mem{weight_file_name, &user_mem, weight_data_length * sizeof(int32_t)};
    if (size_too_small) {
      file_constant_mem.mem_size = weight_data_length;
    }
    args.file_constant_mems = {file_constant_mem};
  }
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  auto model_data = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  model_data.Get().om_path = om_path;
  ge::graphStatus error_code;
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevel(DLOG_INFO);
  auto model_executor = gert::LoadExecutorFromModelData(model_data.Get(), args, error_code);
  gert::DestroyVersionInfo();
  if (use_user_device_mem && size_too_small) {
    std::string error_log = std::string("[Check][Param] The device memory size set by the user via "
        "aclmdlSetExternalWeightAddress for the external weight file is insufficient. ");
    auto log_inx2 = runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, error_log.c_str());
    EXPECT_NE(log_inx2, -1);
    return;
  }
  ASSERT_NE(model_executor, nullptr);
  if (use_user_device_mem) {
    std::string user_mem_log = std::string("found user device memory ");
    auto log_inx2 = runtime_stub.GetSlogStub().FindLog(DLOG_INFO, user_mem_log.c_str());
    EXPECT_NE(log_inx2, -1);
  }
  runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  error_code = model_executor->Load({stream});
  EXPECT_EQ(error_code, load_status);
  ASSERT_TRUE(mmRmdir(om_dir.c_str()) == 0);
  runtime_stub.GetSlogStub().SetLevel(DLOG_ERROR);

  std::string expect_empty_dir_log0 = std::string("Get file constant weight dir[].");
  std::string expect_log1 = "Get file constant info from private attr";
  auto log_idx0 = runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, expect_empty_dir_log0.c_str());
  auto log_idx1 = runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, expect_log1.c_str());
  bool is_weight_dir_concat = log_idx0 == -1;
  bool is_from_location_attr = log_idx1 != -1;
  EXPECT_EQ(is_weight_dir_concat, is_weight_dir_concat_expected);
  EXPECT_EQ(is_from_location_attr, is_from_location_attr_expected);

  if (error_code != ge::GRAPH_SUCCESS) {
    rtStreamDestroy(stream);
    return;
  }

  auto inputs = FakeTensors({}, 0);
  auto output_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({5, 5}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto tensor_data = output_holder.GetTensor()->GetData<int32_t>();
  for (size_t idx = 0U; idx < weight_data_length; ++idx) {
    EXPECT_EQ(tensor_data[idx], idx);
  }

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
}
class GeLocalTest : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
  void TearDown() override {
    Test::TearDown();
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    }
  }
};

TEST_F(GeLocalTest, MultiBatchShapesOK) {
  auto compute_graph = ShareGraph::BuildMultiBatchShapesGraph();
  ASSERT_NE(compute_graph, nullptr);
  GertRuntimeStub rtstub;
  rtstub.StubByNodeTypes({"Shape"});
  compute_graph->TopologicalSorting();
  GeModelBuilder builder(compute_graph);
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("MapIndex", aicpu_task_def_faker).BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);

  ASSERT_NE(exe_graph, nullptr);
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  auto ess = StartExecutorStatistician(model_executor);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  float data[2400] = {0};
  auto inputs = FakeTensors({8, 3, 1, 100}, 4, (void *)data);
  auto output_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Shape({4}).Build();
  auto output_holder2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Shape({4}).Build();
  auto output_holder3 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Shape({4}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor(), output_holder2.GetTensor(), output_holder3.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  ess->Clear();
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GeLocalTest, GatherShapesOK) {
  auto graph = ShareGraph::BuildGatherShapesGraph();
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EGatherShapesGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto inputs = FakeTensors({1, 2, 3, 4}, 1);
  auto output_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT64).Shape({2}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("GatherShapes", "GatherShapesKernel"), 1);
  ess->Clear();
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GeLocalTest, KnownSubgraphWithAllConstNoNeedToUseDavinciModel_ExecuteSuccess) {
  auto graph = ShareGraph::BuildWithAllConstKnownSubgraph();
  GertRuntimeStub fakeRuntime;
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.Build();

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
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "DavinciModelExecute"), 0);
  ASSERT_EQ(model_executor->ExecuteSync(inputs.GetTensorList(), inputs.size(),
                                        reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  rtStreamDestroy(stream);
  mem_block->Free();
}

/**
 * 用例描述: partitioned_call子图中的inner data节点单独设置lowering func
 *
 * 预置条件：
 * 1. 构造包含partitioned_call子图的计算图
 *
 * 测试步骤：
 * 1. 构造包含静态子图的计算图
 * 2. 转换为执行图，构造执行器
 * 3. 构造输入Tensor执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 校验inner data的lower result和parent node的对应输入节点的lower result指针相同
 */
TEST_F(GeLocalTest, PartitionedCallLowerInnerData_ExecuteSuccess) {
  auto graph = ShareGraph::BuildWithInnerDataSubgraph();
  GertRuntimeStub fakeRuntime;
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  auto data_a = graph->FindNode("data_a");
  const auto *const_data_a_result = data_a->GetOpDesc()->GetExtAttr<gert::PlacedLoweringResult>(kLoweringResult);
  auto *data_a_result = const_cast<PlacedLoweringResult *>(const_data_a_result);
  auto data_a_out_result = data_a_result->GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);

  auto parent_node = graph->FindNode("PartitionedCall");
  ASSERT_NE(parent_node, nullptr);
  auto root_compute_graph = ge::GraphUtils::FindRootGraph(parent_node->GetOwnerComputeGraph());
  ASSERT_NE(root_compute_graph, nullptr);
  auto sub_graph = root_compute_graph->GetSubgraph(parent_node->GetOpDesc()->GetSubgraphInstanceName(0U));

  auto data_1 = sub_graph->FindNode("data_1");
  ASSERT_NE(data_1, nullptr);
  const auto *const_data_1_result = data_1->GetOpDesc()->GetExtAttr<gert::PlacedLoweringResult>(kLoweringResult);
  auto *data_1_result = const_cast<PlacedLoweringResult *>(const_data_1_result);
  ASSERT_EQ(data_1_result->GetResult()->out_addrs.size(), 1);
  EXPECT_EQ(data_a_out_result->address, data_1_result->GetResult()->out_addrs[0]);

  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

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
    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "DavinciModelExecute"), 0);
  ASSERT_EQ(model_executor->ExecuteSync(inputs.GetTensorList(), inputs.size(),
    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  rtStreamDestroy(stream);
  mem_block->Free();
}

TEST_F(GeLocalTest, SizeOK) {
  auto graph = ShareGraph::BuildSizeGraph();
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EGatherShapesGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto inputs = FakeTensors({1, 2, 3, 4}, 1);
  auto output_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT64).Shape({}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Size", "GetShapeSizeKernel"), 1);
  ess->Clear();
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GeLocalTest, RankOK) {
  auto graph = ShareGraph::BuildRankGraph();
  GertRuntimeStub runtime_stub;
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2ERankGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto inputs = FakeTensors({1, 2, 3, 4}, 1);
  auto output_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Shape({}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Rank", "RankKernel"), 1);
  ess->Clear();

  auto tensor_data = output_holder.GetTensor()->GetData<int32_t>();
  ASSERT_EQ(*tensor_data, 4);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GeLocalTest, FileConstantOK) {
  std::string file_name = "test_weight.bin";
  int32_t data[25];
  for (int32_t i = 0; i < 25; i++) {
    data[i] = i;
  }
  std::ofstream out1(file_name, std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write(reinterpret_cast<char*>(data), sizeof(data));
  out1.close();

  auto graph = ShareGraph::BuildFileConstantGraph();
  auto FileConstant = graph->FindNode("FileConstant");
  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "file_path", file_name);
  std::vector<int32_t> shape = {5, 5};
  ge::AttrUtils::SetListInt(FileConstant->GetOpDesc(), "shape", shape);

  graph->TopologicalSorting();

  GertRuntimeStub runtime_stub;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  // lowering
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  runtime_stub.Clear();
  auto ess = StartExecutorStatistician(model_executor);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  EXPECT_EQ(model_executor->Load({stream}), ge::GRAPH_SUCCESS);
  auto inputs = FakeTensors({}, 0);
  auto output_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({5, 5}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};

  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto tensor_data = output_holder.GetTensor()->GetData<int32_t>();
  ASSERT_EQ(tensor_data[5], 5);
  ess->PrintExecutionSummary();
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：离线场景下，传入OM路径，fileconstant算子文件路径拼接成功，权重加载成功
 * 预置条件：
 * 1. GertRuntimeStub接管slog打印
 * 2. fileconstant算子的REG_OP和IMPL_OP原型注册，并设置其3个私有属性
 * 3. 一张带有fileconstant算子的计算图, fileconstant节点直连netoutput
 * 4. OM的相对路径路径为"./test_om_dir/test_resnet50.om"
 * 5. 一份预置的fileconstant权重文件./test_om_dir/weight/test_weight_xxxxxx.bin
 * 测试步骤：
 * 1. 将fileconstant算子的权重文件名test_weight.bin设置为location属性，并将权重文件放置在OM目录下
 * 2. 通过计算图生成ModelData，并将OM相对路径设置给ModelData
 * 3. 通过ModelData加载执行器
 * 4. 执行器Load和执行
 * 预期结果：
 * 1. 执行器Load成功
 * 2. 日志打印离线场景以及权重路径由location属性获取
 * 3. 执行器执行输出正确，即tensordata输出为权重文件内容
 */
TEST_F(GeLocalTest, FileConstant_LoadWeightOkOffline_SetOMPath2ModelData) {
  const std::string om_dir = "./xj_test_om_dir";
  const std::string om_path = om_dir + "/test_resnet50.om";
  const ge::graphStatus load_status = ge::GRAPH_SUCCESS;
  bool is_weight_dir_concat_expected = true;
  bool is_from_location_attr_expected = true;
  TestFileConstantLoadStatus(om_dir, om_path, load_status, is_weight_dir_concat_expected,
                             is_from_location_attr_expected);
}

/**
 * 用例描述：用户为FileConstant设置了Device地址，使用用户设置的Device地址
 * 预置条件：
 * 1. GertRuntimeStub接管slog打印
 * 2. fileconstant算子的REG_OP和IMPL_OP原型注册，并设置其3个私有属性
 * 3. 一张带有fileconstant算子的计算图, fileconstant节点直连netoutput
 * 4. OM的相对路径路径为"./test_om_dir/test_resnet50.om"
 * 5. 一份预置的fileconstant权重文件./test_om_dir/weight/test_weight_xxxxxx.bin
 * 测试步骤：
 * 1. fileconstant算子的权重文件名test_weight.bin
 * 2. 通过计算图生成ModelData
 * 3. 通过ModelData加载执行器
 * 4. 执行器Load和执行
 * 预期结果：
 * 1. 执行器Load成功
 * 2. 日志打印使用了用户设置的地址
 * 3. 执行器执行输出正确，即tensordata输出为权重文件内容
 */
TEST_F(GeLocalTest, FileConstant_LoadWeightOkOffline_UserDeviceMem) {
  const std::string om_dir = "./xj_test_om_dir";
  const std::string om_path = om_dir + "/test_resnet50.om";
  const ge::graphStatus load_status = ge::GRAPH_SUCCESS;
  bool is_weight_dir_concat_expected = true;
  bool is_from_location_attr_expected = false;
  TestFileConstantLoadStatus(om_dir, om_path, load_status, is_weight_dir_concat_expected,
                             is_from_location_attr_expected, true, false);
}

/**
 * 用例描述：用户为FileConstant设置了Device地址，大小不匹配，校验报错
 * 预置条件：
 * 1. GertRuntimeStub接管slog打印
 * 2. fileconstant算子的REG_OP和IMPL_OP原型注册，并设置其3个私有属性
 * 3. 一张带有fileconstant算子的计算图, fileconstant节点直连netoutput
 * 4. OM的相对路径路径为"./test_om_dir/test_resnet50.om"
 * 5. 一份预置的fileconstant权重文件./test_om_dir/weight/test_weight_xxxxxx.bin
 * 测试步骤：
 * 1. fileconstant算子的权重文件名test_weight.bin
 * 2. 通过计算图生成ModelData
 * 3. 通过ModelData加载执行器
 * 4. 执行器Load和执行
 * 预期结果：
 * 1. 执行器Load失败
 */
TEST_F(GeLocalTest, FileConstant_LoadWeightOkOffline_UserDeviceMem_SizeInvalid) {
  const std::string om_dir = "./xj_test_om_dir";
  const std::string om_path = om_dir + "/test_resnet50.om";
  const ge::graphStatus load_status = ge::GRAPH_FAILED;
  bool is_weight_dir_concat_expected = true;
  bool is_from_location_attr_expected = false;
  TestFileConstantLoadStatus(om_dir, om_path, load_status, is_weight_dir_concat_expected,
                             is_from_location_attr_expected, true, true);
}

/**
 * 用例描述：离线场景下，不传入OM路径，fileconstant算子文件路径拼接失败，权重加载失败
 * 预置条件：
 * 1. GertRuntimeStub接管slog打印
 * 2. fileconstant算子的REG_OP和IMPL_OP原型注册，并设置其3个私有属性
 * 3. 一张带有fileconstant算子的计算图, fileconstant节点直连netoutput
 * 4. OM的相对路径路径为"./test_om_dir/resnet50.om"
 * 5. 一份预置的fileconstant权重文件./test_om_dir/weight/test_weight_xxxxxx.bin
 * 测试步骤：
 * 1. 将fileconstant算子的权重文件名test_weight.bin设置为location属性，并将权重文件放置在OM目录下
 * 2. 通过计算图生成ModelData，不设置OM相对路径给ModelData
 * 3. 通过ModelData加载执行器
 * 4. 执行器Load和执行
 * 预期结果：
 * 1. 执行器Load失败，权重文件路径拼接失败，无法找到权重文件
 * 2. 日志不打印离线场景，但是权重路径仍由location属性获取
 * 3. 执行器执行失败
 */
TEST_F(GeLocalTest, FileConstant_LoadWeightFailedOffline_NotSetOMPath2ModelData) {
  const std::string om_dir = "./xj_test_om_dir";
  const std::string om_path;
  const ge::graphStatus load_status = ge::PARAM_INVALID;
  bool is_weight_dir_concat_expected = false;
  bool is_from_location_attr_expected = true;
  TestFileConstantLoadStatus(om_dir, om_path, load_status, is_weight_dir_concat_expected,
                             is_from_location_attr_expected);
}

/**
 * 用例描述：离线场景下，传入OM路径有误，fileconstant算子文件路径拼接失败，权重加载失败
 * 预置条件：
 * 1. GertRuntimeStub接管slog打印
 * 2. fileconstant算子的REG_OP和IMPL_OP原型注册，并设置其3个私有属性
 * 3. 一张带有fileconstant算子的计算图, fileconstant节点直连netoutput
 * 4. OM的相对路径路径为"./test_om_dir/resnet50.om"
 * 5. 一份预置的fileconstant权重文件./test_om_dir/weight/test_weight_xxxxxx.bin
 * 测试步骤：
 * 1. 将fileconstant算子的权重文件名test_weight.bin设置为location属性，并将权重文件放置在OM目录下
 * 2. 通过计算图生成ModelData，设置错误的OM相对路径给ModelData
 * 3. 通过ModelData加载执行器
 * 4. 执行器Load和执行
 * 预期结果：
 * 1. 执行器Load失败，权重文件路径拼接失败，无法找到权重文件
 * 2. 日志打印离线场景，但是om路径有误，但是权重路径仍由location属性获取
 * 3. 执行器执行失败
 */
TEST_F(GeLocalTest, FileConstant_LoadWeightFailedOffline_SetErrorOMPath2ModelData) {
  const std::string om_dir = "./xj_test_om_dir";
  const std::string om_path = om_dir + "/weight/resnet50.om";
  const ge::graphStatus load_status = ge::PARAM_INVALID;
  bool is_weight_dir_concat_expected = true;  // 当前的RealPath打桩好像有问题,等打桩问题解决后，修改为false
  bool is_from_location_attr_expected = true;
  TestFileConstantLoadStatus(om_dir, om_path, load_status, is_weight_dir_concat_expected,
                             is_from_location_attr_expected);
}

TEST_F(GeLocalTest, FileConstant_ModelV2ExecutorLoadFailed_StreamIsNullptr) {
  std::string file_name = "test_weight.bin";
  int32_t data[25];
  for (int32_t i = 0; i < 25; i++) {
    data[i] = i;
  }
  std::ofstream out1(file_name, std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write(reinterpret_cast<char*>(data), sizeof(data));
  out1.close();
  auto graph = ShareGraph::BuildFileConstantGraph();
  auto FileConstant = graph->FindNode("FileConstant");
  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "location", "test_weight.bin");
  std::vector<int32_t> shape = {5, 5};
  ge::AttrUtils::SetListInt(FileConstant->GetOpDesc(), "shape", shape);
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EFileConstantGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_NE(model_executor->Load(), ge::GRAPH_SUCCESS);
}

// 用例描述: 在线场景，FileConstant走变量管理器，执行时直接从变量管理器中获取输出地址(且权重数据也ready)。
TEST_F(GeLocalTest, FileConstantLoadSuccess_VarMgrIsExist) {
  // fake var mgr
  UtestRt2VarManager var_mgr;
  string var_id("FileConstant");
  StorageShape ss{{5,5}, {5,5}};
  std::unique_ptr<int32_t[]> magic_addr(new int32_t[25U]);
  for (size_t i = 0U; i < 25U; ++i) {
    magic_addr[i] = i;
  }
  gert::TensorData gert_data(magic_addr.get());
  gert_data.SetPlacement(TensorPlacement::kOnDeviceHbm);
  gert_data.SetSize(25U * sizeof(int32_t));
  var_mgr.SetVarShapeAndMemory(var_id, ss, gert_data);
  RtSession rt_session;
  rt_session.SetVarManager(&var_mgr);

  auto graph = ShareGraph::BuildFileConstantGraph();
  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EFileConstantGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  ASSERT_EQ(model_executor->Load({stream}, {&rt_session}), ge::GRAPH_SUCCESS);

  auto inputs = FakeTensors({}, 0);
  auto output_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({5, 5}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto tensor_data = output_holder.GetTensor()->GetData<int32_t>();
  for (size_t idx = 0U; idx < 5 * 5; ++idx) {
    EXPECT_EQ(tensor_data[idx], idx);
  }

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

// 用例描述：离线场景+静态子图+子图内FileConstant直连Netoutput场景，davinci model kernel执行成功，权重加载成功。
TEST_F(GeLocalTest, StaticSubgraphWithFileConstantLinkNetoutput_FileConstantLoadSuccess) {
  const std::string om_dir = "./xj_test_om_dir";
  const std::string om_path = om_dir + "/test_resnet50.om";
  const std::string weight_file_name = "test_weight_xxxxxx.bin";
  const size_t weight_data_length = 2 * 2;
  ASSERT_EQ(CreateFakeOMAndFileConstantWeightFile(om_dir, weight_file_name, weight_data_length), ge::GRAPH_SUCCESS);

  auto graph = ShareGraph::BuildWithKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);
  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  ASSERT_NE(sub_graph, nullptr);
  // const --> fileconstant
  auto const1 = sub_graph->FindNode("const1");
  auto const_op_desc = const1->GetOpDesc();
  ge::OpDescUtilsEx::SetType(const_op_desc, FILECONSTANT);
  ge::AttrUtils::SetInt(const_op_desc, "offset", 0);
  ge::AttrUtils::SetInt(const_op_desc, "length", 0);
  ge::AttrUtils::SetStr(const_op_desc, "location", weight_file_name);
  ge::AttrUtils::SetDataType(const_op_desc, "dtype", DT_INT32);

  GertRuntimeStub fakeRuntime;
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
  const uint64_t session_id = 1U;

  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetFileConstantWeightDir(om_dir + "/weight/");
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EKnownSubgraphGraph");

  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);

  auto rt_session = gert::RtSession(session_id);
  ModelLoadArg load_arg(&rt_session);
  EXPECT_EQ(model_executor->Load(nullptr, load_arg), ge::GRAPH_SUCCESS);

  auto output_holder_0 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({2, 2}).Build();
  auto output_holder_1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({2, 2}).Build();
  auto output_holder_2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({2, 2}).Build();
  std::vector<Tensor *> outputs{output_holder_0.GetTensor(), output_holder_1.GetTensor(), output_holder_2.GetTensor()};
  auto inputs = FakeTensors({2, 2}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.data()), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "DavinciModelExecute"), 1);
  ASSERT_EQ(model_executor->ExecuteSync(inputs.GetTensorList(), inputs.size(),
                                        reinterpret_cast<Tensor **>(outputs.data()), outputs.size()),
            ge::GRAPH_SUCCESS);


  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  ASSERT_TRUE(mmRmdir(om_dir.c_str()) == 0);
}

TEST_F(GeLocalTest, FileConstantLocationAttrOK) {
  std::string file_name = "test_weight.bin";
  int32_t data[25];
  for (int32_t i = 0; i < 25; i++) {
    data[i] = i;
  }
  std::ofstream out1(file_name, std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write(reinterpret_cast<char*>(data), sizeof(data));
  out1.close();
  auto graph = ShareGraph::BuildFileConstantGraph();
  auto FileConstant = graph->FindNode("FileConstant");
  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "location", "test_weight.bin");
  std::vector<int32_t> shape = {5, 5};
  ge::AttrUtils::SetListInt(FileConstant->GetOpDesc(), "shape", shape);
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EFileConstantGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  EXPECT_EQ(model_executor->Load({stream}), ge::GRAPH_SUCCESS);
  auto inputs = FakeTensors({}, 0);
  auto output_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({5, 5}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};

  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto tensor_data = output_holder.GetTensor()->GetData<int32_t>();
  ASSERT_EQ(tensor_data[5], 5);
  ess->PrintExecutionSummary();
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GeLocalTest, FileConstantFileIdAttrOK) {
  // user set option
  std::map<std::string, std::string> options;
  options["ge.exec.value_bins"] =
      "{\"value_bins\":[{\"value_bin_id\":\"vector_search_buchet_value_bin\", \"value_bin_file\":\"test_file_id.bin\"}]}";
  // get file_path_and_file_id
  ge::GetThreadLocalContext().SetGraphOption(options);
  std::string file_name = "test_file_id.bin";
  int32_t data[25];
  for (int32_t i = 0; i < 25; i++) {
    data[i] = i;
  }
  std::ofstream out1(file_name, std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write(reinterpret_cast<char*>(data), sizeof(data));
  out1.close();
  auto graph = ShareGraph::BuildFileConstantGraph();
  auto FileConstant = graph->FindNode("FileConstant");
  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "file_id", "vector_search_buchet_value_bin");
  std::vector<int32_t> shape = {5, 5};
  ge::AttrUtils::SetListInt(FileConstant->GetOpDesc(), "shape", shape);
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  // lowering
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "E2EFileConstantGraph");
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  EXPECT_EQ(model_executor->Load({stream}), ge::GRAPH_SUCCESS);
  auto inputs = FakeTensors({}, 0);
  auto output_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({5, 5}).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};

  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto tensor_data = output_holder.GetTensor()->GetData<int32_t>();
  ASSERT_EQ(tensor_data[5], 5);
  ess->PrintExecutionSummary();
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GeLocalTest, FileConstantWithCtrlEdgeConvertOK) {
  std::string file_name_0 = "test_weight_0.bin";
  std::string file_name_1 = "test_weight_1.bin";
  int32_t data[25];
  for (int32_t i = 0; i < 25; i++) {
    data[i] = i;
  }
  std::ofstream out0(file_name_0, std::ios::binary);
  if (!out0.is_open()) {
    return;
  }
  out0.write(reinterpret_cast<char*>(data), sizeof(data));
  out0.close();

  std::ofstream out1(file_name_1, std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write(reinterpret_cast<char*>(data), sizeof(data));
  out1.close();

  auto graph = ShareGraph::Build2FileConstantWithCtrlEdgeGraph();

  auto FileConstant0 = graph->FindNode("FileConstant0");
  ge::AttrUtils::SetStr(FileConstant0->GetOpDesc(), "location", file_name_0);
  auto FileConstant1 = graph->FindNode("FileConstant1");
  ge::AttrUtils::SetStr(FileConstant1->GetOpDesc(), "location", file_name_1);
  std::vector<int32_t> shape = {5, 5};
  ge::AttrUtils::SetListInt(FileConstant0->GetOpDesc(), "shape", shape);
  ge::AttrUtils::SetListInt(FileConstant1->GetOpDesc(), "shape", shape);
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
  auto ess = StartExecutorStatistician(model_executor);
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  EXPECT_EQ(model_executor->Load({stream}), ge::GRAPH_SUCCESS);

  auto inputs = FakeTensors({}, 0);
  auto output_holder0 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({5, 5}).Build();
  auto output_holder1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({5, 5}).Build();
  std::vector<Tensor *> outputs{output_holder0.GetTensor(), output_holder1.GetTensor()};

  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto tensor_data0 = output_holder0.GetTensor()->GetData<int32_t>();
  ASSERT_EQ(tensor_data0[5], 5);
  auto tensor_data1 = output_holder1.GetTensor()->GetData<int32_t>();
  ASSERT_EQ(tensor_data1[5], 5);
  ess->PrintExecutionSummary();
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

// FileConstant的执行已挪到加载阶段
TEST_F(GeLocalTest, FileConstantFailed) {
  std::string file_name = "";
  auto graph = ShareGraph::BuildFileConstantGraph();
  auto FileConstant = graph->FindNode("FileConstant");
  ge::AttrUtils::SetStr(FileConstant->GetOpDesc(), "file_path", file_name);
  std::vector<int32_t> shape = {5, 5};
  ge::AttrUtils::SetListInt(FileConstant->GetOpDesc(), "shape", shape);

  graph->TopologicalSorting();

  GertRuntimeStub runtime_stub;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  // lowering
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  EXPECT_EQ(exe_graph, nullptr);
}
}
