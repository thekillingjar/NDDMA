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
#include "engine/aicore/fe_rt2_common.h"
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
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "common/topo_checker.h"

using namespace ge;
namespace gert {
namespace {
constexpr size_t kShapeBufferNum = 0U;
constexpr size_t kReservedWorkspaceNum = 16U;
constexpr size_t kTilingAddrOffsetNum = 1U;
constexpr size_t kOverflowAddrNum = 1U;
constexpr size_t kWorkspaceNum = 16;
constexpr size_t kTilingDataLen = 8U;
constexpr size_t kTilingDfxAppend = 8U;
struct FakeArgsInfo {
  ::domi::ArgsInfo::ArgsType arg_type;
  ::domi::ArgsInfo::ArgsFormat arg_fmt;
  int32_t statrt_index;
  uint32_t arg_num;
};
LoweringGlobalData GlobalDataWithArgsInfo(const NodePtr &aicore_node, const LoweringGlobalData &origin_global_data,
                                          std::vector<std::vector<FakeArgsInfo>> &args_infos) {
  LoweringGlobalData global_data;
  LoweringGlobalData::NodeCompileResult aicore_node_compile_result;
  auto task_defs = origin_global_data.FindCompiledResult(aicore_node)->GetTaskDefs();
  if (task_defs.size() !=args_infos.size()) {
    GELOGE(ge::PARAM_INVALID,
           "task_defs.size()[%zu] must be equal to args_infos.size()[%zu], return origin global data instead.",
           task_defs.size(), args_infos.size());
    return origin_global_data;
  }
  for (size_t i = 0U; i < task_defs.size(); ++i) {
    auto task_def = task_defs[i];
    auto kernel_def = task_def.mutable_kernel();
    for (size_t idx = 0U; idx < args_infos[i].size(); ++idx) {
      auto args_info_ = kernel_def->add_args_info();
      args_info_->set_arg_type(args_infos[i][idx].arg_type);
      args_info_->set_arg_format(args_infos[i][idx].arg_fmt);
      args_info_->set_start_index(args_infos[i][idx].statrt_index);
      args_info_->set_size(args_infos[i][idx].arg_num);
    }
    aicore_node_compile_result.task_defs.push_back(task_def);
  }
  global_data.AddCompiledResult(aicore_node, aicore_node_compile_result);
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  return global_data;
}
size_t GetTilingDataOffset(size_t l1_addr_num) {
  return (l1_addr_num + kShapeBufferNum + kReservedWorkspaceNum + kTilingAddrOffsetNum + kOverflowAddrNum) *
         sizeof(TensorAddress);
}
size_t GetTilingAddrOffset(size_t l1_addr_num) {
  return (l1_addr_num + kShapeBufferNum + kWorkspaceNum) * sizeof(TensorAddress);
}
size_t GetArgsSize(size_t args_info_num, size_t merged_copy_addr_num, size_t host_input_data_num) {
  size_t host_input_data_offset = GetTilingDataOffset(args_info_num) + kTilingDataLen + kTilingDfxAppend;
  return host_input_data_offset + merged_copy_addr_num * sizeof(TensorAddress) + host_input_data_num * 4U;
}
size_t GetArgsSizeNew(const NodePtr node, size_t l1_addr_num, size_t host_input_data_num) {
  auto algin_host_input_len= 0U;
  if (ge::AddOverflow(ge::RoundUp(host_input_data_num * 4U, kHostInputDataAlginSize), kHostInputDataAlginSize,
                  algin_host_input_len)) {
    GELOGE(ge::PARAM_INVALID, "algin host input size failed, host input size = %zu", host_input_data_num * 4U);
    return 0U;
  }
  size_t host_input_data_offset = GetTilingDataOffset(l1_addr_num) + kTilingDataLen + kTilingDfxAppend + algin_host_input_len;

  std::vector<std::vector<int64_t>> dyn_in_vv;
  (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv;
  (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  size_t merged_copy_size = 0;
  for (const auto &dyn_v : dyn_in_vv) {
    size_t dyn_num = dyn_v.size();
    size_t addr_offset = 1;
    for (size_t i = 0; i < dyn_num; ++i) {
      int64_t index = dyn_v[i];
      auto tensor_desc = node->GetOpDesc()->MutableInputDesc(index);
      auto dim_num = tensor_desc->GetShape().IsScalar() ? 1 : tensor_desc->GetShape().GetDimNum();
      if (dim_num > MAX_DIM_NUM || dim_num == 0U) {
        dim_num = MAX_DIM_NUM;
      }
      addr_offset++;           // dim && cnt
      addr_offset += dim_num;  // dim value
    }
    addr_offset = addr_offset * sizeof(TensorAddress);
    size_t group_size = addr_offset + dyn_num * sizeof(TensorAddress);  // ptrs
    merged_copy_size += group_size;
  }
  for (const auto &dyn_v : dyn_out_vv) {
    size_t dyn_num = dyn_v.size();
    size_t addr_offset = 1;
    for (size_t i = 0; i < dyn_num; ++i) {
      int64_t index = dyn_v[i];
      auto tensor_desc = node->GetOpDesc()->MutableOutputDesc(index);
      auto dim_num = tensor_desc->GetShape().IsScalar() ? 1 : tensor_desc->GetShape().GetDimNum();
      if (dim_num > MAX_DIM_NUM || dim_num == 0U) {
        dim_num = MAX_DIM_NUM;
      }
      addr_offset++;           // dim && cnt
      addr_offset += dim_num;  // dim value
    }
    addr_offset = addr_offset * sizeof(TensorAddress);
    size_t group_size = addr_offset + dyn_num * sizeof(TensorAddress);  // ptrs
    merged_copy_size += group_size;
  }
  return host_input_data_offset + merged_copy_size;
}
}
class GraphExecutorWithCopyFlowLaunchKernelUnitTest : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
};
const std::string DynamicAtomicStubName = "DynamicAtomicBin";

TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_CopyFlowLaunch_WithOneFlowLaunch) {
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

  auto kernel_launch_node =
    ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "LaunchKernelWithFlag");
  GELOGD("kernel_launch_node:%s", kernel_launch_node[0]->GetName().c_str());
  auto main_graph = kernel_launch_node[0]->GetExtendInfo()->GetOwnerGraphBarePtr();
  ge::DumpGraph(main_graph, "main_graph");
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);

  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);

  EXPECT_EQ(add_launch_args->GetStream(), stream);
  /*
   * before align:
   * +-----------------------------------------------------------------------+
   * |  args:   |--tilingDataOffset--|--tiling data--|--copy flow data--|    |
   * |  offset: |--                 104  (len:8)    112   (len:4)      116   |
   * +-----------------------------------------------------------------------+
   *
   * after align copy flow size:
   * +------------------------------------------------------------------------------------------------+
   * |  args:   |--tilingDataOffset--|--tiling data--|--copy flow data  (align size)   (add 32) |     |
   * |  offset: |--                 104  (len:8)    112    (len:4      +     28       +    32  176    |
   * +------------------------------------------------------------------------------------------------+
   * */

  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, 248);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, 152);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, 168);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 1);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 8);
  auto args_host_buffer = reinterpret_cast<TensorAddress *>(add_launch_args->GetArgsEx()->args);
  ASSERT_NE(args_host_buffer, nullptr);
  EXPECT_NE(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U], nullptr);
  EXPECT_EQ(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U],
            args_host_buffer[add_launch_args->GetArgsEx()->tilingDataOffset / 8 - 1U]);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_CopyFlowLaunch_WithTwoFlowLaunch) {
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

  auto outputs = FakeTensors({2048}, 1);

  auto input_tensor1 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);

  EXPECT_EQ(add_launch_args->GetStream(), stream);
  /*
   * before align:
   * +-----------------------------------------------------------------------+
   * |  args:   |--tilingDataOffset--|--tiling data--|--copy flow data--|    |
   * |  offset: |--                 104  (len:8)    112   (len:8)      120   |
   * +-----------------------------------------------------------------------+
   *
   * after align copy flow size:
   * +------------------------------------------------------------------------------------------------+
   * |  args:   |--tilingDataOffset--|--tiling data--|--copy flow data  (align size)   (add 32) |     |
   * |  offset: |--                 104  (len:8)    112    (len:8      +     24       +    32  176    |
   * +------------------------------------------------------------------------------------------------+
   * */
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, 248);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, 152);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, 168);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 2);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].addrOffset, 8);
  auto args_host_buffer = reinterpret_cast<TensorAddress *>(add_launch_args->GetArgsEx()->args);
  ASSERT_NE(args_host_buffer, nullptr);
  EXPECT_NE(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U], nullptr);
  EXPECT_EQ(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U],
            args_host_buffer[add_launch_args->GetArgsEx()->tilingDataOffset / 8 - 1U]);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_CopyFlowLaunch_NoFlowLaunch) {
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

  auto outputs = FakeTensors({2048}, 1);
  auto input_holder1 = TensorFaker().Placement(kOnHost).DataType(ge::DT_FLOAT).Shape({2048}).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Shape({2048}).Build();
  std::vector<Tensor *> inputs{input_holder1.GetTensor(), input_tensor2.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, 184);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, 152);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, 168);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 0);
  auto args_host_buffer = reinterpret_cast<TensorAddress *>(add_launch_args->GetArgsEx()->args);
  ASSERT_NE(args_host_buffer, nullptr);
  EXPECT_NE(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U], nullptr);
  EXPECT_EQ(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U],
            args_host_buffer[add_launch_args->GetArgsEx()->tilingDataOffset / 8 - 1U]);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_BinaryArgs_AllRequired) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto origin_global_data = GlobalDataFaker(root_model).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).Build();

  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSize(args_info.size(), 0, 0));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(args_info.size()));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(args_info.size()));
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_BinaryArgs_OneEmptyOptionalInput) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto origin_global_data = GlobalDataFaker(root_model).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).Build();

  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 0},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSize(args_info.size(), 0, 0));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(args_info.size()));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(args_info.size()));
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_BinaryArgs_DynamicInput) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  std::vector<std::vector<int64_t>> dyn_in_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto origin_global_data = GlobalDataFaker(root_model).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).Build();

  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 1);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSizeNew(add_node, 5, 0));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(args_info.size()));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(args_info.size()));
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 8);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].dataOffset, 200);
  rtStreamDestroy(stream);
}
TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_BinaryArgs_EmptyDynamicInput) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto origin_global_data = GlobalDataFaker(root_model).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).Build();

  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 0},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSize(args_info.size(), 0, 0));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(5));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(5));
  rtStreamDestroy(stream);
}
TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_BinaryArgs_CopyFlowLaunch_MixScene1) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  std::vector<std::vector<int64_t>> dyn_in_vv = {{0, 1, 2}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto origin_global_data = GlobalDataFaker(root_model).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).Build();

  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 4);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSizeNew(add_node, 3, 3));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(3));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(3));
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].dataOffset, 184);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].addrOffset, 296);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].dataOffset, 312);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].addrOffset, 304);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].dataOffset, 316);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].addrOffset, 8);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].dataOffset, 320);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
/* *
 * 用例描述：测试ArgsInfo中，对于第1个和第2个输入arg都是来自同一个输出的场景，随路拷贝能够只拷贝1次tensor，且记录2次HostInputInfo，其中dataOffset相同
 * 预置条件：4输入1输出的Aicore算子计算图、ArgsInfo(3输入、1输出；第1个和第2个输入arg都表示图中的同一个输入)、输入tensor的device：1D3H
 * 测试步骤：
 * 1.构造第1个和第2个输入arg都表示图中的同一个输入的ArgsInfo，传给aicore算子；
 * 2.先后执行随路拷贝和launchKernel；
 * 预期结果：随路拷贝能够只拷贝1次tensor，且记录2次HostInputInfo，其中dataOffset相同
 */
TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, CopyFlowLaunch_SingleNodeAiCoreWithDuplicatedBinaryArgs) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto origin_global_data = GlobalDataFaker(root_model).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).Build();
  std::vector<std::vector<int64_t>> dyn_in_vv = {{0, 1, 2}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 5);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSizeNew(add_node, 4, 3));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].dataOffset, 192);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].addrOffset, 304);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].dataOffset, 320);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].addrOffset, 312);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].dataOffset, 324);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].addrOffset, 8);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].dataOffset, 328);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].addrOffset,
            add_launch_args->GetArgsEx()->hostInputInfoPtr[3].addrOffset + 8);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].dataOffset,
            add_launch_args->GetArgsEx()->hostInputInfoPtr[3].dataOffset);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_BinaryArgs_CopyFlowLaunch_MixScene2) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto origin_global_data = GlobalDataFaker(root_model).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).Build();
  std::vector<std::vector<int64_t>> dyn_in_vv = {{2}, {3}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 0},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 5);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSizeNew(add_node, 5, 4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(5));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(5));
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 16);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].dataOffset, 200);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].addrOffset, 24);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].dataOffset, 248);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].addrOffset, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].dataOffset, 296);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].addrOffset, 240);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].dataOffset, 304);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].addrOffset, 288);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].dataOffset, 308);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_BinaryArgs_CopyFlowLaunch_MixScene3) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto origin_global_data = GlobalDataFaker(root_model).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).Build();
  std::vector<std::vector<int64_t>> dyn_in_vv = {{0,1}, {3}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 0},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 5);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSizeNew(add_node, 4, 4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].dataOffset, 192);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].addrOffset, 16);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].dataOffset, 280);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].addrOffset, 264);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].dataOffset, 328);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].addrOffset, 272);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].dataOffset, 332);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].addrOffset, 320);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].dataOffset, 340);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_OutOfOrderArgsInfo_MultiExcuteSUCCESS) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto origin_global_data = GlobalDataFaker(root_model).AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin")).Build();
  std::vector<std::vector<int64_t>> dyn_in_vv = {{0,1}, {3}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv = {{0}};
  (void)ge::AttrUtils::SetListListInt(add_node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  std::vector<FakeArgsInfo> args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, -1, 0},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<std::vector<FakeArgsInfo>> args_infos{args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 5);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSizeNew(add_node, 4, 3));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].dataOffset, 192);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].addrOffset, 16);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].dataOffset, 280);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].addrOffset, 24);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].dataOffset, 328);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].addrOffset, 272);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].dataOffset, 376);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].addrOffset, 320);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].dataOffset, 384);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 5);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, GetArgsSizeNew(add_node, 4, 3));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, GetTilingAddrOffset(4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, GetTilingDataOffset(4));
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].dataOffset, 192);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].addrOffset, 16);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].dataOffset, 280);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].addrOffset, 24);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[2].dataOffset, 328);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].addrOffset, 272);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[3].dataOffset, 376);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].addrOffset, 320);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[4].dataOffset, 384);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleNodeAiCore_BinaryArgs_AtomicClean) {
  auto graph = ShareGraph::AddWith4InputsAicoreGraph();
  auto add_node = graph->FindNode("add1");
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();

  auto origin_global_data =
      GlobalDataFaker(root_model)
          .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").AtomicStubNum(DynamicAtomicStubName))
          .Build();

  std::vector<FakeArgsInfo> add_node_args_info{
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 1, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 2, 1},
      {::domi::ArgsInfo_ArgsType_INPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 3, 1},
      {::domi::ArgsInfo_ArgsType_OUTPUT, ::domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0, 1}};
  std::vector<FakeArgsInfo> atomic_clean_node_args_info{};
  std::vector<std::vector<FakeArgsInfo>> args_infos{add_node_args_info, atomic_clean_node_args_info};
  auto global_data = GlobalDataWithArgsInfo(add_node, origin_global_data, args_infos);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 1);
  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor3 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  auto input_tensor4 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor(), input_tensor3.GetTensor(),
                               input_tensor4.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);
  EXPECT_EQ(add_launch_args->GetStream(), stream);
  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, 200);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, 168);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, 184);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 0);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
  runtime_stub.Clear();
}

TEST_F(GraphExecutorWithCopyFlowLaunchKernelUnitTest, SingleHostCopyToTwoAiCoreNode_CopyFlowLaunch) {
  auto graph = ShareGraph::ShapeToMultiAiCoreGraph();
  auto add_node = graph->FindNode("add1");
  AttrUtils::SetBool(add_node->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(add_node->GetOpDesc(), "globalworkspace_size", 32U);

  auto relu_node = graph->FindNode("relu");
  AttrUtils::SetBool(relu_node->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(relu_node->GetOpDesc(), "globalworkspace_size", 32U);
  graph->TopologicalSorting();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").AtomicStubNum(DynamicAtomicStubName))
      .AddTaskDef("Relu", AiCoreTaskDefFaker("ReluStubBin"))
      .Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape"), nullptr);
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EH2DCopyFlowGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  auto kernel_launch_node =
    ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "LaunchKernelWithFlag");
  GELOGD("kernel_launch_node:%s", kernel_launch_node[0]->GetName().c_str());
  auto main_graph = kernel_launch_node[0]->GetExtendInfo()->GetOwnerGraphBarePtr();
  ge::DumpGraph(main_graph, "main_graph");
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 2);

  auto input_tensor1 = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_INT32).Shape({1,2,3,4}).StorageShape({1,2,3,4}).Build();
  auto input_tensor2 = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Build();
  std::vector<Tensor *> inputs{input_tensor1.GetTensor(), input_tensor2.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  dlog_setlevel(GE_MODULE_NAME, 0, 0);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_add_12345");
  ASSERT_NE(add_launch_args, nullptr);

  EXPECT_EQ(add_launch_args->GetStream(), stream);
  /*
   * before align:
   * +------------------------------------------------------------------------+
   * |  args:   |--tilingDataOffset--|--tiling data--|--copy flow data--|     |
   * |  offset: |--                 104  (len:0)    104   (len:20)      116   |
   * +-----------------------------------------------------------------------+
   *
   * after align copy flow size:
   * +------------------------------------------------------------------------------------------------+
   * |  args:   |--tilingDataOffset--|--tiling data--|--copy flow data  (align size)   (add 32) |     |
   * |  offset: |--                 104  (len:0)    104    (len:4      +     20       +    40  168    |
   * +------------------------------------------------------------------------------------------------+
   * */

  EXPECT_EQ(add_launch_args->GetArgsEx()->argsSize, 104);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingAddrOffset, 24);
  EXPECT_EQ(add_launch_args->GetArgsEx()->tilingDataOffset, 40);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 2);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 0);
  EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[1].addrOffset, 8);
  auto args_host_buffer = reinterpret_cast<TensorAddress *>(add_launch_args->GetArgsEx()->args);
  ASSERT_NE(args_host_buffer, nullptr);
  /*EXPECT_NE(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U], nullptr);
  EXPECT_EQ(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U],
            args_host_buffer[add_launch_args->GetArgsEx()->tilingDataOffset / 8 - 1U]);
  ASSERT_EQ(args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 2U], nullptr);*/


  auto relu_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName("te_relu_12345");
  ASSERT_NE(relu_launch_args, nullptr);
  /*
   * before align:
   * +------------------------------------------------------------------------+
   * |  args:   |--tilingDataOffset--|--tiling data--|--copy flow data--|     |
   * |  offset: |--                 96  (len:0)    96   (len:16)      116   |
   * +-----------------------------------------------------------------------+
   *
   * after align copy flow size:
   * +------------------------------------------------------------------------------------------------+
   * |  args:   |--tilingDataOffset--|--tiling data--|--copy flow data  (align size)   (add 32) |    |
   * |  offset: |--                 96  (len:0)    96    (len:4      +     16       +    44  160    |
   * +------------------------------------------------------------------------------------------------+
   * */

  EXPECT_EQ(relu_launch_args->GetArgsEx()->argsSize, 96);
  EXPECT_EQ(relu_launch_args->GetArgsEx()->tilingAddrOffset, 16);
  EXPECT_EQ(relu_launch_args->GetArgsEx()->tilingDataOffset, 32);
  EXPECT_EQ(relu_launch_args->GetArgsEx()->hostInputInfoNum, 1);
  EXPECT_EQ(relu_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 0);
  auto relu_args_host_buffer = reinterpret_cast<TensorAddress *>(add_launch_args->GetArgsEx()->args);
  ASSERT_NE(relu_args_host_buffer, nullptr);
  /*EXPECT_NE(relu_args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U], nullptr);
  EXPECT_EQ(relu_args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 1U],
            relu_args_host_buffer[add_launch_args->GetArgsEx()->tilingDataOffset / 8 - 1U]);
  ASSERT_EQ(relu_args_host_buffer[add_launch_args->GetArgsEx()->tilingAddrOffset / 8 + 2U], nullptr);*/

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  // exe_graph check topo
  ge::FastNode *relu_launch_kernel = nullptr;
  for (const auto launch_node : kernel_launch_node) {
    if (launch_node->GetName().find("_relu_") != std::string::npos) {
      relu_launch_kernel = launch_node;
      std::cout<< launch_node->GetName()<< std::endl;
    }
  }
  EXPECT_NE(relu_launch_kernel, nullptr);
  FastNodeTopoChecker topo_checker1(relu_launch_kernel);
  EXPECT_EQ(topo_checker1.StrictConnectFrom({
                                                {"SplitRtStreams", 0},
                                                {"InnerData", 0},
                                                {"InnerData", 0},
                                                {"AllocBatchHbm", 0},
                                                {"InnerData", 0},
                                                {"InnerData", 0},
                                                {"InnerData", 0},
                                                {"InnerData", 0},
                                                {"InnerData", 0},
                                                {"InnerData", 0},
                                                {"InnerData", 0},
                                                {"CopyFlowLaunch", 0}, // only one copy flow
                                                {"AllocModelOutTensor", 0},
                                                {"InferShape", 0},
                                                {"InnerData", 0},
                                            }),
            "success");
  rtStreamDestroy(stream);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
}
}  // namespace gert
