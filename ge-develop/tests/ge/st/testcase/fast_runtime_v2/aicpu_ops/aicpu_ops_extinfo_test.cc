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
#include "faker/space_registry_faker.h"
#include "graph/utils/graph_utils.h"
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
#include "graph/utils/graph_dump_utils.h"

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
#include "engine/aicpu/kernel/aicpu_ext_info_handle.h"
#include "aicpu/kernel/aicpu_bin_handler.h"

using namespace ge;
namespace gert {
class AicpuExtInfoSt : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
};

TEST_F(AicpuExtInfoSt, test_get_workspace_info) {
  const std::string node_name = "test";
  uint32_t input_num = 1U;
  uint32_t output_num = 1U;
  ge::UnknowShapeOpType unknown_type = ge::DEPEND_IN_SHAPE;
  AicpuExtInfoHandler ext_handle(node_name, input_num, output_num, unknown_type);
  auto workspace_info = ext_handle.GetWorkSpaceInfo();
  ASSERT_EQ(workspace_info, nullptr);
}

TEST_F(AicpuExtInfoSt, Success) {
  const std::string so_name = "libcust_add.so";
  vector<char> buffer(1, 'a');
  ge::OpKernelBinPtr bin = std::make_shared<ge::OpKernelBin>("op", std::move(buffer));
  rtBinHandle handle = nullptr;
  CustBinHandlerManager::Instance().LoadAndGetBinHandle(so_name, bin, handle);

  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_desc = graph->FindNode("add1")->GetOpDesc();
  add_desc->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  ge::AttrUtils::SetInt(add_desc, "workspaceSize", 1024);
  std::vector<uint32_t> mem_type;
  mem_type.push_back(ge::AicpuWorkSpaceType::CUST_LOG);
  ge::AttrUtils::SetListInt(add_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, mem_type);
  ge::AttrUtils::SetBool(add_desc, "_cust_aicpu_flag", true);
  ge::AttrUtils::SetStr(add_desc, "kernelSo", so_name);
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

  auto i0 = FakeValue<Tensor>(Tensor{
      {{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(Tensor{
      {{2048}, {2048}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, mem_block->GetAddr()});
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

}  // namespace gert
