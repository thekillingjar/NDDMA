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
#include "op_impl/dynamicatomicaddrclean/dynamic_atomic_addr_clean_impl.h"
#include "op_impl/data_flow_op_impl.h"
#include "faker/ge_model_builder.h"
#include "lowering/model_converter.h"
#include "mmpa/mmpa_api.h"
#include "securec.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"

#include "register/op_tiling/op_tiling_constants.h"
#include "register/op_tiling/op_compile_info_manager.h"
#include "register/op_tiling_registry.h"
#include "graph/utils/op_desc_utils.h"
#include "faker/aicpu_taskdef_faker.h"
#include "framework/common/ge_types.h"
#include "runtime/gert_api.h"
#include "mmpa/mmpa_api.h"
#include "stub/hostcpu_mmpa_stub.h"
#include "check/executor_statistician.h"
#include "graph_builder/bg_infer_shape_range.h"
#include "common/helper/model_parser_base.h"
#include "register/kernel_registry.h"
#include "graph_builder/bg_rt_session.h"
#include "framework/runtime/rt_session.h"

using namespace ge;
namespace gert {
namespace {
LowerResult LoweringFoo(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto rt_session = bg::GetRtSession(*lower_input.global_data);
  auto ret = bg::DevMemValueHolder::CreateSingleDataOutput("GetTestSessionId", {rt_session}, node->GetOpDesc()->GetStreamId());
  LowerResult result;
  result.out_shapes.push_back(lower_input.input_shapes[0]);
  result.out_addrs.push_back(ret);
  return result;
}

ge::graphStatus GetTestSessionId(KernelContext *context) {
  auto rt_session = context->GetInputValue<RtSession *>(0);
  auto session_id_holder = context->GetOutputPointer<GertTensorData>(0);
  auto session_id_value = rt_session->GetSessionId();
  memcpy_s(session_id_holder->GetAddr(), sizeof(uint64_t), &session_id_value, sizeof(uint64_t));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateGetSessionIdTensorDataAtHost(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto session_id_chain = context->GetOutput(0);
  const auto data_type_size = ge::GetSizeByDataType(ge::DT_UINT64);
  auto malloc_buffer_size = data_type_size + sizeof(GertTensorData);
  auto out_data = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[malloc_buffer_size]);
  GE_ASSERT_NOTNULL(out_data);
  new (out_data.get())
      GertTensorData(out_data.get() + sizeof(GertTensorData), malloc_buffer_size - sizeof(GertTensorData), kOnHost, -1);
  session_id_chain->SetWithDefaultDeleter<uint8_t[]>(out_data.release());
  return ge::GRAPH_SUCCESS;
}
} // namespace

class RtSessionSystemTest : public bg::BgTest {
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
const std::string TransDataStubName = "TransDataStubBin";
const std::string MulStubName = "MulStubBin";
/*
* Lowering简单图，输出加载时传入的SessionId
*  计算图：
*       Data->Foo->NetOutput
*  执行图：
*    init图：                  main图:
*        ConstData                InnerData
*           |                        |
*        GetSessionId           EnsureTensorAtOutMemory
*           |                         |
*        InnerNetoutput           NetOutput
*/
TEST_F(RtSessionSystemTest,  OneConstData_Load_Execute_Success) {
  // 1.为Foo算子的converter和相关kernel打桩
  // 注册Foo算子的node converter
  // 注册GetSessionId kernel的实现
  GertRuntimeStub stub;
  stub.GetConverterStub().Register("Foo", LoweringFoo);
  KernelRegistry::KernelFuncs funcs = {};
  funcs.run_func = GetTestSessionId;
  funcs.outputs_creator = CreateGetSessionIdTensorDataAtHost;
  stub.GetKernelStub().SetUp("GetTestSessionId", funcs);

  // 2.构造Foo graph
  auto compute_graph = ShareGraph::SimpleFooGraph();
  // 设置输出数据类型为UINT64_T,与session id数据类型一致
  auto foo_node = compute_graph->FindFirstNodeMatchType("Foo");
  foo_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(ge::DT_UINT64);
  auto netoutput_node = compute_graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_UINT64);
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  // 3.将计算图lowering成执行图
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_FooGetSessionExeGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  // 4.构造rt session输入
  uint64_t session_id = 100U;
  auto session = gert::RtSession(session_id);
  gert::OuterWeightMem weight = {nullptr, 0U};
  ModelExecuteArg arg = {(void *)2};
  ModelLoadArg load_arg(&session, weight);
  // 5. 执行加载流程
  ASSERT_EQ(model_executor->Load(&arg, load_arg), ge::GRAPH_SUCCESS);

  // 6. 执行main图
  auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[8]);
  auto outputs = FakeTensors({2}, 1, mem_block.get(), kOnHost); // fake value内部写死了tensor size为4，不好
  auto input0 =
      FakeValue<Tensor>(Tensor{{{1}, {1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_UINT64, 0});
  ASSERT_EQ(
      model_executor->Execute(&arg, std::vector<Tensor *>({input0.holder.get()}).data(),
                              1, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  // 7. 校验main图输出value为GetSessionId，即100U.
  auto ret_session_id = outputs.at(0).GetData<uint64_t>();
  EXPECT_EQ(*ret_session_id, session_id);

  // 不构造rt session输入, 测试默认session功能
  // 1. 执行加载流程
  ASSERT_EQ(model_executor->Load(&arg), ge::GRAPH_SUCCESS);

  // 2. 执行main图
  ASSERT_EQ(
      model_executor->Execute(&arg, std::vector<Tensor *>({input0.holder.get()}).data(),
                              1, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}
}  // namespace gert
