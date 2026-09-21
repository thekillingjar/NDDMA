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
#include "op_impl/less_important_op_impl.h"
#include "op_impl/transdata/trans_data_positive_source_tc_1010.h"
#include "op_impl/dynamic_rnn_impl.h"
#include "op_impl/dynamicatomicaddrclean/dynamic_atomic_addr_clean_impl.h"
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
#include "graph/utils/graph_dump_utils.h"
#include "faker/aicpu_taskdef_faker.h"
#include "framework/common/ge_types.h"
#include "runtime/gert_api.h"
#include "mmpa/mmpa_api.h"
#include "stub/hostcpu_mmpa_stub.h"
#include "check/executor_statistician.h"

using namespace ge;
namespace gert {
class KnownShapeGraphUnitTest : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
};

TEST_F(KnownShapeGraphUnitTest, ControlFlowNodeWithKnownShapeSubgraph) {
  auto graph = ShareGraph::IfWithKnownShapeSubGraph("main");
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto faker = GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithHandleAiCore("StaticFoo", false).Build();
  graph->TopologicalSorting();
  bg::ValueHolder::PopGraphFrame();
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess({"SwitchNotify"});
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  ge::DumpGraph(exe_graph.get(), "ControlFlowNodeWithKnownShapeSubgraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  auto ess = StartExecutorStatistician(model_executor);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto pred_tensor = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT32).Value<int32_t>({1}).Build();

  auto input_holder = TensorFaker().Placement(kOnDeviceHbm).DataType(ge::DT_FLOAT).Shape({8,3,224,224}).Build();
  std::vector<Tensor *> inputs{pred_tensor.GetTensor(), input_holder.GetTensor()};
  auto output_holder = TensorFaker().Placement(kOnHost).DataType(ge::DT_INT64).Build();
  std::vector<Tensor *> outputs{output_holder.GetTensor()};
  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("main/If", "DavinciModelUpdateWorkspaces"), 1U);
  EXPECT_EQ(ess->GetExecuteCountByNodeNameAndKernelType("main/If", "DavinciModelExecute"), 1U);

  // todo 做一个基于图的校验方式，校验某个Node必然在另一个Node前面执行

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
}  // namespace gert
