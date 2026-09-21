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
#include "runtime/model_v2_executor.h"
#include "framework/common/ge_types.h"
#include "exe_graph/runtime/kernel_context.h"
#include "lowering/model_converter.h"
#include "exe_graph/runtime/tiling_context.h"
#include "lowering/graph_converter.h"
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "runtime/dev.h"
// stub and faker
#include "stub/gert_runtime_stub.h"
#include "check/executor_statistician.h"
#include "faker/global_data_faker.h"
#include "faker/ge_model_builder.h"
#include "faker/fake_value.h"
#include "faker/dvpp_taskdef_faker.h"

using namespace ge;
namespace gert {
class DvppGraphExecutorWithKernelUnitTest : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    rtSetDevice(0);
  }
};

TEST_F(DvppGraphExecutorWithKernelUnitTest, Dvpp_ExecuteSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ASSERT_NE(graph, nullptr);
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameDvpp.c_str());
  graph->TopologicalSorting();

  DvppTaskDefFaker dvpp_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).AddTaskDef("Add",
      dvpp_task_def_faker).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "DvppLoweringSTGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubGenerateSqeAndLaunchTask();
  runtime_stub.GetKernelStub().StubCalcDvppWorkSpaceSize();
  runtime_stub.GetKernelStub().StubAllocDvppWorkSpaceMem();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto inputs  = FakeTensors({2048}, 2);
  auto outputs = FakeTensors({2048}, 1);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, 
      static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({stream_value.value},
      inputs.GetTensorList(), inputs.size(),
      outputs.GetTensorList(), outputs.size()),
      ge::GRAPH_SUCCESS);

  Shape expect_out_shape{2048};
  EXPECT_EQ(outputs.GetTensorList()[0]->GetShape().GetStorageShape(),
            expect_out_shape);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
}  // namespace gert