/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/stream_executor.h"
#include "runtime/gert_api.h"
#include <gtest/gtest.h>
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/model_data_faker.h"
#include "faker/model_desc_holder_faker.h"

namespace gert {
class StreamExecutorUT : public testing::Test {
  void SetUp() override {
    std::string opp_version = "./version.info";
    system(("touch " + opp_version).c_str());
    system(("echo 'Version=6.4.T5.0.B121' > " + opp_version).c_str());
    setenv("ASCEND_OPP_PATH", "./", 1);
  }

  void TearDown() override {
    unsetenv("ASCEND_OPP_PATH");
   }
};

TEST_F(StreamExecutorUT, GetOrCreateLoaded_ReturnSameExecutor_WhenStreamTheSame) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();

  SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  auto e1 = stream_executor->GetOrCreateLoaded((rtStream_t)1, {(rtStream_t)1, nullptr});
  auto e2 = stream_executor->GetOrCreateLoaded((rtStream_t)1, {(rtStream_t)1, nullptr});
  ASSERT_NE(e1, nullptr);
  ASSERT_EQ(e1, e2);
}
TEST_F(StreamExecutorUT, GetOrCreateLoaded_ReturnDiffExecutor_WhenStreamDiff) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();

  SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  auto e1 = stream_executor->GetOrCreateLoaded((rtStream_t)1, {(rtStream_t)1, nullptr});
  auto e2 = stream_executor->GetOrCreateLoaded((rtStream_t)2, {(rtStream_t)2, nullptr});
  ASSERT_NE(e1, nullptr);
  ASSERT_NE(e2, nullptr);
  ASSERT_NE(e1, e2);
}
TEST_F(StreamExecutorUT, Erase_CreateNew_AfterErase) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();

  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  auto e1 = stream_executor->GetOrCreateLoaded((rtStream_t)1, {(rtStream_t)1, nullptr});
  ASSERT_NE(e1, nullptr);
  ASSERT_EQ(stream_executor->Erase((rtStream_t)1), ge::GRAPH_SUCCESS);
  auto e2 = stream_executor->GetOrCreateLoaded((rtStream_t)1, {(rtStream_t)1, nullptr});
  ASSERT_NE(e2, nullptr);
  ASSERT_NE(e1, e2);
}
TEST_F(StreamExecutorUT, Erase_Success_EraseNotExists) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();

  SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto ge_root_model = GeModelBuilder(graph)
                           .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .FakeTbeBin({"Add"})
                           .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);

  auto e1 = stream_executor->GetOrCreateLoaded((rtStream_t)1, {(rtStream_t)1, nullptr});
  ASSERT_NE(e1, nullptr);
  ASSERT_EQ(stream_executor->Erase((rtStream_t)1), ge::GRAPH_SUCCESS);
  ASSERT_EQ(stream_executor->Erase((rtStream_t)1), ge::GRAPH_SUCCESS);
}
/*
 * LoadStreamExecutorFromModelData老接口需要适配UT，后续需要配套代码删除
 * */
TEST_F(StreamExecutorUT, TestLoadStreamExecutorFromModelData) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
  graph->TopologicalSorting();

  SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto ge_root_model = GeModelBuilder(graph)
      .AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
      .FakeTbeBin({"Add"})
      .BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  ge::graphStatus error_code = ge::GRAPH_FAILED;
  gert::LoweringOption option;
  auto stream_executor = LoadStreamExecutorFromModelData(model_data_holder.Get(), option,error_code);
  ASSERT_NE(stream_executor, nullptr);
  ASSERT_EQ(error_code, ge::GRAPH_SUCCESS);
}
}  // namespace gert