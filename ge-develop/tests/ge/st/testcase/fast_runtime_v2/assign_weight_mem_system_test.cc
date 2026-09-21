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
#include "common/share_graph.h"
#include "runtime/gert_api.h"
#include "runtime/model_v2_executor.h"
#include "faker/ge_model_builder.h"
#include "faker/model_data_faker.h"
#include "faker/fake_value.h"
#include "runtime/rt.h"
#include "exe_graph/lowering/value_holder.h"
#include "register/op_tiling_info.h"
#include "graph/debug/ge_attr_define.h"
#include "faker/aicore_taskdef_faker.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

namespace gert {
class AssignWeightMemST : public testing::Test {
 public:
  void SetUp() {
    std::string opp_path = "./";
    std::string opp_version = "version.info";
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
    system(("touch " + opp_version).c_str());
    system(("echo 'Version=3.20.T100.0.B356' > " + opp_version).c_str());
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
  }

  void TearDown() {
    system("rm -f ./version.info");
    unsetenv("ASCEND_OPP_PATH");
  }
};
TEST_F(AssignWeightMemST, BigMemSinkWeightSUCCESS) {
  auto graph = ShareGraph::BuildDynamicAndStaticGraph();
  graph->TopologicalSorting();
  auto ge_model_builder = GeModelBuilder(graph);
  auto ge_root_model = ge_model_builder.BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();

  // malloc device mem
  size_t weight_size = 10U; //require 8
  void *weight_mem = nullptr;
  auto rt_ret = rtMalloc(&weight_mem, weight_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  ASSERT_EQ(rt_ret, RT_ERROR_NONE);
  ge::ModelData model_data = model_data_holder.Get();
  
  GELOGI("alloc weight size[%zu], addr[%p]", weight_size, weight_mem);

  // execuge
  ge::graphStatus ret = ge::GRAPH_SUCCESS;
  auto executor = LoadExecutorFromModelData(model_data, ret);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  uint64_t session_id = 100U;
  auto session = gert::RtSession(session_id);
  gert::OuterWeightMem weight = {weight_mem, weight_size};
  ModelExecuteArg arg = {(void *)2};
  ModelLoadArg load_arg(&session, weight);
  ret = executor->Load(&arg, load_arg);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2, 2}, 2);
  auto inputs = FakeTensors({2, 2}, 2);

  ret = executor->ExecuteSync(inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size());
  executor->UnLoad();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  rt_ret = rtFree(weight_mem);
  ASSERT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(AssignWeightMemST, BigMemSinkWeightSUCCESS_bin_Reuse) {
  auto graph = ShareGraph::BuildDynamicAndStaticGraph();
  graph->TopologicalSorting();
  for (auto node : graph->GetAllNodes()) {
    if (node->GetName() == "add") {
      std::string tiling_data = "add_td";
      std::shared_ptr<optiling::utils::OpRunInfo> add_run_info = std::make_shared<optiling::utils::OpRunInfo>();
      add_run_info->AddTilingData(tiling_data.data(), tiling_data.size());
      add_run_info->SetWorkspaces({1000});
      node->GetOpDescBarePtr()->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, add_run_info);

      node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameAiCore);
      node->GetOpDesc()->SetOpEngineName(ge::kEngineNameAiCore);
    }
  }
  auto ge_model_builder = GeModelBuilder(graph);
  auto ge_root_model = ge_model_builder.BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();

  // malloc device mem
  size_t weight_size = 10U;  // require 8
  void *weight_mem = nullptr;
  auto rt_ret = rtMalloc(&weight_mem, weight_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  ASSERT_EQ(rt_ret, RT_ERROR_NONE);
  ge::ModelData model_data = model_data_holder.Get();

  GELOGI("alloc weight size[%zu], addr[%p]", weight_size, weight_mem);

  // execuge
  ge::graphStatus ret = ge::GRAPH_SUCCESS;
  auto executor = LoadExecutorFromModelData(model_data, ret);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  uint64_t session_id = 100U;
  auto session = gert::RtSession(session_id);
  gert::OuterWeightMem weight = {weight_mem, weight_size};
  ModelExecuteArg arg = {(void *)2};
  ModelLoadArg load_arg(&session, weight);
  ret = executor->Load(&arg, load_arg);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2, 2}, 2);
  auto inputs = FakeTensors({2, 2}, 2);

  ret = executor->ExecuteSync(inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(), outputs.size());
  executor->UnLoad();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  rt_ret = rtFree(weight_mem);
  ASSERT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(AssignWeightMemST, SmallMemSinkWeight_FreeBeforeRun_StillRunOK) {
  auto graph = ShareGraph::BuildDynamicAndStaticGraph();
  graph->TopologicalSorting();
  auto ge_model_builder = GeModelBuilder(graph);
  auto ge_root_model = ge_model_builder.BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  // malloc device mem
  size_t weight_size = 4U; //require 8
  void *weight_mem = nullptr;
  auto rt_ret = rtMalloc(&weight_mem, weight_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  ASSERT_EQ(rt_ret, RT_ERROR_NONE);
  ge::ModelData model_data = model_data_holder.Get();

  // execuge
  ge::graphStatus ret = ge::GRAPH_SUCCESS;
  auto executor = LoadExecutorFromModelData(model_data, ret);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  // weight_mem is small than requred(8) by model, will alloc in framework, so free meme will still run ok
  rt_ret = rtFree(weight_mem);
  ASSERT_EQ(rt_ret, RT_ERROR_NONE);

  uint64_t session_id = 100U;
  auto session = gert::RtSession(session_id);
  gert::OuterWeightMem weight = {weight_mem, weight_size};
  ModelExecuteArg arg = {(void *)2};
  ModelLoadArg load_arg(&session, weight);
  ret = executor->Load(&arg, load_arg);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2, 2}, 2);
  auto inputs = FakeTensors({2, 2}, 2);

  ret = executor->ExecuteSync(inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size());
  executor->UnLoad();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(AssignWeightMemST, MemSinkWeight_UserDefaultWeight) {
  auto graph = ShareGraph::BuildDynamicAndStaticGraph();
  graph->TopologicalSorting();
  auto ge_model_builder = GeModelBuilder(graph);
  auto ge_root_model = ge_model_builder.BuildGeRootModel();
  auto model_data_holder = ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();

  // don't malloc
  ge::ModelData model_data = model_data_holder.Get();

  // execuge
  ge::graphStatus ret = ge::GRAPH_SUCCESS;
  auto executor = LoadExecutorFromModelData(model_data, ret);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  uint64_t session_id = 100U;
  auto session = gert::RtSession(session_id);
  ModelExecuteArg arg = {(void *)2};
  ModelLoadArg load_arg(&session);
  ret = executor->Load(&arg, load_arg);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2, 2}, 2);
  auto inputs = FakeTensors({2, 2}, 2);

  ret = executor->ExecuteSync(inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size());
  executor->UnLoad();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}
} // namespace gert
