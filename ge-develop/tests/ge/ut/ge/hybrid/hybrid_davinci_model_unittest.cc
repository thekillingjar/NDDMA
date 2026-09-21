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
#include <gmock/gmock.h>
#include <vector>
#include <memory>

#include "ge_common/ge_api_types.h"
#include "macro_utils/dt_public_scope.h"
#include "hybrid/hybrid_davinci_model.h"
#include "common/dump/dump_manager.h"
#include "../base/graph/manager/graph_manager_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace hybrid;

class HybridDavinciModelTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {

void ListenerCallback(Status status, std::vector<gert::Tensor> &tensors){
  return;
}

TEST_F(HybridDavinciModelTest, HybridDavinciModel_SetGet) {
  auto mod = HybridDavinciModel::Create(std::make_shared<GeRootModel>());
  mod->SetModelId(100);
  mod->SetDeviceId(1001);
  EXPECT_EQ(mod->GetDeviceId(), 1001);
  mod->SetOmName("om_name");
  mod->SetListener(std::make_shared<GraphModelListener>());
  EXPECT_EQ(mod->GetSessionId(), 0);
  std::vector<std::vector<int64_t>> batch_info;
  int32_t dynamic_type = 0;
  EXPECT_EQ(mod->GetDynamicBatchInfo(batch_info, dynamic_type), SUCCESS);
  std::vector<std::string> user_input_shape_order;
  mod->GetUserDesignateShapeOrder(user_input_shape_order);
  EXPECT_EQ(user_input_shape_order.size(), 0);
  std::vector<std::string> dynamic_output_shape_info;
  mod->GetOutputShapeInfo(dynamic_output_shape_info);
  EXPECT_EQ(dynamic_output_shape_info.size(), 0);
  mod->SetModelDescVersion(true);
  EXPECT_EQ(mod->GetRunningFlag(), false);
  EXPECT_EQ(mod->SetRunAsyncListenerCallback(ListenerCallback), PARAM_INVALID);
  EXPECT_EQ(mod->GetGlobalStepAddr(), 0);
}


TEST_F(HybridDavinciModelTest, HybridDavinciModel_Null) {
  auto mod = HybridDavinciModel::Create(std::make_shared<GeRootModel>());
  auto impl = mod->impl_;
  mod->impl_ = nullptr;
  EXPECT_EQ(mod->GetDataInputerSize(), PARAM_INVALID);
  EXPECT_EQ(mod->ModelRunStart(), PARAM_INVALID);
  EXPECT_EQ(mod->ModelRunStop(), PARAM_INVALID);
  EXPECT_EQ(mod->GetDataInputerSize(), PARAM_INVALID);
  EXPECT_EQ(mod->GetGlobalStepAddr(), 0UL);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(mod->Execute(inputs, outputs), PARAM_INVALID);
  EXPECT_EQ(mod->ExecuteWithStreamAsync(inputs, outputs), PARAM_INVALID);
  std::vector<InputOutputDescInfo> input_desc;
  std::vector<InputOutputDescInfo> output_desc;
  std::vector<uint32_t> input_formats;
  std::vector<uint32_t> output_formats;
  EXPECT_EQ(mod->GetInputOutputDescInfo(input_desc, output_desc, input_formats, output_formats), PARAM_INVALID);
  uint32_t stream_id = 0;
  uint32_t task_id = 0;
  OpDescInfo op_desc_info;
  EXPECT_EQ(mod->GetOpDescInfo(stream_id, task_id, op_desc_info), false);
  mod->impl_ = impl;
}

TEST_F(HybridDavinciModelTest, HybridDavinciModel_Execute_UseDefaultStream) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  auto mod = HybridDavinciModel::Create(ge_root_model);
  setenv("ENABLE_RUNTIME_V2", "0", 1);
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(mod->Init(), SUCCESS);
  std::vector<DataBuffer> inputs;
  std::vector<GeTensorDesc> input_desc;
  std::vector<DataBuffer> outputs;
  std::vector<GeTensorDesc> output_desc;
  EXPECT_EQ(mod->Execute(inputs, input_desc, outputs, output_desc, nullptr), SUCCESS);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(HybridDavinciModelTest, HybridDavinciModel_Execute_UseUserStream) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  auto mod = HybridDavinciModel::Create(ge_root_model);
  setenv("ENABLE_RUNTIME_V2", "0", 1);
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(mod->Init(), SUCCESS);
  std::vector<DataBuffer> inputs;
  std::vector<GeTensorDesc> input_desc;
  std::vector<DataBuffer> outputs;
  std::vector<GeTensorDesc> output_desc;
  EXPECT_EQ(mod->Execute(inputs, input_desc, outputs, output_desc, nullptr), SUCCESS);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(HybridDavinciModelTest, HybridDavinciModel_Init_InvalidOption) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  auto mod = HybridDavinciModel::Create(ge_root_model);
  setenv("ENABLE_RUNTIME_V2", "0", 1);
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(mod->Init(), PARAM_INVALID);
}

TEST_F(HybridDavinciModelTest, HybridDavinciModel_Execute_GertTensor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  auto mod = HybridDavinciModel::Create(ge_root_model);
  setenv("ENABLE_RUNTIME_V2", "0", 1);
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(mod->Init(), SUCCESS);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(mod->Execute(inputs, outputs), SUCCESS);
  unsetenv("ENABLE_RUNTIME_V2");
}

}
