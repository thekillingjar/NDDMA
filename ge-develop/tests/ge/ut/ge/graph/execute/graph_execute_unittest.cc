/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/memory/tensor_trans_utils.h"

#include <gtest/gtest.h>
#include <memory>

#include "macro_utils/dt_public_scope.h"
#include "common/profiling/profiling_manager.h"
#include "graph/execute/graph_executor.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"

using namespace std;
using namespace testing;
using namespace domi;

namespace ge {

class UtestGraphExecuteTest : public testing::Test {
 protected:
  void SetUp() {
    const auto davinci_model = MakeShared<DavinciModel>(2001U, MakeShared<RunAsyncListener>());
    ModelManager::GetInstance().InsertModel(2001U, davinci_model);
  }

  void TearDown() {
    EXPECT_EQ(ModelManager::GetInstance().DeleteModel(2001U), SUCCESS);
    EXPECT_TRUE(ModelManager::GetInstance().model_map_.empty());
    EXPECT_TRUE(ModelManager::GetInstance().hybrid_model_map_.empty());
  }
};

TEST_F(UtestGraphExecuteTest, get_execute_model_id_invalid) {
  GraphExecutor executor;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  auto model_id = executor.GetExecuteModelId(ge_root_model);
  EXPECT_EQ(model_id, kInvalidModelId);
}

TEST_F(UtestGraphExecuteTest, get_execute_model_id_1) {
  GraphExecutor executor;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, nullptr);
  davinci_model1->SetId(1);
  ModelManager::GetInstance().InsertModel(1, davinci_model1);
  ge_root_model->SetModelId(1);
  auto model_id = executor.GetExecuteModelId(ge_root_model);
  EXPECT_EQ(model_id, 1);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(1U), SUCCESS);
}

TEST_F(UtestGraphExecuteTest, get_execute_model_id_2) {
  GraphExecutor executor;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  // model1 with 2 load
  shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, nullptr);
  davinci_model1->SetId(1);

  auto data = MakeShared<RunArgs>();
  davinci_model1->Push(data);
  davinci_model1->Push(data);
  ModelManager::GetInstance().InsertModel(1, davinci_model1);
  // model 2 with 3 load
  shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(1, nullptr);
  davinci_model2->SetId(2);
  davinci_model2->Push(data);
  davinci_model2->Push(data);
  davinci_model2->Push(data);
  ModelManager::GetInstance().InsertModel(2, davinci_model2);
  // model 3 witH 1 load
  shared_ptr<DavinciModel> davinci_model3 = MakeShared<DavinciModel>(1, nullptr);
  davinci_model3->SetId(3);
  davinci_model3->Push(data);
  ModelManager::GetInstance().InsertModel(3, davinci_model3);

  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  ge_root_model->SetModelId(3);

  auto model_id = executor.GetExecuteModelId(ge_root_model);
  // model 3 is picked for having least loads
  EXPECT_EQ(model_id, 3);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(1U), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(2U), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(3U), SUCCESS);
}

TEST_F(UtestGraphExecuteTest, test_without_subscribe) {
  auto ret = ModelManager::GetInstance().ModelSubscribe(1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, test_with_subscribe_failed1) {
  uint32_t graph_id = 1;
  ProfilingProperties::Instance().SetSubscribeInfo(0, 1, true);
  auto ret = ModelManager::GetInstance().ModelSubscribe(graph_id);
  ProfilingProperties::Instance().CleanSubscribeInfo();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, test_with_subscribe_failed2) {
  uint32_t graph_id = 1;
  uint32_t model_id = 1;
  ProfilingProperties::Instance().SetSubscribeInfo(0, 1, true);
  ProfilingManager::Instance().SetGraphIdToModelMap(2, model_id);
  auto ret = ModelManager::GetInstance().ModelSubscribe(graph_id);
  ProfilingProperties::Instance().CleanSubscribeInfo();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, test_with_subscribe_success) {
  uint32_t graph_id = 1;
  uint32_t model_id = 2001U;
  GraphNodePtr graph_node = std::make_shared<GraphNode>(graph_id);
  DavinciModel model(model_id, nullptr);
  ProfilingProperties::Instance().SetSubscribeInfo(0, 1, true);
  ProfilingManager::Instance().SetGraphIdToModelMap(graph_id, model_id);
  auto ret = ModelManager::GetInstance().ModelSubscribe(graph_id);
  ProfilingProperties::Instance().CleanSubscribeInfo();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, SetDynamicSize) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  std::vector<uint64_t> batch_num_empty;
  std::vector<uint64_t> batch_num{1};
  auto ret = executor.SetDynamicSize(model_id, batch_num_empty, static_cast<int32_t>(DynamicInputType::DYNAMIC_BATCH));
  ret = executor.SetDynamicSize(model_id, batch_num, static_cast<int32_t>(DynamicInputType::DYNAMIC_BATCH));
  EXPECT_EQ(ret, SUCCESS);
  ret = executor.SetDynamicSize(UINT32_MAX, batch_num, static_cast<int32_t>(DynamicInputType::DYNAMIC_BATCH));
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphExecuteTest, GetInputOutputDescInfo) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  std::vector<InputOutputDescInfo> input_desc;
  std::vector<InputOutputDescInfo> output_desc;
  std::vector<uint32_t> input_formats;
  std::vector<uint32_t> out_formats;
  auto ret = executor.GetInputOutputDescInfo(model_id, input_desc, output_desc, input_formats, out_formats, true);
  ret = executor.GetInputOutputDescInfo(model_id, input_desc, output_desc, input_formats, out_formats, false);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraphExecuteTest, GetDynamicBatchInfo) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  int32_t dynamic_type = static_cast<int32_t>(DynamicInputType::DYNAMIC_BATCH);
  std::vector<std::vector<int64_t>> batch_info;
  auto ret = executor.GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, GetCombinedDynamicDims) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  std::vector<std::vector<int64_t>> batch_info;
  auto ret = executor.GetCombinedDynamicDims(model_id, batch_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, GetUserDesignateShapeOrder) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  std::vector<std::string> user_input_shape_order;
  auto ret = executor.GetUserDesignateShapeOrder(model_id, user_input_shape_order);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, GetCurrentShape) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  std::vector<int64_t> batch_info;
  int32_t dynamic_type = static_cast<int32_t>(DynamicInputType::DYNAMIC_BATCH);
  auto ret = executor.GetCurrentShape(model_id, batch_info, dynamic_type);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, GetOutputShapeInfo) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  std::vector<std::string> dynamic_output_shape_info;
  auto ret = executor.GetOutputShapeInfo(model_id, dynamic_output_shape_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, GetAippInfo) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  uint32_t index = 0;
  AippConfigInfo aipp_info;
  auto ret = executor.GetAippInfo(model_id, index, aipp_info);
  EXPECT_EQ(ret, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(UtestGraphExecuteTest, GetAippType) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  uint32_t index = 0;
  InputAippType type = DATA_WITHOUT_AIPP;
  size_t aipp_index = 0;
  auto ret = executor.GetAippType(model_id, index, type, aipp_index);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphExecuteTest, GetOrigInputInfo) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  uint32_t index = 0;
  OriginInputInfo orig_input_info;
  auto ret = executor.GetOrigInputInfo(model_id, index, orig_input_info);
  EXPECT_EQ(ret, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(UtestGraphExecuteTest, GetAllAippInputOutputDims) {
  GraphExecutor executor;
  uint32_t model_id = 2001U;
  uint32_t index = 0;
  std::vector<InputOutputDims> input_dims;
  std::vector<InputOutputDims> output_dims;
  auto ret = executor.GetAllAippInputOutputDims(model_id, index, input_dims, output_dims);
  EXPECT_EQ(ret, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(UtestGraphExecuteTest, GetOpDescInfo) {
  GraphExecutor executor;
  uint32_t device_id = 0;
  uint32_t stream_id = 0;
  uint32_t task_id = 0;
  OpDescInfo op_desc_info;
  auto ret = executor.GetOpDescInfo(device_id, stream_id, task_id, op_desc_info);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphExecuteTest, SyncExecuteModel_Invalid) {
  GraphExecutor executor;
  std::vector<gert::Tensor> input_tensor(1);
  TensorCheckUtils::ConstructGertTensor(input_tensor[0], {});
  std::vector<gert::Tensor> output_tensor;
  uint32_t model_id = 2001U;
  error_message::ErrorManagerContext error_context;
  auto hybrid_model = hybrid::HybridDavinciModel::Create(std::make_shared<GeRootModel>());
  auto shared_model = std::shared_ptr<hybrid::HybridDavinciModel>(hybrid_model.release());
  shared_model->SetModelId(0U);
  ModelManager::GetInstance().InsertModel(model_id, shared_model);
  auto ret = executor.SyncExecuteModel(model_id, input_tensor, output_tensor, error_context);
  ModelManager::GetInstance().DeleteModel(model_id);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, SyncExecuteModelGertTensor_Invalid) {
  GraphExecutor executor;
  GeTensorDesc td(GeShape(), FORMAT_NCHW, DT_FLOAT);
  GeTensor ge_tensor(td);
  std::vector<GeTensor> input_tensor;
  input_tensor.emplace_back(ge_tensor);
  std::vector<gert::Tensor> output_tensor;
  uint32_t model_id = 2001U;
  error_message::ErrorManagerContext error_context;
  auto hybrid_model = hybrid::HybridDavinciModel::Create(std::make_shared<GeRootModel>());
  auto shared_model = std::shared_ptr<hybrid::HybridDavinciModel>(hybrid_model.release());
  shared_model->SetModelId(0U);
  ModelManager::GetInstance().InsertModel(model_id, shared_model);
  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::GeTensors2GertTensors(input_tensor, gert_inputs);
  auto ret = executor.SyncExecuteModel(model_id, gert_inputs, output_tensor, error_context);
  ModelManager::GetInstance().DeleteModel(model_id);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, ExecuteGraph_Invalid) {
  GraphExecutor executor;
  std::vector<gert::Tensor> input_tensor(1);
  std::vector<gert::Tensor> output_tensor;

  GraphId graph_id = 0;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);

  auto ret = executor.ExecuteGraph(graph_id, ge_root_model, input_tensor, output_tensor);
  EXPECT_EQ(ret, GE_GRAPH_SYNC_MODEL_FAILED);
}

TEST_F(UtestGraphExecuteTest, GetModelByID_Invalid) {
  EXPECT_EQ(ModelManager::GetInstance().GetModel(UINT32_MAX), nullptr);
}

TEST_F(UtestGraphExecuteTest, ExecuteGraphAsync_Invalid3) {
  GraphExecutor executor;
  std::vector<gert::Tensor> input_tensor(1);
  RunAsyncCallback callback;

  GraphId graph_id = 0;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context;
  // create graph
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGraph(graph_ptr);

  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  arg->graph_id = graph_id;
  arg->session_id = session_id;
  arg->error_context = error_context;
  arg->context = context;
  arg->input_tensor = std::move(input_tensor);
  arg->graph_node = graph_node;
  arg->callback = [&](Status status, std::vector<gert::Tensor> &outputs) {};

  {
    auto ret = executor.ExecuteGraphAsync(ge_root_model, arg);
    EXPECT_EQ(ret, GE_GRAPH_SYNC_MODEL_FAILED);
  }

  {
    ge_root_model->SetModelId(666);
    auto ret = executor.ExecuteGraphAsync(ge_root_model, arg);
    EXPECT_EQ(ret, GE_GRAPH_SYNC_MODEL_FAILED);
  }

  {
    compute_graph->SetGraphUnknownFlag(true);
    auto ret = executor.ExecuteGraphAsync(ge_root_model, arg);
    EXPECT_EQ(ret, GE_GRAPH_SYNC_MODEL_FAILED);
  }
}
TEST_F(UtestGraphExecuteTest, ExecuteGraphAsync_GertTensor_Invalid3) {
  GraphExecutor executor;

  GraphId graph_id = 0;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context;
  // create graph
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGraph(graph_ptr);

  std::vector<gert::Tensor> inputs(1);
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  arg->graph_id = graph_id;
  arg->session_id = session_id;
  arg->error_context = error_context;
  arg->context = context;
  arg->input_tensor = std::move(inputs);
  arg->graph_node = graph_node;
  arg->callback = [&](Status status, std::vector<gert::Tensor> &outputs) {};

  {
    auto ret = executor.ExecuteGraphAsync(ge_root_model, arg);
    EXPECT_EQ(ret, GE_GRAPH_SYNC_MODEL_FAILED);
  }

  {
    ge_root_model->SetModelId(666);
    auto ret = executor.ExecuteGraphAsync(ge_root_model, arg);
    EXPECT_EQ(ret, GE_GRAPH_SYNC_MODEL_FAILED);
  }

  {
    compute_graph->SetGraphUnknownFlag(true);
    auto ret = executor.ExecuteGraphAsync(ge_root_model, arg);
    EXPECT_EQ(ret, GE_GRAPH_SYNC_MODEL_FAILED);
  }
}
}  // namespace ge
