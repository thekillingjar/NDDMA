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

#include "macro_utils/dt_public_scope.h"
#include "hybrid/executor/hybrid_model_pipeline_executor.h"
#include "graph/ge_context.h"

namespace ge {
using namespace hybrid;
namespace {
std::vector<gert::Tensor> InputData2GertTensors(const InputData &input_data) {
  std::vector<gert::Tensor> input_tensors;
  for (size_t i = 0; i < input_data.blobs.size(); ++i) {
    gert::Tensor tensor;
    tensor.MutableTensorData().SetSize(input_data.blobs[i].length);
    tensor.MutableTensorData().SetAddr(input_data.blobs[i].data, nullptr);
    tensor.MutableStorageShape().SetDimNum(input_data.shapes[i].size());
    for (size_t j = 0; j < input_data.shapes[i].size(); ++j) {
      tensor.MutableStorageShape().SetDim(0, input_data.shapes[i][j]);
    }
    input_tensors.emplace_back(std::move(tensor));
  }
  return input_tensors;
}
}
class UtestStageExecutor : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() { }
};

TEST_F(UtestStageExecutor, run_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());

  PipeExecutionConfig config;
  config.device_id = 1;
  config.num_executors = 2;
  config.num_stages = 1;
  config.iteration_end = 2;
  rtCtxGetCurrent(&config.rt_context);

  HybridModelPipelineExecutor pip_executor(&hybrid_model, 1, nullptr);
  HybridModelExecutor::ExecuteArgs args;
  ASSERT_EQ(pip_executor.config_.iteration_end, 0);
  ASSERT_EQ(pip_executor.Init(), SUCCESS);
  pip_executor.Execute(args);

  StageSubject stage_subject;
  StageExecutor executor(0, &hybrid_model, &config, &stage_subject);
  StageExecutor next_executor(1, &hybrid_model, &config, &stage_subject);
  EXPECT_EQ(stage_subject.Await(1), SUCCESS);
  executor.SetNext(&next_executor);
  EXPECT_EQ(executor.Init(), SUCCESS);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  EXPECT_NE(allocator, nullptr);
  StageExecutor::StageTask task_info_1;
  task_info_1.stage = 0;
  task_info_1.iteration = 0;
  EXPECT_EQ(rtEventCreate(&task_info_1.event), RT_ERROR_NONE);
  EXPECT_EQ(executor.ExecuteAsync(task_info_1), SUCCESS);
  EXPECT_EQ(executor.Start({}, {}, 2), SUCCESS);

  StageExecutor::StageTask task_info_2;
  task_info_2.stage = 0;
  task_info_2.iteration = 1;
  EXPECT_EQ(rtEventCreate(&task_info_2.event), RT_ERROR_NONE);
  EXPECT_EQ(executor.ExecuteAsync(task_info_2), SUCCESS);
  EXPECT_EQ(executor.Start({}, {}, 2), SUCCESS);
  executor.ExecuteEndTaskAndReleae();
  executor.Reset();
}

TEST_F(UtestStageExecutor, execute_online_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());
  PipeExecutionConfig config;
  config.device_id = 1;
  config.num_executors = 2;
  config.num_stages = 1;
  config.iteration_end = 2;
  rtCtxGetCurrent(&config.rt_context);

  HybridModelPipelineExecutor pip_executor(&hybrid_model, 1, nullptr);
  ASSERT_EQ(pip_executor.config_.iteration_end, 0);
  ASSERT_EQ(pip_executor.Init(), SUCCESS);
  InputData input_data;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  input_data.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  input_data.shapes.push_back({1, 16, 16, 3});
  auto gert_inputs = InputData2GertTensors(input_data);
  EXPECT_EQ(pip_executor.ExecuteOnlineModel(gert_inputs, nullptr), SUCCESS);
}
} // namespace ge
