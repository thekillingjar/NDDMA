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
#include "dflow/inc/data_flow/deploy/npu_sched_model.h"
#include "executor/npu_sched_model_loader.h"
#include "depends/runtime/src/runtime_stub.h"

namespace ge {
// Test cases
TEST(NpuSchedModelTest, TestInitializeNpuSched) {
  uint32_t result = InitializeNpuSched(1);
  EXPECT_EQ(result, 0U);
}

TEST(NpuSchedModelTest, TestCreateNpuSchedModelHandler) {
  NpuSchedModelHandler handler = CreateNpuSchedModelHandler();
  EXPECT_NE(handler, nullptr);
  DestroyNpuSchedModelHandler(handler);
}

TEST(NpuSchedModelTest, TestLoadNpuSchedModel) {
  ge::NpuSchedModelLoader handler;
  NpuSchedLoadParam load_param{};
  uint32_t result = LoadNpuSchedModel(&handler, &load_param);
  // without input and output queue
  EXPECT_NE(result, 0U);
  std::vector<uint32_t> input_queues = {1, 2};
  load_param.model_queue_param.input_queues = input_queues;
  for (auto queue_id : input_queues) {
    QueueAttrs queue = {.queue_id= queue_id, .device_type=NPU, .device_id=0} ;
    load_param.model_queue_param.input_queues_attrs.emplace_back(queue);
  }
  std::vector<uint32_t> output_queues = {1, 2, UINT32_MAX};
  load_param.model_queue_param.output_queues = output_queues;
  for (auto queue_id : output_queues) {
    QueueAttrs queue = {.queue_id= queue_id, .device_type=NPU, .device_id=0} ;
    load_param.model_queue_param.output_queues_attrs.emplace_back(queue);
  }
  g_runtime_stub_mock = "rtCpuKernelLaunchWithFlag";
  result = LoadNpuSchedModel(&handler, &load_param);
  EXPECT_NE(result, 0U);
  g_runtime_stub_mock = "";
  result = LoadNpuSchedModel(&handler, &load_param);
  EXPECT_EQ(result, 0U);
  result = UnloadNpuSchedModel(&handler);
  EXPECT_EQ(result, 0U);
}

TEST(NpuSchedModelTest, TestUnloadNpuSchedModel) {
  ge::NpuSchedModelLoader handler;
  uint32_t result = UnloadNpuSchedModel(&handler);
  EXPECT_EQ(result, 0U);
}

TEST(NpuSchedModelTest, TestFinalizeNpuSched) {
  uint32_t result = FinalizeNpuSched(1);
  EXPECT_EQ(result, 0U);
}
}  // namespace ge