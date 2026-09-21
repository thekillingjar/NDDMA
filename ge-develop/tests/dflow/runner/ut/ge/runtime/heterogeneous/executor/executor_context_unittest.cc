/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <gtest/gtest.h>
#include "dflow/inc/data_flow/model/flow_model.h"
#include "executor/executor_context.h"
using namespace std;

namespace ge {
class ExecutorContextTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ExecutorContextTest, AttachQueuesSuccess) {
  deployer::ExecutorRequest_LoadModelRequest model_request;
  auto *queue_def = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  queue_def->set_queue_id(0);
  queue_def->set_device_type(static_cast<int32_t>(NPU));
  queue_def->set_device_id(0);

  queue_def = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
  queue_def->set_queue_id(1);
  queue_def->set_device_type(static_cast<int32_t>(NPU));
  queue_def->set_device_id(1);

  queue_def = model_request.mutable_status_queues()->add_output_queues_attrs();
  queue_def->set_queue_id(2);
  queue_def->set_device_type(static_cast<int32_t>(NPU));
  queue_def->set_device_id(0);

  EXPECT_EQ(ExecutorContext::AttachQueues(model_request), SUCCESS);
}

TEST_F(ExecutorContextTest, TestModelHandle) {
  ExecutorContext context;
  auto model_ptr = MakeShared<FlowModel>();
  context.AddLocalModel(0, 0, model_ptr);
  auto model = context.GetLocalModel(0, 0);
  EXPECT_NE(model, nullptr);
  auto model2 = context.GetLocalModel(0, 1);
  EXPECT_EQ(model2, nullptr);
  context.RemoveLocalModel(0);
}
}  // namespace ge