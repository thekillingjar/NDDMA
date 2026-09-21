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
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_type.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_producer_config.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_package.h"
#include "fake_execution_data.h"

using namespace gert;
class SingleTaskProducerUnitTest : public testing::Test {
  void SetUp() override {
    KernelSpy::GetInstance().Clear();
  }
};

TEST_F(SingleTaskProducerUnitTest, create_producer_failed) {
  FakeExecutionData executionData(2);
  executionData.Chain({0, 1}).StartNodes({0});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::MAX, 10});
  ASSERT_EQ(producer, nullptr);
  delete producer;
}

TEST_F(SingleTaskProducerUnitTest, should_run_all_kernel_for_two_nodes) {
  FakeExecutionData executionData(2);
  executionData.Chain({0, 1}).StartNodes({0});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::SINGLE, 10});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(0, 1);
  delete producer;
}

TEST_F(SingleTaskProducerUnitTest, should_run_all_kernel_for_five_normal_nodes) {
  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6}).Chain({7, 8, 9}).StartNodes({3});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::SINGLE, 10});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(3, 7, 6, 8, 9);
  producer->Recycle(package);
  delete producer;
}
