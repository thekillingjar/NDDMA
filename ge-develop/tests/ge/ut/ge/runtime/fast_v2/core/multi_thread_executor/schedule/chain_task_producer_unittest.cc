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
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/chain_task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_type.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_producer_config.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_package.h"
#include "core/executor_error_code.h"
#include "fake_execution_data.h"

using namespace gert;

class ChainTaskProducerUnitTest : public testing::Test {
  void SetUp() override {
    KernelSpy::GetInstance().Clear();
  }
};

TEST_F(ChainTaskProducerUnitTest, should_get_one_link_task_success) {
  FakeExecutionData executionData(10);
  executionData.Chain({0, 1}).StartNodes({0});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::CHAIN, 10});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(0, 1);
  producer->Recycle(package);
  delete producer;
}

TEST_F(ChainTaskProducerUnitTest, should_get_one_link_task_failed) {
  FakeExecutionData executionData(10);
  executionData.Chain({0, 1}).StartNodes({0});
  executionData.FuncFailed(1, kStatusFailed);
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::CHAIN, 10});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(0);
  producer->Recycle(package);
  delete producer;
}

TEST_F(ChainTaskProducerUnitTest, should_run_all_kernel_for_two_chains) {
  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6})
      .KernelAttr({{3, {"conv2d", "AllocMemHbm"}},
                   {5, {"conv2d", "Tiling"}},
                   {6, {"conv2d", "Launch"}},
                   {7, {"transdata", "AllocMemHbm"}},
                   {8, {"transdata", "launch"}},
                   {9, {"netoutput", "Output"}}})
      .Chain({5, 8, 9})
      .StartNodes({3, 5});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::CHAIN, 10});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 2);
  TaskRun(package);
  KERNEL_RUN_EXPECT(3, 7, 6, 5, 8, 9);
  producer->Recycle(package);
  delete producer;
}

TEST_F(ChainTaskProducerUnitTest, should_fetch_task_for_multi_step) {
  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 6}).StartNodes({3, 5});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::CHAIN, 10});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  // step1
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 2);
  TaskRun(package);
  KERNEL_RUN_EXPECT(3, 7, 5, 8);

  producer->Recycle(package);

  // step2
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(6);

  producer->Recycle(package);

  // step3
  package = producer->Produce();
  ASSERT_EQ(package.size(), 0);
  delete producer;
}

TEST_F(ChainTaskProducerUnitTest, should_fetch_more_task_for_multi_step) {
  FakeExecutionData executionData(20);
  executionData.Chain({1, 2, 3, 4, 8, 11, 12}).Chain({1, 5, 6, 7, 8}).Chain({7, 9, 10, 12}).StartNodes({1});

  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::CHAIN, 10});
  producer->Prepare(executionData.Data());
  ASSERT_EQ(producer->StartUp(), ge::SUCCESS);

  // step1
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(1, 2, 3, 4, 5, 6, 7, 9, 10);

  producer->Recycle(package);

  // step2
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(8, 11);

  producer->Recycle(package);

  // step3
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(12);

  producer->Recycle(package);

  // step4
  package = producer->Produce();
  ASSERT_EQ(package.size(), 0);
  delete producer;
}

TEST_F(ChainTaskProducerUnitTest, should_fetch_task_but_ont_task_failed_for_multi_step) {
  FakeExecutionData executionData(10);
  executionData.Chain({1, 3, 5}).Chain({2, 4, 5}).StartNodes({1, 2});
  executionData.FuncFailed(3, kStatusFailed);
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::CHAIN, 10});
  producer->Prepare(executionData.Data());
  producer->StartUp();
  // step1
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 2);
  TaskRun(package);
  KERNEL_RUN_EXPECT(1, 2, 4);

  producer->Recycle(package);
  ASSERT_EQ(package.size(), 0);
  delete producer;
}
