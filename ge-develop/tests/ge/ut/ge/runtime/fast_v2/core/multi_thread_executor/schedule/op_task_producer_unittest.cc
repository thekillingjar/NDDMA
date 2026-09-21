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
class OpTaskProducerUnitTest : public testing::Test {
  void SetUp() override {
    KernelSpy::GetInstance().Clear();
  }
};

TEST_F(OpTaskProducerUnitTest, should_get_op_task_success) {
  FakeExecutionData executionData(10);
  executionData
      .KernelAttr({{1, {"conv2d", "AllocMemHbm"}},
                   {2, {"conv2d", "Tiling"}},
                   {3, {"conv2d", "Launch"}},
                   {4, {"transdata", "AllocMemHbm"}},
                   {5, {"transdata", "launch"}},
                   {6, {"netoutput", "Output"}}})
      .Chain({1, 2, 3, 4, 5, 6})
      .StartNodes({1});

  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::OP, 10});
  ASSERT_EQ(producer->Prepare(executionData.Data()), ge::SUCCESS);
  ASSERT_EQ(producer->StartUp(), ge::SUCCESS);
  // step1
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(1, 2, 3);
  producer->Recycle(package);

  // step2
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(4, 5);
  producer->Recycle(package);

  // step3
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(6);
  producer->Recycle(package);
  delete producer;
}

TEST_F(OpTaskProducerUnitTest, should_get_op_task_for_multi_edge_success) {
  FakeExecutionData executionData(10);
  executionData
      .KernelAttr({{1, {"conv2d", "AllocMemHbm"}},
                   {2, {"conv2d", "Launch"}},
                   {3, {"transdata", "AllocMemHbm"}},
                   {4, {"transdata", "Launch"}},
                   {5, {"dynamic", "AllocMemHbm"}},
                   {6, {"dynamic", "Launch"}}})
      .Chain({1, 2, 5})
      .Chain({3, 4, 5})
      .Chain({5, 6})
      .StartNodes({1, 3});

  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::OP, 10});
  ASSERT_EQ(producer->Prepare(executionData.Data()), ge::SUCCESS);
  ASSERT_EQ(producer->StartUp(), ge::SUCCESS);
  // step1
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 2);
  TaskRun(package);

  KERNEL_RUN_EXPECT(1, 2, 3, 4);
  producer->Recycle(package);

  // step2
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);
  KERNEL_RUN_EXPECT(5, 6);
  producer->Recycle(package);

  // step3
  package = producer->Produce();
  ASSERT_EQ(package.size(), 0);
  delete producer;
}

TEST_F(OpTaskProducerUnitTest, should_get_op_task_for_kernel_run_order_success) {
  FakeExecutionData executionData(10);
  executionData
      .KernelAttr({{1, {"conv2d", "AllocMemHbm"}},
                   {2, {"conv2d", "Tiling"}},
                   {3, {"conv2d", "Launch"}},
                   {4, {"conv2d", "CalcSize"}},
                   {5, {"Netoutput", "Output"}}})
      .Chain({1, 2, 3})
      .Chain({1, 4, 3})
      .Chain({3, 5})
      .StartNodes({1});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::OP, 10});
  ASSERT_EQ(producer->Prepare(executionData.Data()), ge::SUCCESS);
  ASSERT_EQ(producer->StartUp(), ge::SUCCESS);

  // step1
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(1, 2, 4);
  producer->Recycle(package);

  // step2
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(3);
  producer->Recycle(package);

  // step3
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(5);
  producer->Recycle(package);
  delete producer;
}

TEST_F(OpTaskProducerUnitTest, should_get_multi_op_task_success) {
  FakeExecutionData executionData(10);
  executionData
      .KernelAttr({{1, {"conv2d", "InferShape"}},
                   {2, {"conv2d", "Tiling"}},
                   {3, {"conv2d", "Launch"}},
                   {4, {"conv2d", "AllocMemHbm"}},
                   {5, {"transdata", "InferShape"}},
                   {6, {"transdata", "Tiling"}},
                   {7, {"transdata", "Launch"}},
                   {8, {"Netoutput", "Output"}}})
      .Chain({1, 2, 3})
      .Chain({1, 4, 3})
      .Chain({5, 6, 7})
      .Chain({1, 5})
      .Chain({3, 7})
      .Chain({7, 8})
      .StartNodes({1});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::OP, 10});
  producer->Prepare(executionData.Data());
  ASSERT_EQ(producer->StartUp(), ge::SUCCESS);

  // step1
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(1, 2, 4, 3);
  producer->Recycle(package);

  // step2
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(5, 6, 7);
  producer->Recycle(package);

  // step3
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(8);
  producer->Recycle(package);
  delete producer;
}

TEST_F(OpTaskProducerUnitTest, task_hasloop_success) {
  FakeExecutionData executionData(10);
  executionData
      .KernelAttr({{1, {"transdata", "Tiling"}},
                   {2, {"transdata", "Tiling"}},
                   {3, {"transdata", "AllocMemHbm"}},
                   {4, {"transdata", "AtomicLaunchKernelWithFlag"}},
                   {5, {"transdata", "LaunchKernelWithHandle"}}})
      .Chain({1, 2, 3, 4, 5})
      .Chain({1, 3})
      .Chain({2, 4})
      .StartNodes({1});
  auto producer = TaskProducerFactory::GetInstance().Create(TaskProducerConfig{TaskProducerType::OP, 10});
  producer->Prepare(executionData.Data());
  ASSERT_EQ(producer->StartUp(), ge::SUCCESS);

  // step1
  TaskPackage package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(1, 2);
  producer->Recycle(package);

  // step2
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(3);
  producer->Recycle(package);

  // step3
  package = producer->Produce();
  ASSERT_EQ(package.size(), 1);
  TaskRun(package);

  KERNEL_RUN_EXPECT(4, 5);
  producer->Recycle(package);
  delete producer;
}
