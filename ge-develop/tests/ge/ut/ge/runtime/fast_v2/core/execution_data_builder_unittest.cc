/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "core/executor/sequential/execution_data/sequential_execution_data_builder.h"
#include "core/executor/topological/execution_data/topological_execution_data_builder.h"
#include "core/executor/priority_topological/execution_data/priority_topological_execution_data_builder.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data_builder.h"
#include <gtest/gtest.h>
#include <memory>

namespace gert {
class ExecutionDataBuilderUT : public testing::Test {
protected:
  void SetUp() override {
    executor_builder = std::make_unique<GraphExecutorBuilder>(ModelLevelData{}, nullptr, nullptr);
  }
  std::unique_ptr<GraphExecutorBuilder> executor_builder;
};

TEST_F(ExecutionDataBuilderUT, SequentialBuilder_GetExecuteFunc_Correct) {
  SequentialExecutionDataBuilder execution_data_builder(*executor_builder);
  auto funcs = execution_data_builder.GetExecuteFunc();
  EXPECT_EQ(funcs.first, SequentialExecute);
  EXPECT_EQ(funcs.second, SequentialExecuteWithCallback);
}
TEST_F(ExecutionDataBuilderUT, TopologicalBuilder_GetExecuteFunc_Correct) {
  TopologicalExecutionDataBuilder execution_data_builder(*executor_builder);
  auto funcs = execution_data_builder.GetExecuteFunc();
  EXPECT_EQ(funcs.first, TopologicalExecute);
  EXPECT_EQ(funcs.second, TopologicalExecuteWithCallback);
}
TEST_F(ExecutionDataBuilderUT, PriorityTopologicalBuilder_GetExecuteFunc_Correct) {
  PriorityTopologicalExecutionDataBuilder execution_data_builder(*executor_builder);
  auto funcs = execution_data_builder.GetExecuteFunc();
  EXPECT_EQ(funcs.first, PriorityTopologicalExecute);
  EXPECT_EQ(funcs.second, PriorityTopologicalExecuteWithCallback);
}

TEST_F(ExecutionDataBuilderUT, MultiThreadBuilder_GetExecuteFunc_Correct) {
  MultiThreadExecutionDataBuilder execution_data_builder(*executor_builder);
  auto funcs = execution_data_builder.GetExecuteFunc();
  EXPECT_EQ(funcs.first, MultiThreadTopologicalExecute);
  EXPECT_EQ(funcs.second, MultiThreadTopologicalExecuteWithCallback);
}
}  // namespace gert