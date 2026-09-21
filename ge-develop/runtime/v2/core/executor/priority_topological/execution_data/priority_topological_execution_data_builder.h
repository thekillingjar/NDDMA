/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_PRIORITY_TOPOLOGICAL_EXECUTION_DATA_BUILDER_H_
#define AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_PRIORITY_TOPOLOGICAL_EXECUTION_DATA_BUILDER_H_
#include "core/executor/topological/execution_data/topological_execution_data_builder.h"
#include "core/executor/priority_topological/executor/priority_topological_executor.h"
namespace gert {
class PriorityTopologicalExecutionDataBuilder : public TopologicalExecutionDataBuilder {
 public:
  explicit PriorityTopologicalExecutionDataBuilder(GraphExecutorBuilder &executor_builder)
      : TopologicalExecutionDataBuilder(executor_builder) {
    PriorityExecution(true);
  }

  std::pair<ExeGraphExecutor::ExecuteFunc, ExeGraphExecutor::ExecuteWithCallbackFunc> GetExecuteFunc() const override {
    return {PriorityTopologicalExecute, PriorityTopologicalExecuteWithCallback};
  }
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_PRIORITY_TOPOLOGICAL_EXECUTION_DATA_BUILDER_H_
