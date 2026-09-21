/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_MULTI_THREAD_EXECUTION_DATA_BUILDER_H
#define AIR_CXX_RUNTIME_V2_MULTI_THREAD_EXECUTION_DATA_BUILDER_H
#include "core/executor/sequential/execution_data/sequential_execution_data_builder.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_scheduler_config.h"
#include "core/executor/multi_thread_topological/executor/multi_thread_topological_executor.h"
namespace gert {
class MultiThreadExecutionDataBuilder : public ExecutionDataBuilder {
 public:
  explicit MultiThreadExecutionDataBuilder(GraphExecutorBuilder &executor_builder);
  ResourceGuardPtr Build() override;
  ResourceGuardPtr Build(GraphNode &graph_nodes);

  std::pair<ExeGraphExecutor::ExecuteFunc, ExeGraphExecutor::ExecuteWithCallbackFunc> GetExecuteFunc() const override {
    return {MultiThreadTopologicalExecute, MultiThreadTopologicalExecuteWithCallback};
  }

 private:
  ge::graphStatus CreateExecutionData(GraphNode &graph_node, TopologicalExecutionData *topo_execution_data,
                                      ResourceGuard *resource_guard) const;
 private:
  SequentialExecutionDataBuilder base_ed_builder_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_MULTI_THREAD_EXECUTION_DATA_BUILDER_H
