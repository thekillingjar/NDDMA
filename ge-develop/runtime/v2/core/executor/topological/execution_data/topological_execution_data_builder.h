/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_TOPOLOGICAL_EXECUTION_DATA_BUILDER_H_
#define AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_TOPOLOGICAL_EXECUTION_DATA_BUILDER_H_
#include "core/executor/sequential/execution_data/sequential_execution_data_builder.h"
#include "core/builder/graph_node.h"
#include "core/executor/topological/executor/topological_executor.h"
namespace gert {
class TopologicalExecutionDataBuilder : public ExecutionDataBuilder {
 public:
  explicit TopologicalExecutionDataBuilder(GraphExecutorBuilder &executor_builder);

  /**
   * 是否按照优先级执行
   * @param flag 为true代表需要按照优先级执行，生成优先级执行的ExecutionData，默认为false
   * @return
   */
  TopologicalExecutionDataBuilder &PriorityExecution(bool flag);

  ResourceGuardPtr Build() override;
  ResourceGuardPtr Build(GraphNode &graph_nodes);
  ge::graphStatus Build(TopologicalExecutionData *execution_data, ResourceGuard *resource_guard,
                        GraphNode &graph_nodes);

  std::vector<std::pair<ge::FastNode *, Node *>> &GetOrderedGraphToExeNodes() {
    return base_ed_builder_.GetOrderedGraphToExeNodes();
  }

  std::pair<ExeGraphExecutor::ExecuteFunc, ExeGraphExecutor::ExecuteWithCallbackFunc> GetExecuteFunc() const override {
    return {TopologicalExecute, TopologicalExecuteWithCallback};
  }

 private:
  ge::graphStatus CreateExecutionData(GraphNode &graph_node, TopologicalExecutionData *execution_data,
                                      ResourceGuard *resource_guard) const;

 private:
  bool priority_execution_ = false;
  SequentialExecutionDataBuilder base_ed_builder_;
};
}

#endif  // AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_TOPOLOGICAL_EXECUTION_DATA_BUILDER_H_
