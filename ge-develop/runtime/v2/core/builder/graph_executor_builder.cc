/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_executor_builder.h"

#include <utility>
#include "register/kernel_registry.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "core/builder/node_types.h"
#include "execution_data_builder/execution_data_builder.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/executor/topological/execution_data/topological_execution_data_builder.h"
#include "core/executor/priority_topological/execution_data/priority_topological_execution_data_builder.h"
#include "core/executor/topological/executor/topological_executor.h"
#include "core/executor/priority_topological/executor/priority_topological_executor.h"
#include "core/executor/sequential/executor/sequential_executor.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data_builder.h"
#include "core/executor/multi_thread_topological/executor/multi_thread_topological_executor.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"

namespace gert {
namespace {
const char_t *GetExecutorTypeStr(ExecutorType e_type) {
  constexpr static const char_t *kTypeNames[static_cast<size_t>(ExecutorType::kEnd) + 1U] = {
      "sequential-executor", "general-topo-executor", "priority-topo-executor", "multi-thread-topo-executor",
      "unknown"};
  if (e_type > ExecutorType::kEnd) {
    e_type = ExecutorType::kEnd;
  }
  return kTypeNames[static_cast<size_t>(e_type)];
}

struct ExecutorSelectionArgs {
  bool has_ctrl_node;
  size_t launch_num;
};
void ExtractImportantInfo(const ge::ExecuteGraphPtr &exe_graph, ExecutorSelectionArgs &arg) {
  for (const auto node : exe_graph->GetAllNodes()) {
    const auto node_type = node->GetTypePtr();
    if (!arg.has_ctrl_node) {
      if (IsIfOrCaseType(node_type) || IsWhileType(node_type)) {
        arg.has_ctrl_node = true;
        continue;
      }
    }
    if (IsLaunchNode(node_type)) {
      arg.launch_num++;
      continue;
    }
  }
}
ExecutorType SelectExecutor(const ge::ExecuteGraphPtr &exe_graph, const ExecutorOption *executor_option) {
  if (executor_option && (executor_option->GetExecutorType() != ExecutorType::kEnd)) {
    return executor_option->GetExecutorType();
  }
  constexpr size_t kLaunchNumLimit = 5UL;
  ExecutorSelectionArgs arg = {false, 0UL};
  ExtractImportantInfo(exe_graph, arg);

  if (!arg.has_ctrl_node) {
    return ExecutorType::kSequentialPriority;
  }
  if (arg.launch_num >= kLaunchNumLimit) {
    return ExecutorType::kTopologicalPriority;
  }
  return ExecutorType::kTopological;
}

// refactoring to factory if we need
std::unique_ptr<ExecutionDataBuilder> CreateExecutionDataBuilder(ExecutorType executor_type,
                                                                 GraphExecutorBuilder &executor_builder) {
  switch (executor_type) {
    case ExecutorType::kSequentialPriority:
      return ge::MakeUnique<SequentialExecutionDataBuilder>(executor_builder);
    case ExecutorType::kTopological:
      return ge::MakeUnique<TopologicalExecutionDataBuilder>(executor_builder);
    case ExecutorType::kTopologicalPriority:
      return ge::MakeUnique<PriorityTopologicalExecutionDataBuilder>(executor_builder);
    case ExecutorType::kTopologicalMultiThread:
      return ge::MakeUnique<MultiThreadExecutionDataBuilder>(executor_builder);
    default:
      return nullptr;
  }
}

void ReleaseOutputs(const TopologicalExecutionData *ed) {
  for (size_t i = 0U; i < ed->base_ed.output_num; ++i) {
    reinterpret_cast<Chain *>(ed->base_ed.output_values[i])->Set(nullptr, nullptr);
  }
}
}  // namespace
ge::graphStatus GraphExecutorBuilder::Build(ExeGraphExecutor &executor) {
  // select a executor
  auto executor_type = SelectExecutor(exe_graph_, executor_option_);
  GE_ASSERT_TRUE(executor_type != ExecutorType::kEnd, "Failed to select a executor for exe graph %s",
                 exe_graph_->GetName().c_str());
  GELOGI("Select executor type %s for graph %s", GetExecutorTypeStr(executor_type), exe_graph_->GetName().c_str());

  return Build(executor_type, executor);
}
ge::graphStatus GraphExecutorBuilder::Build(ExecutorType executor_type, ExeGraphExecutor &executor) {
  auto execution_data_builder = CreateExecutionDataBuilder(executor_type, *this);
  GE_ASSERT_NOTNULL(execution_data_builder);

  auto resource_guard = execution_data_builder->Build();
  GE_ASSERT_NOTNULL(resource_guard);
  auto execution_data = resource_guard->GetExecutionData();
  GE_ASSERT_NOTNULL(execution_data);
  executor.SetExecutionData(execution_data, std::move(resource_guard));

  auto funcs = execution_data_builder->GetExecuteFunc();
  executor.SetExecuteFunc(funcs.first, funcs.second);

  // todo 应该不需要了，在OutputData直连NetOutput后，OutputData的输出本身就没有初始化过，所以也不需要释放
  ReleaseOutputs(reinterpret_cast<const TopologicalExecutionData *>(execution_data));
  return ge::GRAPH_SUCCESS;
}

GraphExecutorBuilder &GraphExecutorBuilder::ExecutorOpt(const ExecutorOption &option) {
  executor_option_ = const_cast<ExecutorOption *>(&option);
  return *this;
}

GraphExecutorBuilder::GraphExecutorBuilder(const ModelLevelData &model_level_data, ge::ExecuteGraphPtr exe_graph,
                                           SymbolsToValue *symbols_to_value)
    : model_level_data_(model_level_data),
      exe_graph_(std::move(exe_graph)),
      symbols_to_value_(symbols_to_value) {}
}  // namespace gert
