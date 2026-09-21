/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_SEQUENTIAL_EXECUTION_DATA_BUILDER_H_
#define AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_SEQUENTIAL_EXECUTION_DATA_BUILDER_H_
#include "core/builder/execution_data_builder/execution_data_builder.h"
#include "core/executor/sequential/executor/sequential_executor.h"
#include "sequential_execution_data.h"
#include "core/builder/graph_node.h"
#include "common/plugin/ge_make_unique_util.h"

namespace gert {
template <typename T>
std::unique_ptr<uint8_t[]> CreateCArray(const std::vector<T> &vec) {
  auto total_size = sizeof(T) * vec.size();
  if (total_size == 0U) {
    return nullptr;
  }
  std::unique_ptr<uint8_t[]> guard = ge::MakeUnique<uint8_t[]>(total_size);
  if (guard == nullptr) {
    return nullptr;
  }
  if (total_size > 0U && memcpy_s(guard.get(), total_size, vec.data(), total_size) != EOK) {
    return nullptr;
  }
  return guard;
}
template <typename T>
std::unique_ptr<uint8_t[]> CopyCArray(T *vec, size_t vec_size) {
  auto total_size = sizeof(T) * vec_size;
  if (total_size == 0U) {
    return nullptr;
  }
  std::unique_ptr<uint8_t[]> guard = ge::MakeUnique<uint8_t[]>(total_size);
  if (guard == nullptr) {
    return nullptr;
  }
  if (total_size > 0U && memcpy_s(guard.get(), total_size, vec, total_size) != EOK) {
    return nullptr;
  }
  return guard;
}
class SequentialExecutionDataBuilder : public ExecutionDataBuilder {
 public:
  explicit SequentialExecutionDataBuilder(GraphExecutorBuilder &executor_builder);

  ResourceGuardPtr Build() override;
  std::vector<std::pair<ge::FastNode *, Node *>> &GetOrderedGraphToExeNodes() {
    return ordered_graph_to_exe_nodes_;
  }

  std::pair<ExeGraphExecutor::ExecuteFunc, ExeGraphExecutor::ExecuteWithCallbackFunc> GetExecuteFunc() const override {
    return {SequentialExecute, SequentialExecuteWithCallback};
  }

  ge::graphStatus Build(SequentialExecutionData *execution_data, ResourceGuard *resource_guard, GraphNode &graph_nodes);

  /**
   * 是否基于执行节点的优先级重排ExecutionData中的Node次序。
   * 理论上可以无脑重排的，不过为了减少一次上库的风险，提供此选项，在general-topo执行器时设置为false
   * @param flag 为true时代表按照优先级重排，默认为true
   * @return
   */
  SequentialExecutionDataBuilder &ReOrderByPriority(bool flag);

  ge::graphStatus AllocGraphAsyncValues(const std::vector<ge::FastNode *> &graph_nodes,
                                        GraphAsyncValue &graph_async_value);

 private:
  ge::graphStatus ReadInNodes(GraphNode &graph_nodes, ResourceGuard *resource_guard);
  ge::graphStatus ReadInNodesByPriority(GraphNode &graph_nodes, ResourceGuard *resource_guard);
  ge::graphStatus InitAllKernelExtendInfos(
      const std::vector<std::pair<ge::FastNode *, Node *>> &graph_nodes_to_executor_node) const;

  ge::graphStatus CreateExecutionData(GraphNode &graph_node, SequentialExecutionData *execution_data,
                                      ResourceGuard *resource_guard) const;
  ge::graphStatus ReadInOneNode(ge::FastNode *const node, GraphNode &graph_nodes, ResourceGuard *resource_guard);

 private:
  bool re_order_by_priority_ = true;
  GraphAsyncValue graph_async_value_;
  std::vector<std::pair<ge::FastNode *, Node *>> ordered_graph_to_exe_nodes_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_SEQUENTIAL_EXECUTION_DATA_BUILDER_H_
