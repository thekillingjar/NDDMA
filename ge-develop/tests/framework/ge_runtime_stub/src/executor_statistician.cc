/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "check/executor_statistician.h"
#include <sstream>
#include <functional>
#include "exe_graph/runtime/context_extend.h"

namespace gert {
ge::graphStatus ExecutorStatistician::OnEvent(ExecutorEvent event, const ::Node *node, ::KernelStatus result) {
  const std::lock_guard<std::mutex> lk{mtx_};
  if (event != kExecuteEnd) {
    return ge::GRAPH_SUCCESS;
  }
  GE_ASSERT_NOTNULL(node);
  auto kernel = static_cast<const KernelExtendInfo *>(node->context.kernel_extend_info);
  GE_ASSERT_NOTNULL(kernel);

  auto compute_node = static_cast<const ComputeNodeInfo *>(node->context.compute_node_info);
  GE_ASSERT_NOTNULL(compute_node);

  node_names_to_run_stat_[compute_node->GetNodeName()].kernel_types_to_run_count_[kernel->GetKernelType()]++;
  node_types_to_names_[compute_node->GetNodeType()].insert(compute_node->GetNodeName());

  NodeExecuteData node_execute_data;
  node_execute_data.node_name = kernel->GetKernelName();
  node_execute_data.node_type = kernel->GetKernelType();
  node_execute_data.compute_node_name = compute_node->GetNodeName();
  node_execute_data.compute_node_type = compute_node->GetNodeType();
  node_execute_data.execute_index = node_execute_data_list_.size();
  node_execute_data_list_.emplace_back(node_execute_data);
  return ge::GRAPH_SUCCESS;
}
int64_t ExecutorStatistician::GetExecuteIndexByNodeNameAndKernelType(const std::string &compute_node_name,
                                                                     const string &kernel_type) const {
  for (const auto& node_execute_data : node_execute_data_list_) {
    if (node_execute_data.node_type == kernel_type && node_execute_data.compute_node_name == compute_node_name) {
      return node_execute_data.execute_index;
    }
  }
  std::stringstream ss;
  ss << "The compute node name " << compute_node_name << " kernel type " << kernel_type << " does not exists";
  throw std::out_of_range(ss.str());
}
void ExecutorStatistician::OnExecuteEvent(int type, void *void_arg, ExecutorEvent event,
                                          const void *node, KernelStatus result) {
  reinterpret_cast<ExecutorStatistician *>(void_arg)->OnEvent(event, static_cast<const Node *>(node), result);
}
void ExecutorStatistician::Clear() {
  const std::lock_guard<std::mutex> lk{mtx_};
  node_names_to_run_stat_.clear();
  node_execute_data_list_.clear();
}
int64_t ExecutorStatistician::GetExecuteCountByNodeNameAndKernelType(const std::string &compute_node_name,
                                                                     const string &kernel_type) const {
  auto iter = node_names_to_run_stat_.find(compute_node_name);
  if (iter == node_names_to_run_stat_.cend()) {
    return 0;
  }
  auto kt_iter = iter->second.kernel_types_to_run_count_.find(kernel_type);
  if (kt_iter == iter->second.kernel_types_to_run_count_.cend()) {
    return 0;
  }
  return kt_iter->second;
}
int64_t ExecutorStatistician::GetExecuteCountByNodeTypeAndKernelType(const string &node_type,
                                                                     const string &kernel_type) const {
  auto iter = node_types_to_names_.find(node_type);
  if (iter == node_types_to_names_.end()) {
    return 0;
  }
  int64_t total_count = 0;
  for (const auto &node_name : iter->second) {
    total_count += GetExecuteCountByNodeNameAndKernelType(node_name, kernel_type);
  }
  return total_count;
}
void ExecutorStatistician::PrintExecutionSummary() const {
  for (const auto &node_name_and_stat : node_names_to_run_stat_) {
    std::cout << "====================" << std::endl;
    for (const auto &kernel_type_and_count : node_name_and_stat.second.kernel_types_to_run_count_) {
      std::cout << "[" << node_name_and_stat.first << "][" << kernel_type_and_count.first << "]"
                << kernel_type_and_count.second << std::endl;
    }
  }
}
ExecutorStatistician *StartExecutorStatistician(const std::unique_ptr<ModelV2Executor> &executor) {
  return executor->GetSubscribers().AddSubscriber<ExecutorStatistician>();
}
}  // namespace gert
