/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_EXECUTOR_STATISTICIAN_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_EXECUTOR_STATISTICIAN_H_
#include "runtime/gert_api.h"
#include "core/execution_data.h"
#include "runtime/subscriber/executor_subscriber_c.h"
#include "runtime/model_v2_executor.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstdlib>
namespace gert {
struct ComputeNodeRunStatistics {
  std::unordered_map<std::string, int64_t> kernel_types_to_run_count_;
};
class ExecutorStatistician {
 public:
  struct NodeExecuteData {
    std::string node_name;
    std::string node_type;
    std::string compute_node_name;
    std::string compute_node_type;
    int64_t execute_index;
  };

  static void OnExecuteEvent(int type, void *void_arg, ExecutorEvent event, const void *node, KernelStatus result);

  ge::graphStatus OnEvent(ExecutorEvent event, const ::Node *node, ::KernelStatus result);

  void Clear();
  int64_t GetExecuteCountByNodeNameAndKernelType(const std::string &node_name, const std::string &kernel_type) const;
  int64_t GetExecuteCountByNodeTypeAndKernelType(const std::string &node_type, const std::string &kernel_type) const;
  int64_t GetExecuteIndexByNodeNameAndKernelType(const std::string &compute_node_name, const string &kernel_type) const;
  void PrintExecutionSummary() const;

 private:
  std::vector<NodeExecuteData> node_execute_data_list_;
  std::unordered_map<std::string, ComputeNodeRunStatistics> node_names_to_run_stat_;
  std::unordered_map<std::string, std::unordered_set<std::string>> node_types_to_names_;
  std::mutex mtx_;
};

ExecutorStatistician *StartExecutorStatistician(const std::unique_ptr<ModelV2Executor> &executor);
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_EXECUTOR_STATISTICIAN_H_
