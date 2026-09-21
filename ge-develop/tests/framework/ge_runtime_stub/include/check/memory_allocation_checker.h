/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_CHECK_MEMORY_ALLOCATION_CHECKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_CHECK_MEMORY_ALLOCATION_CHECKER_H_
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
enum class MemoryOperationType { kAlloc, kFree, kLaunch, kNormal };
struct MemoryOperation {
  MemoryOperationType op_type;
  const void *address;
  std::string kernel_name;
};
class MemoryAllocationChecker {
 public:
  explicit MemoryAllocationChecker(ge::ExecuteGraphPtr exe_graph);
  static void OnExecuteEvent(int type, void *void_arg, ExecutorEvent event, const void *node, KernelStatus result);

  ge::graphStatus OnEvent(ExecutorEvent event, const ::Node *node, ::KernelStatus result);

  void Clear();

  bool CheckFreeEarlyEnough() const;
  bool CheckAllocLatelyEnough() const;

 private:
  void AddOperation(MemoryOperationType op_type, const void *addr, const char *kernel_name);

 private:
  mutable ge::ExecuteGraphPtr exe_graph_;  // ge::GraphUtils::FindNodeFromAllNodes 不支持const输入
  std::vector<MemoryOperation> memory_operations_;
  std::map<const void *, std::vector<MemoryOperation>> addresses_to_operations_;
  std::unordered_map<std::string, std::set<std::string>> alloc_nodes_to_free_nodes_;
  std::unordered_map<std::string, std::set<std::string>> alloc_nodes_to_normal_nodes_;
  std::unordered_map<std::string, std::unordered_set<std::string>> launch_nodes_to_relevant_alloc_nodes_;
};
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_CHECK_MEMORY_ALLOCATION_CHECKER_H_
