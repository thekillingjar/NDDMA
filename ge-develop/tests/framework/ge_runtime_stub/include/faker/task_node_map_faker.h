/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_TASK_NODE_MAP_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_TASK_NODE_MAP_FAKER_H_
#include "graph/load/model_manager/task_node_map.h"
#include "graph/load/model_manager/task_info/task_info.h"

#include <cstdlib>
#include <map>
#include <vector>
#include <cstdint>

namespace ge {
class TaskNodeMapFaker {
 public:
  explicit TaskNodeMapFaker(const vector<TaskRunParam> &params);
  TaskNodeMap Build() const;
  TaskNodeMapFaker &RangeOneOneMap(size_t start_task_index, size_t count);
  TaskNodeMapFaker &NodeInFromIdentity(size_t task_index, int32_t in_index);
  TaskNodeMapFaker &NodeOutToIdentity(size_t task_index, int32_t out_index);
  TaskNodeMapFaker &NodeType(size_t task_index, std::string node_type);

 public:
  struct IoSet {
    std::set<int32_t> input_indexes;
    std::set<int32_t> output_indexes;
  };

 private:
  ComputeGraphPtr FakeGraph(std::map<size_t, int64_t> &task_indexes_to_node_id,
                            std::map<int64_t, std::vector<size_t>> &node_ids_to_task_indexes) const;
  NodePtr FakeNode(const ComputeGraphPtr &graph, size_t task_index) const;
  std::map<size_t, std::vector<NodePtr>> CreateInIdentityNodes(const ComputeGraphPtr &graph) const;

 private:
  const std::vector<TaskRunParam> &task_indexes_to_param_;
  size_t task_num_{0};
  std::map<size_t, std::set<int32_t>> task_indexes_to_in_identity_indexes_;
  std::map<size_t, std::set<int32_t>> task_indexes_to_out_identity_indexes_;
  std::map<size_t, std::string> task_indexes_to_node_type_;
};
}  // namespace ge

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_TASK_NODE_MAP_FAKER_H_
