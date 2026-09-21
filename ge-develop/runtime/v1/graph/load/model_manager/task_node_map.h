/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_TASK_NODE_MAP_H_
#define AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_TASK_NODE_MAP_H_
#include <cstdint>
#include "graph/node.h"
#include "ge_common/ge_api_error_codes.h"
namespace ge {
class TaskNodeMap {
 public:
  struct NodeInfo {
    int64_t node_id;
    NodePtr node;
  };

 public:
  Status Init(const ComputeGraphPtr &graph, size_t task_num);
  Status AddRelation(size_t task_index, int64_t node_id);

  const NodeInfo &FindNodeByTaskIndex(size_t task_index) const;
  const std::vector<size_t> &FindTasksByNodeId(int64_t node_id) const;

 private:
  std::vector<NodeInfo> task_indexes_to_node_;
  std::map<int64_t, std::vector<size_t>> node_ids_to_task_indexes_;
  std::map<int64_t, NodePtr> node_ids_to_node_;
};
}

#endif  // AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_TASK_NODE_MAP_H_
