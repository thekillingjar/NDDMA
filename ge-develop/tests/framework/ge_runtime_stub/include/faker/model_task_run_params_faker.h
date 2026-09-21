/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MODEL_TASK_RUN_PARAMS_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MODEL_TASK_RUN_PARAMS_FAKER_H_
#include "task_run_param_faker.h"
#include "task_node_map_faker.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include <vector>
namespace ge {
class ModelTaskRunParamsFaker {
 public:
  struct ParamsAndMap {
    std::vector<TaskRunParam> params;
    TaskNodeMap tnm;
    std::unordered_map<std::string, size_t> node_names_to_task_index;
    std::unordered_map<size_t, std::string> task_indexes_to_node_name;
  };

 public:
  explicit ModelTaskRunParamsFaker(const ComputeGraphPtr &graph) : graph_(graph) {}

  ModelTaskRunParamsFaker &Param(std::string node_name, TaskRunParamFaker &&faker) {
    node_names_to_faker_[std::move(node_name)] = std::move(faker);
    return *this;
  }

  ModelTaskRunParamsFaker::ParamsAndMap Build() const {
    std::list<NodePtr> task_nodes;
    for (const auto &node : graph_->GetAllNodes()) {
      if (node->GetType() == "Data" || node->GetType() == "NetOutput" || node->GetType() == "Const" ||
          node->GetType() == "PhonyConcat") {
        continue;
      }
      task_nodes.emplace_back(node);
    }

    ParamsAndMap params_and_map{};
    auto &tnm = params_and_map.tnm;
    tnm.Init(graph_, task_nodes.size());
    auto &params = params_and_map.params;
    params.reserve(task_nodes.size());
    for (const auto &node : task_nodes) {
      auto task_index = params.size();
      tnm.AddRelation(task_index, node->GetOpDesc()->GetId());
      params_and_map.node_names_to_task_index[node->GetName()] = task_index;
      params_and_map.task_indexes_to_node_name[task_index] = node->GetName();
      params.emplace_back(node_names_to_faker_.at(node->GetName()).Build());
    }
    return params_and_map;
  }

 private:
  const ComputeGraphPtr graph_;
  std::unordered_map<std::string, TaskRunParamFaker> node_names_to_faker_;
};
}  // namespace ge

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MODEL_TASK_RUN_PARAMS_FAKER_H_
