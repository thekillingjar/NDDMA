/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_TENSORFLOW_SCOPE_SCOPE_PASS_MANAGER_H_
#define PARSER_TENSORFLOW_SCOPE_SCOPE_PASS_MANAGER_H_

#include <vector>
#include "register/scope/scope_fusion_pass_register.h"
#include "proto/tensorflow/graph.pb.h"

namespace ge {
/**
 * @ingroup domi_omg
 * @brief manage passes
 */
class ScopePassManager {
 public:
  ScopePassManager() : scope_graph_(nullptr) {}
  ScopePassManager(const ScopePassManager &scope_pass_manager) = delete;
  ScopePassManager &operator=(const ScopePassManager &scope_pass_manager) = delete;
  ~ScopePassManager() {}

  std::shared_ptr<ScopeGraph> BuildScopeGraph(domi::tensorflow::GraphDef *graph_def);

  domi::Status AddPass(std::unique_ptr<ScopeBasePass> &pass);
  domi::Status Run(std::shared_ptr<ScopeGraph> &graph);

  std::shared_ptr<ScopeGraph> scope_graph_;

 private:
  std::vector<std::unique_ptr<ScopeBasePass>> graph_passes_;
};
}  // namespace ge

#endif  // PARSER_TENSORFLOW_SCOPE_SCOPE_PASS_MANAGER_H_
