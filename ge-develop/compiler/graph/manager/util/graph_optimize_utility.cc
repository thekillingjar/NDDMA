/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/util/graph_optimize_utility.h"
#include <memory>
#include "graph/preprocess/graph_prepare.h"
#include "graph/preprocess/multi_batch_copy_graph.h"
#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_compute_pass.h"
#include "graph/passes/standard_optimize/constant_folding/replace_with_empty_const_pass.h"

namespace ge {
GraphOptimizeUtility::GraphOptimizeUtility() {}
GraphOptimizeUtility::~GraphOptimizeUtility() {}

Status GraphOptimizeUtility::InferShape(const ComputeGraphPtr &compute_graph) {
  auto compute_graph_ptr = compute_graph;
  return GraphPrepare::InferShapeForPreprocess(compute_graph_ptr, nullptr, nullptr);
}

Status GraphOptimizeUtility::MultiDimsProcess(const ComputeGraphPtr &compute_graph) {
  return multibatch::ProcessMultiBatch(compute_graph, compute_graph->GetSessionID());
}

Status GraphOptimizeUtility::ConstantFolding(NodePtr &node) {
  std::vector<std::shared_ptr<BaseNodePass>> pass_list;
  pass_list.emplace_back(std::make_shared<ConstantFoldingPass>());
  pass_list.emplace_back(std::make_shared<DimensionComputePass>());
  pass_list.emplace_back(std::make_shared<ReplaceWithEmptyConstPass>());
  for (auto &pass : pass_list) {
    GE_ASSERT_SUCCESS(pass->Run(node));
    if (pass->GetNodesDeleted().count(node) > 0) {
      return SUCCESS;
    }
  }
  return NOT_CHANGED;
}
}  // namespace ge
