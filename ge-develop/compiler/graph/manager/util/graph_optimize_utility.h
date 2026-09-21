/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZE_UTILITY_H_
#define GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZE_UTILITY_H_

#include "common/optimizer/optimize_utility.h"
#include "ge/ge_api_types.h"
#include "graph/manager/util/graph_rebuild_state_ctrl.h"

namespace ge {
class GraphOptimizeUtility : public OptimizeUtility {
 public:
  GraphOptimizeUtility();
  ~GraphOptimizeUtility() override;

  // some functions required session_id on compute_graph, which set when graph_prepare init.
  // so every func which invoke InferShape need keep position after graph_prepare init.
  Status InferShape(const ComputeGraphPtr &compute_graph) override;

  // multi dim and pre/post process
  Status MultiDimsProcess(const ComputeGraphPtr &compute_graph) override;

  Status ConstantFolding(NodePtr &node) override;
};
}  // namespace ge
#endif  // GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZE_UTILITY_H_
