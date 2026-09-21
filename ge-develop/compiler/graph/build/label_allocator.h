/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LABEL_ALLOCATOR_H_
#define GE_GRAPH_LABEL_ALLOCATOR_H_

#include <set>

#include "graph/node.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
class LabelAllocator {
 public:
  explicit LabelAllocator(const ComputeGraphPtr &graph);
  ~LabelAllocator() = default;

  Status AssignFunctionalLabels();

 private:
  bool CollectFunctionalNode(ComputeGraphPtr &graph, std::set<NodePtr> &functional_nodes) const;

  ComputeGraphPtr compute_graph_;
};
}  // namespace ge
#endif  // GE_GRAPH_LABEL_ALLOCATOR_H_