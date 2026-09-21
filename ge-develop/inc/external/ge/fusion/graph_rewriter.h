/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GRAPH_FUSION_GRAPH_REWRITER_H
#define INC_EXTERNAL_GE_GRAPH_FUSION_GRAPH_REWRITER_H

#include "graph/graph.h"
#include "subgraph_boundary.h"
#include "ge_common/ge_api_types.h"

namespace ge {
namespace fusion {
class SubgraphRewriter {
 public:
  /**
   * 给定SubgraphBoundary，将边界内算子替换为replacement
   * @param subgraph
   * @param replacement
   * @return
   */
  static Status Replace(const SubgraphBoundary &subgraph, const Graph &replacement);

  /**
   * 给定SubgraphBoundary，将边界内算子替换为replacement
   * @param subgraph
   * @param replacement
   * @return
   */
  static Status Replace(const SubgraphBoundary &subgraph, Graph &&replacement);
};
}  // namespace fusion
} // namespace ge

#endif  // INC_EXTERNAL_GE_GRAPH_FUSION_PATTERN_MATCHER_H
