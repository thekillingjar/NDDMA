/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_OPTIMIZE_STREAM_GRAPH_H_
#define GE_GRAPH_BUILD_OPTIMIZE_STREAM_GRAPH_H_

#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"

namespace ge {
class StreamGraphOptimizer {
 public:
  StreamGraphOptimizer() = default;

  StreamGraphOptimizer(const StreamGraphOptimizer &) = delete;

  StreamGraphOptimizer &operator=(const StreamGraphOptimizer &) = delete;

  virtual ~StreamGraphOptimizer();

  Status OptimizeStreamedSubGraph(const ComputeGraphPtr &comp_graph, Graph2SubGraphInfoList &subgraph_map,
                                  struct RunContext &run_context) const;

  Status OptimizeStreamedWholeGraph(const ComputeGraphPtr &comp_graph) const;

 private:
  void RefreshNodeId(const ComputeGraphPtr &comp_graph, Graph2SubGraphInfoList &subgraph_map) const;

  bool IsSameStreamIdOrBatchLabel(const ComputeGraphPtr &comp_graph) const;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_OPTIMIZE_STREAM_GRAPH_H_
