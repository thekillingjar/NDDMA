/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_WEIGHT_PREFETCH_WEIGHT_PREFETCH_UTILS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_WEIGHT_PREFETCH_WEIGHT_PREFETCH_UTILS_H_

#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "graph/compute_graph.h"

namespace fe {
class WeightPrefetchUtils {
 public:
  static Status HandleWeightPrefetch(const ge::ComputeGraph &graph);

 private:
  static bool IsGraphWeightPrefetch(const ge::ComputeGraph &graph);
  static bool HasWeightPrefetchAttr(const ge::OpDescPtr &op_desc);
  static Status HandleGraphPrefetch(const ge::ComputeGraph &graph);
  static Status HandleNodesPrefetch(const ge::ComputeGraph &graph);
  static Status SetNodePrefetchAttr(const ge::NodePtr &prefetch_node, const ge::ComputeGraph &graph);
  static Status SetWeightPrefetchAttr(const ge::OpDescPtr &op_desc,
                                      const std::vector<ge::ConstGeTensorPtr> &weights);
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_WEIGHT_PREFETCH_WEIGHT_PREFETCH_UTILS_H_
