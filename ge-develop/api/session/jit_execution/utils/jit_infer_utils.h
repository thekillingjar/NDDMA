/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_JIT_INFER_UTILS_H
#define CANN_GRAPH_ENGINE_JIT_INFER_UTILS_H

#include "ge/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/preprocess/graph_prepare.h"
#include "graph/optimize/graph_optimize.h"
#include "graph/utils/graph_utils_ex.h"
#include "exe_graph/runtime/tensor.h"

namespace ge {
class JitInferUtils {
 public:
  static Status InferSymbolForGraph(const ComputeGraphPtr &graph, const std::vector<GeTensor> &inputs,
                                    std::vector<NodePtr> &infered_nodes);

  static Status PrepareBeforeInferSymbol(const ComputeGraphPtr &graph, const std::vector<GeTensor> &inputs);

  static Status InferGraphAndGetInferredNodes(const ComputeGraphPtr &graph, const std::vector<GeTensor> &inputs,
                                              std::vector<NodePtr> &infered_nodes);
};
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_JIT_INFER_UTILS_H
