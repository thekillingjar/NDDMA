/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_OPTIMIZER_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_OPTIMIZER_H_

#include "graph/compute_graph.h"
#include "runtime/base.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class L2Optimizer {
 public:
  explicit L2Optimizer(const std::string &engine_name);

  ~L2Optimizer();
  Status GetL2DataAlloc(ge::ComputeGraph &stream_graph, uint64_t mem_base, const int64_t &stream_id) const;
  Status UpdateInputForL2Fusion(const ge::ComputeGraph &stream_graph) const;
  Status UpdateDDRForL2Fusion(const ge::ComputeGraph &stream_graph, uint64_t mem_base) const;

 private:
  std::string engine_name_;
};
using L2OptimizerPtr = std::shared_ptr<L2Optimizer>;
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_OPTIMIZER_H_
