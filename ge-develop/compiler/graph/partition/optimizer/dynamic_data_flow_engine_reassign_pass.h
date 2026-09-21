/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_PARTITION_OPTIMIZER_DYNAMIC_DATA_FLOW_ENGINE_REASSIGN_PASS_H_
#define GRAPH_PARTITION_OPTIMIZER_DYNAMIC_DATA_FLOW_ENGINE_REASSIGN_PASS_H_

#include "graph/partition/engine_place.h"

namespace ge {
class DynamicDataFlowEngineReassignPass : public EngineReAssignPass {
 public:
  DynamicDataFlowEngineReassignPass() = default;
  ~DynamicDataFlowEngineReassignPass() override = default;
  Status Run(const ComputeGraphPtr &graph,
             NodeEngineMap &node_atomic_engine_map,
             NodeEngineMap &node_composite_engine_map) override;
};
}  // namespace ge
#endif  // GRAPH_PARTITION_OPTIMIZER_DYNAMIC_DATA_FLOW_ENGINE_REASSIGN_PASS_H_
