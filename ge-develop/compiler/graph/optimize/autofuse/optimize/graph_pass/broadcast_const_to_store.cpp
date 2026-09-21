/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "broadcast_const_to_store.h"
#include "attr_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "graph_utils.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "node_utils.h"

using namespace ascir;
using namespace ge::ascir_op;
using namespace ge::ops;

namespace optimize {
Status BroadcastConstToStorePass::RunPass(ge::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!IsOps<Store>(node) || !ge::ascir::AscTensorUtils::IsConstTensor(node->inputs[0])) {
      continue;
    }
    const auto const_node = ge::ascir::AscTensorUtils::GetOwner(node->inputs[0]);
    if (const_node == nullptr) {
      continue;
    }
    const std::string node_name = "scalar_broadcast_" + node->GetName();
    Broadcast scalar_broadcast(node_name.c_str());

    ge::AscNodePtr broadcast_node = graph.AddNode(scalar_broadcast);
    GE_ASSERT_NOTNULL(broadcast_node);
    ge::GraphUtils::RemoveEdge(const_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(broadcast_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), broadcast_node->GetInDataAnchor(0));
    scalar_broadcast.attr.sched = node->attr.sched;
    scalar_broadcast.attr.api.compute_type = ge::ComputeType::kComputeBroadcast;
    scalar_broadcast.attr.api.type = ge::ApiType::kAPITypeCompute;
    scalar_broadcast.y.dtype = static_cast<ge::DataType>(node->inputs[0].attr.dtype);
    *scalar_broadcast.y.axis = node->outputs[0].attr.axis;
    *scalar_broadcast.y.repeats = node->outputs[0].attr.repeats;
    *scalar_broadcast.y.strides = node->outputs[0].attr.strides;
  }
  return ge::SUCCESS;
}
}  // namespace optimize
