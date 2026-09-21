/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering_utils.h"
#include "common/checker.h"

namespace gert {
namespace {
const std::string kAttrEngineTaskNodeId = "_engine_task_node_id";
}

ge::NodePtr LoweringUtils::CreateEngineTaskNode(const ge::ComputeGraphPtr &graph, const ge::OpDescPtr op,
                                                const ge::NodePtr origin_node) {
  const auto origin_desc = origin_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(origin_desc);
  const int64_t id = origin_desc->GetId();
  const auto &node = graph->AddNode(op, id);
  if (node != nullptr) {
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    // this ExtAttr is used to identify engine task node
    (void)op_desc->SetExtAttr(kAttrEngineTaskNodeId, id);
    // new node should follow same stream of origin node
    op_desc->SetStreamId(origin_desc->GetStreamId());
    return node;
  }
  return nullptr;
}

bool LoweringUtils::IsEngineTaskNode(const ge::NodePtr &node) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  return (op_desc->GetExtAttr<int64_t>(kAttrEngineTaskNodeId) != nullptr);
}
}  // namespace gert
