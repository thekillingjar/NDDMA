/**
  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_remove_pass.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils.h"

namespace ge {

graphStatus CastRemovePass::Run(const ComputeGraphPtr &graph, bool &changed) {
  GE_CHECK_NOTNULL(graph);

  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != "Cast") {
      continue;
    }

    const auto &input_desc = node->GetOpDesc()->GetInputDesc(0);
    const auto &output_desc = node->GetOpDesc()->GetOutputDesc(0);
    if (input_desc.GetDataType() == output_desc.GetDataType()) {
      GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(node, {0}));
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, node));
      changed = true;
    }
  }

  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != "Cast") {
      continue;
    }
    if (node->GetOutDataNodesSize() != 1UL) {
      continue;
    }

    auto next_node = node->GetOutDataNodes().at(0);
    if (next_node->GetType() != "Cast") {
      continue;
    }

    const auto &cast1_input_dtype = node->GetOpDesc()->GetInputDesc(0).GetDataType();
    const auto &cast2_output_dtype = next_node->GetOpDesc()->GetOutputDesc(0).GetDataType();

    if (cast1_input_dtype == cast2_output_dtype) {
      GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(node, {0}));
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, node));
      GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(next_node, {0}));
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, next_node));
      changed = true;
    }
  }

  return GRAPH_SUCCESS;
}

}  // namespace ge
