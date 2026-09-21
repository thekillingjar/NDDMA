/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "special_node_checker.h"
#include "node_checker_utils.h"
#include "ge_common/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/unfold/graph_unfolder.h"
#include "node_checker_register.h"

namespace ge {
ge::Status SpecialNodeChecker::Check(const ge::ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodesPtr()) {
    // ffts+模式下有些特别的场景，比如PhonyConcat的两个输入都是variable，但是不会插入identity，不满足输入连续的要求，
    // 比如用例ImmutableOutAndNoPaddingContinuousInputInFftsSubgraph_SipInsert_SUCCESS
    if ((node->GetType() == ge::PARTITIONEDCALL) && (node->GetOpDesc() != nullptr)) {
      if (node->GetOpDesc()->HasAttr(ge::ATTR_NAME_FFTS_SUB_GRAPH)
          || node->GetOpDesc()->HasAttr(ge::ATTR_NAME_FFTS_PLUS_SUB_GRAPH)) {
        GELOGI("graph %s has ffts+ subgraph, skip special node memory check.", graph->GetName().c_str());
        return SUCCESS;
      }
    }
    const auto types = NodeCheckerUtils::GetSpecialNodeTypes(node);
    for (const auto &type : types) {
      const auto func = GetSpecialNodeChecker(type);
      if (func != nullptr) {
        NodeCheckerParam param{graph, node, type};
        GE_ASSERT_SUCCESS(func(param), "special node memory check failed. type: %s, graph: %s, node: %s(%s)",
                          SpecialNodeTypeStr(type), graph->GetName().c_str(), NodeCheckerUtils::NodeName(node).c_str(),
                          node->GetTypePtr());
      }
    }
  }
  GELOGI("special node memory check success, graph: %s", graph->GetName().c_str());
  return SUCCESS;
}

ge::Status SpecialNodeChecker::CheckBeforeAssign(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodesPtr()) {
    GE_ASSERT_SUCCESS(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(node), "attrs check failed. graph: %s.",
                      graph->GetName().c_str());
    GE_ASSERT_SUCCESS(NodeCheckerUtils::CheckNoPaddingContinuousOutputNodeAttrs(node), "attrs check failed. graph: %s.",
                      graph->GetName().c_str());
    GE_ASSERT_SUCCESS(NodeCheckerUtils::CheckPhonySplitInputSize(node),
                      "NoPaddingContinuousOutputNode check failed. graph: %s.", graph->GetName().c_str());
    GE_ASSERT_SUCCESS(NodeCheckerUtils::CheckPhonyConcatOutputSize(node),
                      "NoPaddingContinuousInputNode check failed. graph: %s.", graph->GetName().c_str());
  }
  GELOGI("check attrs success, graph: %s", graph->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge