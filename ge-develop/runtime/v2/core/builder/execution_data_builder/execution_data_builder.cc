/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "execution_data_builder.h"
#include "graph/compute_graph.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "register/kernel_registry.h"
#include "runtime/exe_graph_executor.h"

namespace gert {
bool ExecutionDataBuilder::HasOutputsNeedCreate(const ge::FastNode &node) {
  const auto op_desc = node.GetOpDescBarePtr();
  for (uint32_t i = 0U; i < node.GetDataOutNum(); ++i) {
    if (!ge::AttrUtils::HasAttr(op_desc->GetOutputDesc(i), kRefFromNode)) {
      return true;
    }
  }
  return false;
}

ge::graphStatus ExecutionDataBuilder::CreateKernelOutputs(
    std::vector<std::pair<ge::FastNode *, Node *>> &graph_nodes_to_executor_node) const {
  for (const auto &graph_node_to_exe_node : graph_nodes_to_executor_node) {
    auto org_node = graph_node_to_exe_node.first;
    auto exe_node = graph_node_to_exe_node.second;

    if (!HasOutputsNeedCreate(*org_node)) {
      GELOGD("The node %s type %s does not need to create outputs", org_node->GetNamePtr(), org_node->GetTypePtr());
      continue;
    }

    auto kernel_funcs = KernelRegistry::GetInstance().FindKernelFuncs(org_node->GetType());
    if (kernel_funcs == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "Failed to find exec node [%s] kernel funcs", org_node->GetType().c_str());
      return ge::GRAPH_FAILED;
    }

    if (kernel_funcs->outputs_creator != nullptr) {
      if (kernel_funcs->outputs_creator(org_node, reinterpret_cast<KernelContext *>(&exe_node->context)) !=
          ge::GRAPH_SUCCESS) {
        GELOGE(ge::FAILED, "Failed to create output for node %s type %s", org_node->GetName().c_str(),
               org_node->GetType().c_str());
        return ge::FAILED;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace gert
