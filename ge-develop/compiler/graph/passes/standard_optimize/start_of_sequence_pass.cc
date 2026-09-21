/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/start_of_sequence_pass.h"

#include "common/plugin/ge_make_unique_util.h"
#include "common/profiling/profiling_properties.h"

namespace ge {
Status StartOfSequencePass::Run(ComputeGraphPtr graph) {
  if (graph->GetParentGraph() != nullptr) {
    return SUCCESS;
  }
  const bool is_profiling =
      ProfilingProperties::Instance().ProfilingOn() || ProfilingProperties::Instance().ProfilingTrainingTraceOn();
  if (!is_profiling) {
    GELOGD("Profiling is not open.");
    return SUCCESS;
  }

  GELOGI("StartOfSequencePass begin.");
  std::vector<NodePtr> target_nodes;
  for (const NodePtr &node : graph->GetDirectNode()) {
    if (node->GetInNodes().empty()) {
      target_nodes.push_back(node);
    }
  }

  const OpDescPtr op_desc = MakeShared<OpDesc>("StartOfSequence", STARTOFSEQUENCE);
  GE_CHECK_NOTNULL(op_desc);

  const NodePtr start_of_sequence = graph->AddNode(op_desc);
  GE_CHECK_NOTNULL(start_of_sequence);

  // Insert control edge
  for (const auto &node : target_nodes) {
    GELOGI("Add ctrl edge between %s and %s", start_of_sequence->GetName().c_str(), node->GetName().c_str());
    if (GraphUtils::AddEdge(start_of_sequence->GetOutControlAnchor(), node->GetInControlAnchor()) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add ctrl edge between %s and %s failed", start_of_sequence->GetName().c_str(),
                        node->GetName().c_str());
      GELOGE(FAILED, "[Add][CtrlEdge] between %s and %s failed", start_of_sequence->GetName().c_str(),
             node->GetName().c_str());
      return FAILED;
    }
  }

  GELOGI("StartOfSequencePass end.");
  return SUCCESS;
}

REG_PASS_OPTION("StartOfSequencePass").LEVELS(OoLevel::kO0);
}  // namespace ge
