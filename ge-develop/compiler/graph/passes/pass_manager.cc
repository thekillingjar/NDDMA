/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/pass_manager.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/ge_context.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/node_utils.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "framework/omg/omg_inner_types.h"
#include "common/util/trace_manager/trace_manager.h"

namespace ge {
const std::vector<std::pair<std::string, GraphPass *>>& PassManager::GraphPasses() const {
  return names_to_graph_passes_;
}

Status PassManager::AddPass(const std::string &pass_name, GraphPass *pass) {
  GE_CHECK_NOTNULL(pass);
  names_to_graph_passes_.emplace_back(pass_name, pass);
  return SUCCESS;
}

Status PassManager::Run(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  return Run(graph, names_to_graph_passes_);
}

Status PassManager::Run(const ComputeGraphPtr &graph,
                        std::vector<std::pair<std::string, GraphPass *>> &names_to_passes) {
  GE_CHECK_NOTNULL(graph);
  const auto &disabled_optimizations = PassUtils::GetDisabledOptimizations();
  for (auto &pass_pair : names_to_passes) {
    const auto &pass_name = pass_pair.first;
    if (PassUtils::IsOptimizationDisabled(disabled_optimizations, pass_name)) {
      GELOGI("Pass [%s] is disabled, skip it", pass_name.c_str());
      continue;
    }
    const auto &pass = pass_pair.second;
    GE_CHECK_NOTNULL(pass);
    TraceOwnerGuard guard("GE", pass_name, graph->GetName());
    GE_TRACE_START(PassRun);
    Status status = pass->Run(graph);
    if ((status != SUCCESS) && (status != NOT_CHANGED)) {
      GELOGE(status, "[Pass][Run] failed on graph %s", graph->GetName().c_str());
      return status;
    }
    for (const auto &subgraph :graph->GetAllSubgraphs()) {
      GE_CHECK_NOTNULL(subgraph);
      GE_CHK_STATUS_RET(pass->ClearStatus(), "[Pass][ClearStatus] failed for subgraph %s", subgraph->GetName().c_str());
      std::string subgraph_pass_name = pass_name + "::" + graph->GetName();
      GE_TRACE_START(PassRunSubgraph);
      TraceOwnerGuard sub_guard("GE_SUB", subgraph_pass_name, graph->GetName());
      status = pass->Run(subgraph);
      GE_COMPILE_TRACE_TIMESTAMP_END(PassRunSubgraph, subgraph_pass_name.c_str());
      if ((status != SUCCESS) && (status != NOT_CHANGED)) {
        GELOGE(status, "[Pass][Run] failed on subgraph %s", subgraph->GetName().c_str());
        return status;
      }
    }
    GE_COMPILE_TRACE_TIMESTAMP_END(PassRun, pass_name.c_str());
  }

  return SUCCESS;
}

PassManager::~PassManager() {
  for (auto &pass_pair : names_to_graph_passes_) {
    auto &pass = pass_pair.second;
    GE_DELETE_NEW_SINGLE(pass);
  }
}
}  // namespace ge
