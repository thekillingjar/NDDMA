/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/model_data_info.h"

#include "graph/def_types.h"
#include "common/debug/log.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/math/math_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/build/graph_builder.h"
#include "graph/manager/session_id_manager.h"
#include "graph/ge_context.h"

namespace ge {
graphStatus EvaluateGraphResource(const std::map<std::string, std::string> &options,
    ge::ComputeGraphPtr &compute_graph, ModelDataInfo &model) {
  GE_CHECK_NOTNULL(compute_graph);
  GELOGI("Evaluate graph resource name:%s.", compute_graph->GetName().c_str());
  // add new options
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  for (const auto &option : options) {
    graph_options[option.first] = option.second;
    GELOGI("Option name:%s value:%s.", option.first.c_str(), option.second.c_str());
  }
  graph_options[EVALUATE_GRAPH_RESOURCE_MODE] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);

  GE_MAKE_GUARD(graph_options, ([&graph_options, &options] {
    // restore origion options
    for (const auto &option : options) {
      (void) graph_options.erase(option.first);
    }
    (void) graph_options.erase(EVALUATE_GRAPH_RESOURCE_MODE);
    GetThreadLocalContext().SetGraphOption(graph_options);
  }));

  uint64_t session_id = SessionIdManager::GetNextSessionId();
  uint32_t graph_id = 1U;
  compute_graph->SetSessionID(session_id);
  compute_graph->SetGraphID(graph_id);
  const auto var_manager = VarManager::Instance(session_id);
  GE_CHECK_NOTNULL(var_manager);
  auto ret = var_manager->Init(0U, session_id, 0U, 0U);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed init var instance, session_id %lu", session_id);
    return FAILED;
  }

  std::string session_graph_id = std::to_string(session_id) + "_" + std::to_string(graph_id);
  if (!AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    return FAILED;
  }

  GraphBuilder graph_builder;
  return graph_builder.BuildForEvaluate(compute_graph, model);
}

Status GetGraphAvailableMemory(const ComputeGraphPtr &graph, uint64_t &available_mem) {
  auto var_manager = VarManager::Instance(graph->GetSessionID());
  GE_CHECK_NOTNULL(var_manager);
  uint64_t new_var_total_size = 0U;
  for (const NodePtr &node : graph->GetDirectNode()) {
    const auto node_type = node->GetType();
    if ((node_type != VARIABLE) && (node_type != CONSTANTOP) && (node_type != FILECONSTANT) &&
        (node_type != CONSTANT)) {
      continue;
    }
    if (var_manager->IsVarExist(node->GetName())) {
      continue;
    }
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (const auto &output_desc : op_desc->GetAllOutputsDesc()) {
      int64_t tensor_size = 0L;
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorMemorySizeInBytes(output_desc, tensor_size),
                              "[Get][TensorMemorySize] Failed to calc tensor size");
      GE_CHK_STATUS_RET(CheckUint64AddOverflow(new_var_total_size, static_cast<uint64_t>(tensor_size)),
                        "[Overflow] total_size %lu add tensor_size %ld overflow", new_var_total_size, tensor_size);
      GELOGD("In graph_%s, var:%s size:%ld", graph->GetName().c_str(), node->GetName().c_str(), tensor_size);
      new_var_total_size += static_cast<uint64_t>(tensor_size);
    }
  }
  const uint64_t max_var_size = var_manager->GetVarMemMaxSize();
  const uint64_t current_var_size = static_cast<uint64_t>(var_manager->GetVarMemSize(RT_MEMORY_HBM));
  const uint64_t var_size = current_var_size >= max_var_size ? current_var_size : max_var_size;

  available_mem = var_manager->GetUseMaxMemorySize() - var_size - new_var_total_size;
  GELOGI("max memory size:%lu, max_var_size:%lu, current_var_size:%lu, new_var_total_size:%lu, free_graph_size:%lu",
         var_manager->GetUseMaxMemorySize(), max_var_size, current_var_size, new_var_total_size, available_mem);
  return SUCCESS;
}
};      // namespace ge
