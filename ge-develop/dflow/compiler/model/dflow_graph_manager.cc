/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/model/dflow_graph_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/manager/graph_manager.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/graph_utils_ex.h"
#include "opt_info/ge_opt_info.h"
#include "graph/ge_context.h"
#include "common/context/local_context.h"
#include "common/compile_profiling/ge_call_wrapper.h"


namespace ge {
DflowGraphManager::~DflowGraphManager() {
  // set thread local omg_context to default value, to avoid other use invalid memory
  SetLocalOmgContext(domi::GetContext());
}

Status DflowGraphManager::Initialize(const std::map<std::string, std::string> &options,
                                     const std::shared_ptr<ProcessNodeEngineImpl> &pneImpl) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (is_initialized_) {
    GELOGW("[DflowGraphManager] Already initialized.");
    return SUCCESS;
  }
  options_ = options;
  GE_CHK_STATUS_RET(flow_model_builder_.InitProcessNodeEngines(options, pneImpl),
                    "Flow model builder init process node engines failed.");
  is_initialized_ = true;
  GELOGI("[DflowGraphManager] Initialize success.");
  return SUCCESS;
}

void DflowGraphManager::Finalize() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!is_initialized_) {
    GELOGW("[DflowGraphManager] Not initialized.");
    return;
  }

  graph_options_map_.clear();
  is_initialized_ = false;
  
  GELOGI("[DflowGraphManager] Finalize success.");
}

Status DflowGraphManager::AddGraph(uint32_t graph_id, const Graph &graph, 
                                   const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!is_initialized_) {
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Add][Graph] DflowGraphManager not initialized.");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  {
    std::lock_guard<std::mutex> graph_lock(graph_mutex_);
    if (flow_graph_map_.find(graph_id) != flow_graph_map_.end()) {
      GELOGE(FAILED, "[Add][Graph] Graph already added, graph_id=%u.", graph_id);
      return FAILED;
    } else {
      graph_options_map_.emplace(graph_id, options);
      flow_graph_map_.emplace(graph_id, graph);
      omg_contexts_.emplace(graph_id, domi::GetContext());
      SetLocalOmgContext(omg_contexts_[graph_id]);
    }
  }
  GELOGI("[DflowGraphManager] Add graph success, graph_id=%u.", graph_id);
  return SUCCESS;
}

Status DflowGraphManager::RemoveGraph(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!is_initialized_) {
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Remove][Graph] DflowGraphManager not initialized.");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  Status ret = SUCCESS;
  {
    std::lock_guard<std::mutex> graph_lock(graph_mutex_);
    if ((flow_graph_map_.find(graph_id) == flow_graph_map_.end()) ||
        (omg_contexts_.find(graph_id) == omg_contexts_.end()) ||
        (graph_options_map_.find(graph_id) == graph_options_map_.end())) {
      GELOGE(FAILED, "[Remove][Graph] Graph id [%u] not found. AddGraph should be called before RemoveGraph.",
             graph_id);
      ret = FAILED;
    }
    (void)flow_graph_map_.erase(graph_id);
    (void)omg_contexts_.erase(graph_id);
    (void)graph_options_map_.erase(graph_id);
  }
  {
    std::lock_guard<std::mutex> flow_model_lock(model_mutex_);
    (void)flow_model_map_.erase(graph_id);
  }
  GELOGI("[DflowGraphManager] Remove graph finished, graph_id=%u.", graph_id);
  return ret;
}

Status DflowGraphManager::CompileGraph(uint32_t graph_id, const std::vector<GeTensor> &inputs) {
  if (!is_initialized_) {
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Build][Graph] DflowGraphManager not initialized.");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  GELOGD("Start compile graph in dflow graph manager, graph_id=%u.");
  {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (flow_model_map_.find(graph_id) != flow_model_map_.end()) {
      GEEVENT("Graph id %u has already been compiled.", graph_id);
      return SUCCESS;
    }
  }
  Graph graph;
  std::map<std::string, std::string> graph_options;
  {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    const auto iter = graph_options_map_.find(graph_id);
    const auto graph_iter = flow_graph_map_.find(graph_id);
    if ((graph_iter == flow_graph_map_.end()) || (iter == graph_options_map_.end())) {
      GELOGE(FAILED, "Graph id[%u] can not be found in graph map. AddGraph should be called before CompileGraph.",
            graph_id);
      return GE_GRAPH_GRAPH_NOT_EXIST;
    }
    graph = graph_iter->second;
    graph_options = iter->second;
  }

  GE_CHK_STATUS_RET(GeOptInfo::SetOptInfo(), "Set opt info failed.");
  FlowModelPtr flow_model = nullptr;
  GE_TIMESTAMP_START(BuildModel);
  GE_CHK_STATUS_RET(flow_model_builder_.BuildModel(graph, inputs, graph_options, flow_model),
                    "Build graph failed, graph_id=%u.", graph_id);
  GE_TIMESTAMP_EVENT_END(BuildModel, "FlowModelBuild");
  GE_CHECK_NOTNULL(flow_model);
  {
    std::lock_guard<std::mutex> lock(model_mutex_);
    flow_model_map_.emplace(graph_id, flow_model);
  }
  GELOGI("[DflowGraphManager] Compile graph success, graph_id=%u.", graph_id);
  return SUCCESS;
}

const std::map<std::string, std::string> *DflowGraphManager::GetGraphOptions(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(graph_mutex_);
  auto it = graph_options_map_.find(graph_id);
  if (it != graph_options_map_.end()) {
    return &(it->second);
  }
  return nullptr;
}

Status DflowGraphManager::GetGraphModelId(uint32_t graph_id, uint32_t &model_id) {
  model_id = INVALID_MODEL_ID; 
  if (!is_initialized_) {
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Get][GraphModelId] DflowGraphManager not initialized.");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  std::lock_guard<std::mutex> lock(model_mutex_);
  const auto iter = flow_model_map_.find(graph_id);
  if (iter != flow_model_map_.end()) {
    const auto flow_model = iter->second;
    GE_CHECK_NOTNULL(flow_model);
    model_id = flow_model->GetModelId();
  }
  return SUCCESS;
}

FlowModelPtr DflowGraphManager::GetFlowModel(uint32_t graph_id) const {
  if (!is_initialized_) {
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Get][FlowModel] DflowGraphManager not initialized.");
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(model_mutex_);
  const auto iter = flow_model_map_.find(graph_id);
  if (iter != flow_model_map_.end()) {
    return iter->second;
  }
  return nullptr;
}

bool DflowGraphManager::GetOptionsRunGraphFlag() {
  if (!is_initialized_) {
    GELOGW("[Get][OptionsRunGraphFlag] DflowGraphManager not initialized.");
    return false;
  } 
  const auto iter = options_.find(RUN_FLAG);
  if ((iter != options_.end()) && (iter->second == "0")) {
    return false;
  }
  return true;
}
}  // namespace ge 