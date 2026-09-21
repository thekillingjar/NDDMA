/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_COMPILER_MODEL_DFLOW_GRAPH_MANAGER_H_
#define DFLOW_COMPILER_MODEL_DFLOW_GRAPH_MANAGER_H_

#include <map>
#include <string>
#include <vector>
#include <mutex>
#include "dflow/inc/data_flow/model/flow_model.h"
#include "dflow/compiler/model/flow_model_builder.h"
#include "flow_graph/data_flow.h"
// external
#include "framework/common/ge_types.h"
#include "graph/preprocess/graph_prepare.h"

namespace ge {
using ge::dflow::FlowGraph;

/**
@brief DflowGraphManager - Dflow-specific graph manager
Acts as the interface layer between dflow_session_impl and the underlying graph_manager
Takes over the management of graphs, while the actual compilation work is handled by graph_manager
*/
class DflowGraphManager {
 public:
  DflowGraphManager() = default;
  ~DflowGraphManager();

  /// @brief data flow graph manager init
  /// @param [in] options user config params
  /// @param [in] pneImpl default graph compiler
  /// @return Status result of function
  Status Initialize(const std::map<std::string, std::string> &options,
                    const std::shared_ptr<ProcessNodeEngineImpl> &pneImpl = nullptr);

  /// @brief data flow graph manager finalize
  void Finalize();

  /// @brief add specific flow graph
  /// @param [in] graph_id graph id
  /// @param [in] graph ge graph
  /// @param [in] options user config params
  /// @return Status result of function
  Status AddGraph(uint32_t graph_id, const Graph &graph,
                  const std::map<std::string, std::string> &options);

  /// @brief remove specific graph
  /// @param [in] graph_id graph id
  /// @return Status result of function
  Status RemoveGraph(uint32_t graph_id);

  /// @brief compile specific graph
  /// @param [in] graph_id graph id
  /// @param [in] inputs inputs for graph
  /// @return Status result of function
  Status CompileGraph(uint32_t graph_id, const std::vector<GeTensor> &inputs);

  /// @brief get all options define for specific graph id
  /// @param [in] graph_id graph id
  /// @return all options pointer
  const std::map<std::string, std::string> *GetGraphOptions(uint32_t graph_id);

  /// @brief get all model id for specific graph id
  /// @param [in] graph_id graph id
  /// @param [out] model_id model id got by graph id
  Status GetGraphModelId(uint32_t graph_id, uint32_t &model_id);

  /// @brief get flow model pointer by specific graph id
  /// @param [in] graph_id graph id
  /// @return flow model pointer
  FlowModelPtr GetFlowModel(uint32_t graph_id) const;

  /// @brief get flag of whether need load and run model or not
  /// @return True or False
  bool GetOptionsRunGraphFlag();

 private:
  bool is_initialized_{false};
  std::map<std::string, std::string> options_;

  mutable std::mutex mutex_;
  mutable std::mutex graph_mutex_;
  mutable std::mutex model_mutex_;
  
  // dflow专用的图管理数据结构
  std::map<uint32_t, std::map<std::string, std::string>> graph_options_map_;
  std::map<uint32_t, FlowModelPtr> flow_model_map_;
  std::map<uint32_t, Graph> flow_graph_map_;
  std::map<uint32_t, OmgContext> omg_contexts_;
  FlowModelBuilder flow_model_builder_;
};

}  // namespace ge

#endif  // DFLOW_COMPILER_MODEL_DFLOW_GRAPH_MANAGER_H_ 