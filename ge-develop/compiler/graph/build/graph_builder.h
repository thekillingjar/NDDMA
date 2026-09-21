/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_GRAPH_BUILD_H_
#define GE_GRAPH_BUILD_GRAPH_BUILD_H_
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "framework/common/debug/log.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/build/model_builder.h"
#include "graph/build/task_generator.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/partition/engine_partitioner.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/model/ge_root_model.h"

namespace ge {
class GraphBuilder {
 public:
  GraphBuilder();
  GraphBuilder(const GraphBuilder &in) = delete;
  GraphBuilder &operator=(const GraphBuilder &in) = delete;
  virtual ~GraphBuilder() = default;
  Status Build(ComputeGraphPtr &comp_graph, GeRootModelPtr &ge_root_model_ptr,
               uint64_t session_id = INVALID_SESSION_ID);
  void SetOptions(const GraphManagerOptions &options);
  Status BuildForEvaluate(ComputeGraphPtr &compute_graph, ModelDataInfo &model);

 private:
  Status CalcOpParam(const ge::ComputeGraphPtr &graph);
  Status ProcessAppendWs(const ModelPtr &model_ptr, const ComputeGraphPtr &comp_graph,
                         std::vector<Node *> &refresh_nodes) const;
  Status UpdateMemAfterAppendWs(const ModelPtr &model_ptr,
                                const std::map<int64_t, int64_t> &append_ws_stm_max,
                                const std::map<int64_t, std::vector<NodePtr>> &append_ws_stm_nodes,
                                int64_t &last_append_ws_size) const;
  Status RefreshOffsetAfterAppendWs(const ComputeGraphPtr &comp_graph,
                                    const int64_t origin_memory_size, const int64_t zero_copy_size,
                                    const int64_t last_append_ws_size, std::vector<Node *> &refresh_nodes) const;
  Status RefreshNodeOffsetAfterAppendWs(const Node *node,
                                        const int64_t origin_memory_size, const int64_t zero_copy_size,
                                        const int64_t last_append_ws_size, bool &need_refresh) const;
  Status GetCurrentRunContext(const ge::ModelBuilder &builder, const ModelPtr &model_ptr,
                              ComputeGraphPtr &comp_graph, uint64_t session_id);
  Status GetTaskInfo(const ge::ModelBuilder &builder, const ModelPtr &model_ptr, ComputeGraphPtr &comp_graph,
                     Graph2SubGraphInfoList &subgraph_map, uint64_t session_id = INVALID_SESSION_ID);
  Status SetInputSize(const ge::NodePtr &node_ptr);
  Status UpdateDataInputSize(const ge::NodePtr &node_ptr) const;
  Status UpdateParentNodeOutputSize(const ge::ComputeGraphPtr &graph, const ge::NodePtr &parent_node_ptr) const;
  Status CalcDynShapeRootGraphDataSize(const ge::OpDescPtr &op_desc) const;
  Status SecondPartition(const ge::ComputeGraphPtr &comp_graph);
  Status MarkFpBpProfilingTaskAttr(ComputeGraphPtr &com_graph) const;
  Status BuildForDynamicShapeGraph(ComputeGraphPtr &comp_graph, GeRootModelPtr &ge_root_model_ptr,
                                   GeModelPtr &ge_model_ptr, const uint64_t session_id = INVALID_SESSION_ID,
                                   const bool has_assigned_var_mem = false);
  Status AssignAttachedResourceForAllDynamicGraph(const ComputeGraphPtr &comp_graph,
                                                  GeRootModelPtr &ge_root_model) const;
  Status BuildForUnknownShapeAllGraphs(ComputeGraphPtr &comp_graph, GeRootModelPtr &ge_root_model_ptr,
                                       GeModelPtr &ge_model_ptr, const uint64_t session_id = INVALID_SESSION_ID,
                                       const bool has_assigned_var_mem = false);
  Status RefreshInfoOfDynamicShapeGraph(ComputeGraphPtr &comp_graph, GeRootModelPtr &ge_root_model) const;
  Status BuildForKnownShapeGraph(ComputeGraphPtr &comp_graph,
                                 GeModelPtr &ge_model_ptr, const uint64_t session_id = INVALID_SESSION_ID,
                                 const bool has_assigned_var_mem = false);
  Status BuildForUnknownShapeGraph(ComputeGraphPtr &comp_graph, GeModelPtr &ge_model_ptr,
                                   uint64_t session_id = INVALID_SESSION_ID);
  Status SetConstantInputOffset(const ComputeGraphPtr &comp_graph) const;
  Status AddOutputMemTypeForNode(const NodePtr &node) const;
  Status CalcLogIdAndSetAttr(const OpDescPtr &op_desc, const std::vector<int64_t> &trace_nodes,
                             const int64_t node_index, const int64_t start_id) const;
  static bool IsDataDirectConnNetoutput(ComputeGraphPtr &comp_graph);
  Status ReGetTaskInfo(const ComputeGraphPtr &comp_graph, uint64_t session_id, Model &model);

  int32_t build_mode_;

  std::map<std::string, int32_t> stream_max_parallel_num_;
  bool hcom_parallel_;

  std::map<ComputeGraphPtr, RunContext> graph_2_run_context_;
  std::map<ComputeGraphPtr, std::unique_ptr<TaskGenerator>> graph_2_task_generator_;

  EnginePartitioner graph_partitioner_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_GRAPH_BUILD_H_
