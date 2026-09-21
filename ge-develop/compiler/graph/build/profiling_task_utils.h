/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_BUILD_PROFILING_TASK_UTILS_H_
#define AIR_CXX_COMPILER_GRAPH_BUILD_PROFILING_TASK_UTILS_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/types.h"
#include "common/profiling/profiling_properties.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "common/math/math_util.h"
#include "proto/task.pb.h"

namespace ge {
struct ProfilingPoint {
  int64_t fp_index = 0;
  int64_t bp_index = 0;
  std::set<int64_t> end_index;
  std::vector<int64_t> all_reduce_node_index;
  std::vector<int64_t> get_next_node_index;
};
class ProfilingTaskUtils {
 public:
  explicit ProfilingTaskUtils(std::vector<Node *> nodes) : nodes_(std::move(nodes)) {}
  Status FindProfilingTaskIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point) const;
  static Status InsertProfilingTaskBefore(const OpDescPtr &op_desc, const ProfilingPoint &profiling_point,
                                          std::vector<domi::TaskDef> &task_def_list);

  static Status InsertProfilingTaskAfter(const OpDescPtr &op_desc, const ProfilingPoint &profiling_point,
                                         std::vector<domi::TaskDef> &task_def_list);

 private:
  // profiling interface
  static void AssembleTaskForProfilerTrace(const int64_t stream_id, const int64_t log_id,
                                           bool is_iteration_end,
                                           std::vector<domi::TaskDef> &task_def_list);
  static Status CalculateLogId(const std::vector<int64_t> &trace_nodes,
                               const int64_t node_index, const uint64_t start_id, int64_t &log_id);
  static Status InsertProfilingArTaskBefore(const OpDescPtr &op_desc, const std::vector<int64_t> &all_reduce_nodes,
                                            int64_t node_index, std::vector<domi::TaskDef> &task_def_list,
                                            bool is_insert_bp_profiling_task);
  static Status InsertProfilingArTaskAfter(const OpDescPtr &op_desc, const std::vector<int64_t> &all_reduce_nodes,
                                           int64_t node_index, std::vector<domi::TaskDef> &task_def_list,
                                           bool is_insert_bp_profiling_task);
  Status FindGetNextNodes(std::vector<int64_t> &get_next_nodes) const;
  Status FindFpOpCrossSubgraph(const NodePtr &node, int32_t index, OpDescPtr &fp_op_desc) const;
  Status AutoFindFpOpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point) const;
  static Status FindLastBpCrossSubgraph(const NodePtr &in_node, OpDescPtr &bp_op_desc);
  Status AutoFindBpOpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                           std::vector<int64_t> &all_reduce_nodes) const;
  static Status FindLastBpFromBpNode(const ComputeGraphPtr &graph, const Node *target_node, int64_t &bp_index);

  Status FindFpOfEnv(const ComputeGraphPtr &graph, const std::string &fp_point_str,
                     ProfilingPoint &profiling_point) const;
  Status FindBpOfEnv(const ComputeGraphPtr &graph, const std::string &bp_point_str, ProfilingPoint &profiling_point,
                     std::vector<int64_t> &all_reduce_nodes) const;

  void GetFpBpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                    std::vector<int64_t> &all_reduce_nodes, std::string &fp_point_str, std::string &bp_point_str) const;

  static bool IsProfPoint(const OpDescPtr &op, const std::string &name);
  static bool IsGlobalStep(const NodePtr &node);
  std::vector<Node *> nodes_;
};
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_BUILD_PROFILING_TASK_UTILS_H_
