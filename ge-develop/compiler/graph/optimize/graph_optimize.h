/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZE_H_
#define GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZE_H_

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "common/optimizer/graph_optimizer.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_context.h"
#include "graph/manager/graph_manager_utils.h"
#include "framework/omg/omg_inner_types.h"

namespace ge {
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;
class GraphOptimize {
 public:
  GraphOptimize();
  ~GraphOptimize() = default;

  Status OptimizeGraphInit(const ComputeGraphPtr &compute_graph) const;

  Status FinalizeSessionInfo(const ComputeGraphPtr &compute_graph) const;

  // subgraph optimize
  Status OptimizeSubGraph(ComputeGraphPtr &compute_graph, const std::string &engine_name) const;

  // original graph optimize
  Status OptimizeOriginalGraph(const ComputeGraphPtr &compute_graph) const;

  Status OptimizeOriginalGraphJudgePrecisionInsert(const ComputeGraphPtr &compute_graph) const;

  Status OptimizeOriginalGraphJudgeFormatInsert(const ComputeGraphPtr &compute_graph) const;

  Status OptimizeAfterGraphNormalization(const ComputeGraphPtr& compute_graph) const;

  // for fe prepare optimize in quantize scene
  Status OptimizeOriginalGraphForQuantize(const ComputeGraphPtr &compute_graph) const;

  // for engine to optimize merged whole graph before ge Optimize2
  Status OptimizeWholeGraph(const ComputeGraphPtr &compute_graph) const;

  // for optimize before build
  Status OptimizeGraphBeforeBuild(const ComputeGraphPtr &compute_graph) const;

  // optimize whole graph, using after stage1
  Status OptimizeAfterStage1(const ComputeGraphPtr &compute_graph) const;

  // set options
  Status SetOptions(const GraphManagerOptions &options);

  const std::map<uint32_t, std::map<std::string, size_t>> &GetSummaryOutputIndexes() const {
    return summary_output_indexes_;
  }  // lint !e1073

  // handle summary node before preRun graph
  Status HandleSummaryOp(const ComputeGraphPtr &compute_graph);

  // Identify reference node before optimize subgraph
  Status IdentifyReference(const ComputeGraphPtr &compute_graph) const;

  Status HandleMemoryRWConflict(ComputeGraphPtr &compute_graph) const;

  Status HandleMemoryLayoutConflict(ComputeGraphPtr &compute_graph) const;

  Status CheckRWConflict(ComputeGraphPtr &compute_graph, bool &has_conflict) const;

  void TranFrameOp(const ComputeGraphPtr &compute_graph) const;

  bool IsExclude(const std::string &engine_name) const;

  Status OptimizeSubgraphPreProc(ComputeGraph &graph) const;

  Status OptimizeSubgraphPostProc(ComputeGraph &graph) const;

 private:
  Status GetValidGraphOptimizer(std::vector<std::pair<std::string, GraphOptimizerPtr>> &valid_optimizer) const;
  Status OptimizeSubgraphProc(ComputeGraph &graph, const bool is_pre) const;
  std::mutex mutex_;
  domi::FrameworkType optimize_type_;
  std::string cal_config_;
  std::string insert_op_config_;
  std::string core_type_;
  bool local_fmk_op_flag_ = false;
  // record the summary names for filter sumarry result.
  std::map<uint32_t, std::map<std::string, size_t>> summary_output_indexes_ = {};
  std::string func_bin_path_;
  std::string build_mode_;
  std::string build_step_;
  std::set<std::string> exclude_engines_;
};
}  // namespace ge
#endif  // GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZE_H_
