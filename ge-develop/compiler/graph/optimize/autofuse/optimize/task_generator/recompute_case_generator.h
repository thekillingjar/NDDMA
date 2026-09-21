/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_RECOMPUTE_CASE_GENERATOR_H_
#define ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_RECOMPUTE_CASE_GENERATOR_H_

#include "ascir.h"
#include "ascgen_log.h"
#include "schedule_task_generator.h"
#include "task_generator/schedule_case_generator.h"

namespace optimize {
class RecomputeCaseGenerator : public FusionCaseGenerator {
 public:
  Status Generate( [[maybe_unused]]ascir::HintGraph &graph, [[maybe_unused]]std::vector<ascir::ImplGraph> &graphs,
                   [[maybe_unused]]std::vector<std::string> &score_functions) override {
   return ge::GRAPH_SUCCESS;
  }
  Status GeneratorTask(ascir::HintGraph &hint_graph, std::vector<ScheduleTask> &tasks,
                       const OptimizerOptions &options) override;

 private:
  struct SplitResultInOnePath {
    std::unordered_set<ge::Node *> recompute_nodes;  // 需要重计算的节点
    std::unordered_set<ge::Node *> valid_path;       // 有效路径
    std::unordered_set<ge::Node *> output_nodes;     // output节点
  };

  struct SplitResult {
    std::vector<SplitResultInOnePath> valid_path_results;   // 有效路径
    std::unordered_set<ge::Node *> merged_recompute_nodes;  // 需要重计算的节点
    std::unordered_set<ge::Node *> merged_output_nodes;     // output节点
    std::unordered_set<ge::Node *> merged_path_nodes;       // 所有在拆分路径上的节点
  };

  bool IsRecomputableNode(ascir::HintGraph &hint_graph, ge::AscNode *node) const;
  bool IsRecomputableAlwaysBetter() const;
  Status AnalyzeSplittablePath(ascir::HintGraph &hint_graph, const ge::AscNodePtr &potential_store);
  void MergeAllPaths();
  Status DoTaskGenerator(ascir::ImplGraph &impl_graph, std::vector<ScheduleTask> &tasks) const;
  Status DoGraphSplit(ge::AscGraph &graph) const;

  bool is_static_graph_{false};
  SplitResult result_;
};
}  // namespace optimize

#endif  // ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_RECOMPUTE_CASE_GENERATOR_H_
