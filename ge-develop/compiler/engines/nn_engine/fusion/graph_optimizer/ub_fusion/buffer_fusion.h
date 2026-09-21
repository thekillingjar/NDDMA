/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BUFFER_FUSION_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BUFFER_FUSION_H_

#include <memory>
#include <utility>

#include "common/fe_inner_attr_define.h"
#include "common/graph_comm.h"
#include "graph/compute_graph.h"
#include "fusion_config_manager/fusion_priority_manager.h"
#include "fusion_rule_manager/fusion_cycle_detector.h"
#include "graph_optimizer/ub_fusion/fusion_graph_merge/fusion_graph_merge.h"
#include "graph_optimizer/ub_fusion/fusion_graph_merge/ub_fusion_graph_merge.h"
#include "graph_optimizer/ub_fusion/fusion_graph_merge/l1_fusion_graph_merge.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph_optimizer/ub_fusion/buffer_fusion_pass_runner.h"

namespace fe {
/** @brief ub fusion: find subgraphs that match fusion patterns from graph firstly,
*        and fusion ops into one, change graph topology structure correspondingly. */
class BufferFusion {
  using FusionGraphMergeUniquePtr = std::unique_ptr<FusionGraphMerge>;

 public:
  BufferFusion(GraphCommPtr graph_comm_ptr, FusionPriorityMgrPtr fusion_priority_mgr_ptr,
               OpStoreAdapterBasePtr op_store_adapter_ptr, FusionCycleDetectorPtr cycle_detector = nullptr)
      : fusion_priority_mgr_ptr_(std::move(fusion_priority_mgr_ptr)),
        op_store_adapter_ptr_(std::move(op_store_adapter_ptr)),
        cycle_detector_(std::move(cycle_detector)) {
    ub_fusion_graph_merge_ptr_ =
        std::unique_ptr<FusionGraphMerge>(new (std::nothrow) UBFusionGraphMerge(SCOPE_ID_ATTR, graph_comm_ptr));
    l1_fusion_graph_merge_ptr_ =
        std::unique_ptr<FusionGraphMerge>(new (std::nothrow) L1FusionGraphMerge(L1_SCOPE_ID_ATTR, graph_comm_ptr));
  }

  ~BufferFusion() {}

  /*
   * @brief: match defined fusion pattern from graph and assign scope id to fusion op
   */
  Status MatchFusionPatternFromGraph(ge::ComputeGraph &graph);

  /*
   * @brief: create fusion graph with scope_id create by MatchFusionPatternFromGraph,
   *        i.e. nodes have same scope_id will be fused into one fusion op,
   *        the topo of graph will be changed.
   */
  Status BuildFusionGraph(ge::ComputeGraph &graph);

  void SetEngineName(const std::string &engine_name) { engine_name_ = engine_name; }
  Status MatchFusionPattern(ge::ComputeGraph &graph);

 private:
  Status RunBuiltInFusion(ge::ComputeGraph &graph);
  Status RunRegisterBufferFusionPass(ge::ComputeGraph &graph, BufferFusionPassType pass_type);
  BaseBufferFusionPassRunnerPtr MakePassRunnerPtr(const BufferFusionInfo &sorted_buffer_fusion_info,
                                                  const BufferFusionOptimizerPtr buffer_fusion_optimizer) const;
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_;

  FusionGraphMergeUniquePtr ub_fusion_graph_merge_ptr_;
  FusionGraphMergeUniquePtr l1_fusion_graph_merge_ptr_;
  OpStoreAdapterBasePtr op_store_adapter_ptr_;
  std::shared_ptr<fe::FusionCycleDetector> cycle_detector_;
  std::string engine_name_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BUFFER_FUSION_H_
