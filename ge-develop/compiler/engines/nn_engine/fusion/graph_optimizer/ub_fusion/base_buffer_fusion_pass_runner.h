/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BASE_BUFFER_FUSION_PASS_RUNNER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BASE_BUFFER_FUSION_PASS_RUNNER_H_

#include <memory>
#include "common/aicore_util_types.h"
#include "common/op_store_adapter_base.h"
#include "common/fe_inner_attr_define.h"
#include "common/math_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "fusion_rule_manager/fusion_cycle_detector.h"

namespace fe {
using BufferFusionPassBasePtr = std::unique_ptr<BufferFusionPassBase>;

class BaseBufferFusionPassRunner {
 public:
  BaseBufferFusionPassRunner(const std::string &pass_name, BufferFusionPassBase *(*create_fn)(),
                             const FusionCycleDetectorPtr &fusion_cycle_detector);

  BaseBufferFusionPassRunner(const std::string &pass_name, BufferFusionPassBase *(*create_fn)(),
                             const FusionCycleDetectorPtr &fusion_cycle_detector,
                             const OpStoreAdapterBasePtr &op_store_adapter_ptr);

  BaseBufferFusionPassRunner(const std::string &pass_name, BufferFusionPassBase *(*create_fn)(),
                             const FusionCycleDetectorPtr &fusion_cycle_detector,
                             const bool is_fusion_check, const OpStoreAdapterBasePtr &op_store_adapter_ptr);

  virtual ~BaseBufferFusionPassRunner();

  Status Run(const ge::ComputeGraph &graph);

 protected:
  virtual Status MatchEachPattern(const ge::ComputeGraph &graph, BufferFusionPattern &pattern,
                                  std::map<int64_t, std::vector<ge::NodePtr>> &match_nodes_map) = 0;
  virtual bool IsNodeFusible(const ge::NodePtr &node) const;
  static bool IsNodePatternMatched(const ge::NodePtr &node, const std::vector<std::string> &patterns);
  static const BufferFusionOpDesc* GetMatchedFusionDesc(const ge::NodePtr &node, const BufferFusionPattern &pattern,
                                                        const BufferFusionMapping &mapping);
  Status GetFusionNodesByMapping(const ge::NodePtr &first_node, const BufferFusionMapping &mapping,
                                 std::vector<ge::NodePtr> &fusion_nodes) const;
  static void AddNodeToMapping(const ge::NodePtr &node, const BufferFusionOpDesc *op_desc,
                               BufferFusionMapping &mapping);
  static int64_t SetFusionNodesScopeId(const std::vector<ge::NodePtr> &fusion_nodes);
  void SetSingleOpUbPassNameAttr(const ge::NodePtr &node) const;
  bool IsNodeDataFlowConnected(const ge::NodePtr &node_a, const ge::NodePtr &node_b) const;
  Status UpdateCycleDetector(const ge::ComputeGraph &graph, const vector<ge::NodePtr> &fusion_nodes);

  static void GetFusionNodesMap(const ge::ComputeGraph &graph,
                                std::map<int64_t, std::vector<ge::NodePtr>> &fusion_nodes_map);
  static void TopoSortForFusionNodes(std::vector<ge::NodePtr> &fusion_nodes);

  Status CheckMatchedNodesCanFusion(const BufferFusionNodeDescMap &fusion_nodes, const ge::NodePtr &next_node);
  Status GetFusionNodes(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) const;
  const std::string& GetPassName() const;
  FusionCycleDetectorPtr GetFusionCycleDetectorPtr() const;

 private:
  void GetFusionPatterns(std::vector<BufferFusionPattern *> &patterns) const;
  void BackUpCycleDetector();
  void RollbackCycleDetector(const ge::ComputeGraph &graph,
                             const std::map<int64_t, std::vector<ge::NodePtr>> &fusion_nodes_map);
  void RestoreCycleDetector();
  UBFusionType GetUBFusionType(const BufferFusionMapping &mapping) const;
  static bool IsCubeVecAllExist(const vector<ge::NodePtr> &fusion_nodes);
  Status FusionCheck(std::map<int64_t, std::vector<ge::NodePtr>> &match_nodes_map, bool &has_rollback_fusion) const;
  void SetFusionNodesPassNameAttr(const std::vector<ge::NodePtr> &fusion_nodes) const;
  static void ReleaseFusionPatterns(std::vector<BufferFusionPattern *> &patterns);
  void CalcSliceInfo(std::vector<ge::NodePtr> &fusion_nodes) const;

 private:
  std::string pass_name_;
  BufferFusionPassBasePtr buffer_fusion_pass_base_ptr_;
  FusionCycleDetectorPtr fusion_cycle_detector_;
  bool is_fusion_check_;
  OpStoreAdapterBasePtr op_store_adapter_ptr_;
};
using BaseBufferFusionPassRunnerPtr = std::shared_ptr<BaseBufferFusionPassRunner>;
}
#endif
