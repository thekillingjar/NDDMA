/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_AUTO_BUFFER_FUSION_PASS_RUNNER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_AUTO_BUFFER_FUSION_PASS_RUNNER_H_

#include "graph_optimizer/ub_fusion/base_buffer_fusion_pass_runner.h"

namespace fe {
class AutoBufferFusionPassRunner : public BaseBufferFusionPassRunner {
 public:
  AutoBufferFusionPassRunner(const std::string &pass_name,
                             BufferFusionPassBase *(*create_fn)(),
                             const FusionCycleDetectorPtr &cycle_detector,
                             const OpStoreAdapterBasePtr &op_store_adapter_ptr);

  virtual ~AutoBufferFusionPassRunner();

 protected:
  bool IsNodeFusible(const ge::NodePtr &node) const override;
  Status MatchEachPattern(const ge::ComputeGraph &graph, BufferFusionPattern &pattern,
                          std::map<int64_t, std::vector<ge::NodePtr>> &match_nodes_map) override;

 private:
  static const BufferFusionOpDesc* GetMainPatternDesc(const BufferFusionPattern &pattern,
                                                      const std::vector<const BufferFusionOpDesc*> &exclude_desc_vec,
                                                      bool &need_check_next);
  void GetMainPatternNodes(const BufferFusionPattern &pattern, const ge::ComputeGraph &graph,
                           std::vector<ge::NodePtr> &main_nodes) const;
  void MatchPatternFromNode(const ge::NodePtr &node, const BufferFusionPattern &pattern, BufferFusionMapping &mapping,
                            std::vector<ge::NodePtr> &matched_nodes);
  bool CheckNodeSeries(const ge::NodePtr &node, const BufferFusionOpDesc *op_desc,
                       const BufferFusionMapping &mapping) const;
  bool CheckNodeRelation(const ge::NodePtr &node, const BufferFusionOpDesc *op_desc,
                         const BufferFusionMapping &mapping) const;
  bool CheckLoopForMatchedNodes(const ge::NodePtr &node, const std::vector<ge::NodePtr> &matched_nodes) const;
  static bool VerifyMinCountForMatchedNodes(const BufferFusionPattern &pattern, const BufferFusionMapping &mapping);
};
using AutoBufferFusionPassRunnerPtr = std::shared_ptr<AutoBufferFusionPassRunner>;
}  // namespace fe
#endif
