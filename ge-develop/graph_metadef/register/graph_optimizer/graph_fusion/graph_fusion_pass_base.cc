/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"
#include <memory>
#include <string>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"

namespace fe {
GraphFusionPassBase::GraphFusionPassBase() {
  pattern_fusion_base_pass_impl_ptr_ = std::make_shared<PatternFusionBasePassImpl>();
}

GraphFusionPassBase::~GraphFusionPassBase() {}

/**
 * @ingroup fe
 * @brief execute pass
 */
Status GraphFusionPassBase::Run(ge::ComputeGraph &graph) {
  bool is_patterns_ok = true;
  // build Pattern
  std::vector<FusionPattern *> patterns;
  std::string invalid_patterns;
  pattern_fusion_base_pass_impl_ptr_->GetPatterns(patterns);
  if (patterns.empty()) {
    patterns = DefinePatterns();
    for (FusionPattern *pattern : patterns) {
      if (pattern != nullptr) {
        const bool ok = pattern->Build();
        if (!ok) {
          GELOGW("[RunFusionPass][Check] Pattern: %s build failed", pattern->GetName().c_str());
          invalid_patterns += pattern->GetName() + ",";
        }
        pattern->Dump();
        is_patterns_ok = is_patterns_ok && ok;
      }
    }

    pattern_fusion_base_pass_impl_ptr_->SetPatterns(patterns);
  }
  if (!is_patterns_ok) {
    GELOGE(FAILED, "[Check][Patterns]Pattern:%s invalid.", invalid_patterns.c_str());
    return FAILED;
  }

  NodeMapInfoPtr node_map_info = nullptr;
  if (GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph) == SUCCESS) {
    if (node_map_info->run_count == std::numeric_limits<int64_t>::max()) {
      GELOGE(ge::FAILED, "Run count is overflow.");
      return FAILED;
    }
    node_map_info->run_count++;
  }
  // do matching and fusion for each pattern
  bool final_changed = false;
  for (const FusionPattern * const pattern : patterns) {
    if (pattern != nullptr) {
      bool changed = false;
      const Status ret = RunOnePattern(graph, *pattern, changed);
      if (ret != SUCCESS) {
        GELOGW("[RunFusionPass][Check] Run pattern %s failed, the graph was not modified by it.", pattern->GetName().c_str());
        return ret;
      }
      final_changed = final_changed || changed;
    }
  }
  return final_changed ? SUCCESS : NOT_CHANGED;
}

/**
 * @ingroup fe
 * @brief do matching and fusion in graph based on the pattern
 */
Status GraphFusionPassBase::RunOnePattern(ge::ComputeGraph &graph, const FusionPattern &pattern, bool &changed) {
  changed = false;
  Mappings mappings;
  int32_t effect_times = 0;
  const uint32_t graph_id = graph.GetGraphID();
  FusionInfo fusion_info(graph.GetSessionID(), to_string(graph_id), GetName(), static_cast<int32_t>(mappings.size()),
                         effect_times);
  // match all patterns in graph, and save them to mappings
  if (!MatchAll(graph, pattern, mappings)) {
    GELOGD("GraphFusionPass[%s]: pattern=%s, matched_times=%zu, effected_times=%d.", GetName().c_str(),
           pattern.GetName().c_str(), mappings.size(), effect_times);
    return SUCCESS;
  }

  GELOGD("This graph has been matched with pattern[%s]. The mappings are as follows.", pattern.GetName().c_str());

  // print the results of matching
  pattern_fusion_base_pass_impl_ptr_->DumpMappings(pattern, mappings);
  NodeMapInfoPtr node_map_info = nullptr;
  // get nodes by type from node
  (void)GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph);
  // do fusion for each mapping
  for (Mapping &mapping : mappings) {
    std::vector<ge::NodePtr> fus_nodes;
    ge::NodePtr first_node = nullptr;
    for (auto &item : mapping) {
      if (!item.second.empty()) {
        first_node = item.second[0];
        break;
      }
    }

    const Status status = Fusion(graph, mapping, fus_nodes);
    if ((status != SUCCESS) && (status != NOT_CHANGED)) {
      GELOGE(status, "[Fuse][Graph]Fail with pattern[%s].", pattern.GetName().c_str());
      return status;
    }

    if (status == SUCCESS) {
      effect_times++;
      if (!fus_nodes.empty()) {
        // add fusednode to node map info
        for (ge::NodePtr &node : fus_nodes) {
          GraphPassUtil::AddNodeFromOpTypeMap(node_map_info, node);
        }
      }
    }
    changed = changed || (status == SUCCESS);
  }

  // get match times and effect times
  FusionStatisticRecorder &fusion_statistic_inst = FusionStatisticRecorder::Instance();
  fusion_info.SetMatchTimes(static_cast<int32_t>(mappings.size()));
  fusion_info.SetEffectTimes(effect_times);
  fusion_statistic_inst.UpdateGraphFusionMatchTimes(fusion_info);
  fusion_statistic_inst.UpdateGraphFusionEffectTimes(fusion_info);
  GELOGD("GraphId[%d], GraphFusionPass[%s]: pattern=%s, matched_times=%d, effected_times=%d.", graph_id,
         GetName().c_str(), pattern.GetName().c_str(), static_cast<int32_t>(mappings.size()), effect_times);
  return SUCCESS;
}

/**
 * @ingroup fe
 * @brief match all nodes in graph according to pattern
 * match nodes in graph according to pattern, the algorithm is shown as following:
 * 1. get output node from pattern
 * 2. Search for candidate nodes in Graph (network Graph generated after parsing) according to Op Type and
 * (optional), and add the candidate node to the list of candidates
 * 3. For each Node in the candidate list, check whether the type and the number
 * of precursors are consistent with the description of corresponding Op
 * in pattern. If they are consistent, add the precursor Node to the
 * candidate list, and add "PatternOp-GraphNode" to the mapping; otherwise, return an empty mapping
 * 4. repeat step 3 until all the Ops in pattern are matched
 * 5. if all the Ops in pattern are matched successfully, return the mapping of PatternOp and GraphNode
 */
bool GraphFusionPassBase::MatchAll(const ge::ComputeGraph &graph, const FusionPattern &pattern,
    Mappings &mappings) const {
  std::vector<ge::NodePtr> matched_output_nodes;

  // find all the output nodes of pattern in the graph based on Op type
  std::shared_ptr<FusionPattern::OpDesc> output_op_desc = pattern.GetOutput();
  if (output_op_desc == nullptr) {
    return false;
  }

  if (!pattern_fusion_base_pass_impl_ptr_->GetMatchOutputNodes(graph, pattern, matched_output_nodes)) {
    return false;
  }

  // begin matching from every output node
  for (ge::NodePtr &output_node : matched_output_nodes) {
    Mapping mapping;
    if (pattern_fusion_base_pass_impl_ptr_->MatchFromOutput(output_node, output_op_desc, mapping)) {
      mappings.push_back(mapping);
    }
  }
  // if matching is successful, return true; otherwise false
  return !mappings.empty();
}

/**
 * @ingroup fe
 * @brief get an op from mapping according to ID
 */
ge::NodePtr GraphFusionPassBase::GetNodeFromMapping(const std::string &id, const Mapping &mapping) {
  for (auto &item : mapping) {
    const std::shared_ptr<OpDesc> op_desc = item.first;
    if ((op_desc != nullptr) && (op_desc->id == id)) {
      if (item.second.empty()) {
        return nullptr;
      } else {
        return item.second[0];
      }
    }
  }
  return nullptr;
}

}  // namespace fe
