/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/ub_fusion/auto_buffer_fusion_pass_runner.h"

namespace fe {
AutoBufferFusionPassRunner::AutoBufferFusionPassRunner(const std::string &pass_name,
    BufferFusionPassBase *(*create_fn)(), const FusionCycleDetectorPtr &cycle_detector,
    const OpStoreAdapterBasePtr &op_store_adapter_ptr) :
          BaseBufferFusionPassRunner(pass_name, create_fn, cycle_detector, op_store_adapter_ptr) {}
AutoBufferFusionPassRunner::~AutoBufferFusionPassRunner() {}

Status AutoBufferFusionPassRunner::MatchEachPattern(const ge::ComputeGraph &graph, BufferFusionPattern &pattern,
                                                    std::map<int64_t, std::vector<ge::NodePtr>> &match_nodes_map) {
  // get main pattern nodes
  std::vector<ge::NodePtr> main_nodes;
  GetMainPatternNodes(pattern, graph, main_nodes);
  if (main_nodes.empty()) {
    FE_LOGD("Matched main pattern nodes for fusion pattern [%s] are empty.", pattern.GetName().c_str());
    return SUCCESS;
  }
  for (const ge::NodePtr &node : main_nodes) {
    // match from main node
    BufferFusionMapping mapping;
    std::vector<ge::NodePtr> matched_nodes;
    MatchPatternFromNode(node, pattern, mapping, matched_nodes);

    // check min nums
    if (!VerifyMinCountForMatchedNodes(pattern, mapping)) {
      FE_LOGD("The verification of the minimum repeat count did not pass.");
      continue;
    }

    // get fusion nodes by pass
    std::vector<ge::NodePtr> fusion_nodes;
    Status status = GetFusionNodesByMapping(node, mapping, fusion_nodes);
    if (status != SUCCESS && status != NOT_CHANGED) {
      FE_LOGW("Failed to get fusion nodes for pass[%s] and pattern[%s].",
              GetPassName().c_str(), pattern.GetName().c_str());
      return status;
    }
    if (fusion_nodes.empty() || fusion_nodes.size() == 1) {
      FE_LOGD("The number of fusion nodes[%zu] is less than 2, so buffer fusion is not required.", fusion_nodes.size());
      continue;
    }

    std::string reason_not_support;
    if (!graph.IsSupportFuse(fusion_nodes, reason_not_support)) {
      FE_LOGD("IsSupportFuse unsuccessful, reason: [%s].", reason_not_support.c_str());
      continue;
    }

    // sort node by node id
    TopoSortForFusionNodes(fusion_nodes);
    int64_t scope_id = SetFusionNodesScopeId(fusion_nodes);
    match_nodes_map.emplace(scope_id, fusion_nodes);
    (void)UpdateCycleDetector(graph, fusion_nodes);
  }
  return SUCCESS;
}

void AutoBufferFusionPassRunner::MatchPatternFromNode(const ge::NodePtr &node, const BufferFusionPattern &pattern,
                                                      BufferFusionMapping &mapping,
                                                      std::vector<ge::NodePtr> &matched_nodes) {
  if (!IsNodeFusible(node)) {
    FE_LOGD("Node [%s, %s] cannot be fused.", node->GetName().c_str(), node->GetType().c_str());
    return;
  }
  if (std::find(matched_nodes.begin(), matched_nodes.end(), node) != matched_nodes.end()) {
    FE_LOGD("Node[%s, %s] has already been matched.", node->GetName().c_str(), node->GetType().c_str());
    return;
  }

  const BufferFusionOpDesc *op_desc = GetMatchedFusionDesc(node, pattern, mapping);
  if (op_desc == nullptr) {
    FE_LOGD("Node[%s, %s] is not matched with any pattern op desc.", node->GetName().c_str(), node->GetType().c_str());
    return;
  }

  // check is series allowed
  if (!CheckNodeSeries(node, op_desc, mapping)) {
    return;
  }

  // check node relation
  if (!CheckNodeRelation(node, op_desc, mapping)) {
    FE_LOGD("Relation verification for node [%s, %s] did not pass.", node->GetName().c_str(), node->GetType().c_str());
    return;
  }

  // check loop and matched nodes
  matched_nodes.push_back(node);  // add node to matched nodes first
  if (!CheckLoopForMatchedNodes(node, matched_nodes)) {
    matched_nodes.pop_back(); // remove the node which may make a loop
    FE_LOGD("Loop verification of node [%s, %s] did not pass.", node->GetName().c_str(), node->GetType().c_str());
    return;
  }

  // add to mapping
  AddNodeToMapping(node, op_desc, mapping);

  // match input data nodes
  for (const ge::NodePtr &in_node : node->GetInDataNodes()) {
    MatchPatternFromNode(in_node, pattern, mapping, matched_nodes);
  }
  // match output data nodes
  for (const ge::NodePtr &out_node : node->GetOutDataNodes()) {
    MatchPatternFromNode(out_node, pattern, mapping, matched_nodes);
  }
}

bool AutoBufferFusionPassRunner::CheckNodeSeries(const ge::NodePtr &node, const BufferFusionOpDesc *op_desc,
                                                 const BufferFusionMapping &mapping) const {
  if (op_desc->is_allow_series) {
    return true;
  }
  auto iter = mapping.find(op_desc);
  if (iter == mapping.end()) {
    return true;
  }
  for (const ge::NodePtr &matched_node : iter->second) {
    if (IsNodeDataFlowConnected(node, matched_node) || IsNodeDataFlowConnected(matched_node, node)) {
      FE_LOGD("Node[%s, %s] is connected with node[%s, %s].", node->GetName().c_str(), node->GetType().c_str(),
              matched_node->GetName().c_str(), matched_node->GetType().c_str());
      return false;
    }
  }
  return true;
}

bool AutoBufferFusionPassRunner::CheckNodeRelation(const ge::NodePtr &node, const BufferFusionOpDesc *op_desc,
                                                   const BufferFusionMapping &mapping) const {
  for (const std::pair<const BufferFusionOpDesc *, PatternRelation> &relation : op_desc->relations) {
    auto iter = mapping.find(relation.first);
    if (iter == mapping.end() || iter->second.empty()) {
      continue;
    }
    // relative position consistent
    if (relation.second == PatternRelation::RELATIVE_POSITION_CONSISTENT) {
      std::set<bool> relative_position_set;
      std::vector<ge::NodePtr> node_vec;
      auto iter2 = mapping.find(op_desc);
      if (iter2 != mapping.end() && !iter2->second.empty()) {
        node_vec = iter2->second;
      }
      node_vec.push_back(node);
      for (const ge::NodePtr &src_node : node_vec) {
        for (const ge::NodePtr &dst_node : iter->second) {
          bool is_up_connected = IsNodeDataFlowConnected(dst_node, src_node);
          bool is_down_connected = IsNodeDataFlowConnected(src_node, dst_node);
          if (!is_up_connected && !is_down_connected) {
            continue;
          }
          if (is_up_connected && is_down_connected) {
            // cycle loop
            return false;
          }
          if (is_up_connected || is_down_connected) {
            relative_position_set.emplace(is_up_connected);
            if (relative_position_set.size() > 1) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

bool AutoBufferFusionPassRunner::CheckLoopForMatchedNodes(const ge::NodePtr &node,
                                                          const std::vector<ge::NodePtr> &matched_nodes) const {
  if (node == nullptr) {
    return false;
  }
  ge::ComputeGraphPtr graph_ptr = node->GetOwnerComputeGraph();
  return !GetFusionCycleDetectorPtr()->CycleDetection(*graph_ptr, matched_nodes);
}

const BufferFusionOpDesc* AutoBufferFusionPassRunner::GetMainPatternDesc(const BufferFusionPattern &pattern,
    const std::vector<const BufferFusionOpDesc*> &exclude_desc_vec, bool &need_check_next) {
  BufferFusionOpDesc *main_op_desc = nullptr;
  bool is_at_least_one = false;
  for (BufferFusionOpDesc *op_desc : pattern.GetOpDescs()) {
    if (op_desc == nullptr) {
      continue;
    }
    if (std::find(exclude_desc_vec.begin(), exclude_desc_vec.end(), op_desc) != exclude_desc_vec.end()) {
      continue;
    }
    if (op_desc->repeate_min > 0) {
      if (!is_at_least_one) {
        main_op_desc = op_desc;
        is_at_least_one = true;
        break;
      }
    } else {
      if (main_op_desc == nullptr) {
        main_op_desc = op_desc;
      }
    }
  }
  need_check_next = !is_at_least_one;
  return main_op_desc;
}

void AutoBufferFusionPassRunner::GetMainPatternNodes(const BufferFusionPattern &pattern, const ge::ComputeGraph &graph,
                                                     std::vector<ge::NodePtr> &main_nodes) const {
  std::vector<const BufferFusionOpDesc*> exclude_desc_vec;
  bool need_check_next = false;
  const BufferFusionOpDesc *main_op_desc = GetMainPatternDesc(pattern, exclude_desc_vec, need_check_next);
  while (main_op_desc != nullptr) {
    FE_LOGD("Main op desc of fusion pattern[%s] is [%s].", main_op_desc->desc_name.c_str(), pattern.GetName().c_str());
    for (const ge::NodePtr &node : graph.GetDirectNode()) {
      if (!IsNodeFusible(node)) {
        continue;
      }
      if (IsNodePatternMatched(node, main_op_desc->types)) {
        main_nodes.push_back(node);
      }
    }
    if (!main_nodes.empty() || !need_check_next) {
      break;
    }
    exclude_desc_vec.push_back(main_op_desc);
    main_op_desc = GetMainPatternDesc(pattern, exclude_desc_vec, need_check_next);
  }
}

bool AutoBufferFusionPassRunner::VerifyMinCountForMatchedNodes(const BufferFusionPattern &pattern,
                                                               const BufferFusionMapping &mapping) {
  for (const BufferFusionOpDesc *op_desc : pattern.GetOpDescs()) {
    if (op_desc == nullptr || op_desc->repeate_min == 0) {
      continue;
    }
    auto iter = mapping.find(op_desc);
    if (iter == mapping.end()) {
      FE_LOGD("The minimum repeat count of descriptor [%s] and pattern [%s] is [%ld], but the node for this opdesc does not match.",
              op_desc->desc_name.c_str(), pattern.GetName().c_str(), op_desc->repeate_min);
      return false;
    }
    if (static_cast<int64_t>(iter->second.size()) < op_desc->repeate_min) {
      FE_LOGD("The matched node size [%zu] for desc [%s] and pattern [%s] is less than the minimum repeat count [%ld].",
              iter->second.size(), op_desc->desc_name.c_str(), pattern.GetName().c_str(), op_desc->repeate_min);
      return false;
    }
  }
  return true;
}

bool AutoBufferFusionPassRunner::IsNodeFusible(const ge::NodePtr &node) const {
  if (!BaseBufferFusionPassRunner::IsNodeFusible(node)) {
    return false;
  }
  // check is dynamic impl, the template for
  return IsOpDynamicImpl(node->GetOpDesc());
}
}
