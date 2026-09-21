/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/graph_matcher.h"

#include <algorithm>
#include <cinttypes>
#include <sstream>

#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/math_util.h"
#include "common/util/op_info_util.h"
#include "graph_optimizer/fusion_common/graph_node_map_util.h"

using ge::NodePtr;
using std::map;
using std::string;
using std::vector;

namespace fe {
// Default anchor number.
static const size_t DEFAULT_ANCHOR_NUM = 1;

// Default node type.
static const string DEFAULT_NODE_TYPE;

GraphMatcher::GraphMatcher() {}

GraphMatcher::~GraphMatcher() {}

static bool CompareRuleNodeType(FusionRuleAnchorPtr anchor1, FusionRuleAnchorPtr anchor2) {
  auto get_first_node_type = [](FusionRuleAnchorPtr anchor) -> string {
    if (anchor != nullptr) {
      FusionRuleNodePtr node = anchor->GetOwnerNode();
      if (node != nullptr) {
        const vector<string> &types = node->GetNodeType();
        if (!types.empty()) {
          return types.front();
        }
      }
    }
    return DEFAULT_NODE_TYPE;
  };

  return get_first_node_type(anchor1) < get_first_node_type(anchor2);
}

static bool CompareGraphNodeType(ge::AnchorPtr anchor1, ge::AnchorPtr anchor2) {
  auto get_node_type = [](ge::AnchorPtr anchor) -> string {
    if (anchor != nullptr) {
      NodePtr node = anchor->GetOwnerNode();
      if (node != nullptr) {
        return node->GetType();
      }
    }
    return DEFAULT_NODE_TYPE;
  };

  return get_node_type(anchor1) < get_node_type(anchor2);
}

Status GraphMatcher::GetRuleOuterInOutAnchorCount() {
  // Calculate outer input count.
  rule_outer_input_count_ = 0;
  for (FusionRuleNodePtr input_rule_node : rule_->GetInputInfo()) {
    if (input_rule_node != nullptr) {
      const auto &out_data_anchors = input_rule_node->GetOutputDataAnchors();
      size_t output_size = 0;
      for (const auto &out_data_anchor : out_data_anchors) {
        if (!out_data_anchor->GetPeerAnchors().empty()) {
          output_size++;
        }
      }
      FE_SIZET_ADDCHECK(rule_outer_input_count_, output_size);
      rule_outer_input_count_ += output_size;
      if (input_rule_node->GetOutputCtrlAnchor()) {
        rule_outer_input_count_++;
      }
    }
  }

  rule_outer_output_count_ = 0;
  for (FusionRuleNodePtr output_rule_node : rule_->GetOutputInfo()) {
    if (output_rule_node != nullptr) {
      size_t output_size = output_rule_node->GetInputDataAnchors().size();
      FE_SIZET_ADDCHECK(rule_outer_output_count_, output_size);
      rule_outer_output_count_ += output_size;
      if (output_rule_node->GetInputCtrlAnchor()) {
        rule_outer_output_count_++;
      }
    }
  }
  return SUCCESS;
}

Status GraphMatcher::Match(const FusionRulePattern &rule, const ge::ComputeGraph &graph,
                           vector<GraphMatchResult> &match_results) {
  FE_LOGD("Match rule %s start.", rule.GetRuleName().c_str());

  // Init variables.
  rule_ = &rule;
  prev_matched_origin_nodes_.clear();
  match_results.clear();

  if (GetRuleOuterInOutAnchorCount() != SUCCESS) {
    return FAILED;
  }

  // Find first output node from rule.
  FusionRuleNodePtr output_rule_node = GetFirstOutputRuleNode();
  if (output_rule_node == nullptr) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][Match] Failed to get the first output node from rule [%s].",
                    rule_->GetRuleName().c_str());
    return GRAPH_MATCHER_GET_RULE_OUTPUT_NODE_FAILED;
  }

  // Find all output nodes from compute graph.
  vector<NodePtr> matched_output_nodes;
  NodeMapInfoPtr node_map_info = nullptr;
  // get nodes by type from node
  if (GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph) != SUCCESS) {
    for (NodePtr &node : graph.GetDirectNode()) {
      if (IsNodeTypeMatch(output_rule_node, node)) {
        matched_output_nodes.push_back(node);
      }
    }
  } else {
    for (auto &OutOpType : output_rule_node->GetNodeType()) {
      GraphPassUtil::GetNodesFromNodeTypeMap(node_map_info->node_type_map, OutOpType, matched_output_nodes);
    }
  }

  // Match graphs from each output node.
  for (NodePtr &output_graph_node : matched_output_nodes) {
    GraphMatchResult match_result;
    if (MatchFromOutput(output_rule_node, output_graph_node, match_result)) {
      match_results.push_back(match_result);
    }
  }

  // Dump matched graphs.
  DumpMatcheResults(match_results);

  // get match times
  uint32_t graph_id = graph.GetGraphID();
  FusionInfo fusion_info(graph.GetSessionID(), to_string(graph_id), rule.GetRuleName(),
                         static_cast<int32_t>(match_results.size()), 0);
  FusionStatisticRecorder::Instance().UpdateGraphFusionMatchTimes(fusion_info);
  FE_LOGD("SessionId %lu GraphId %u Match rule %s end, %zu times matched.", graph.GetSessionID(), graph_id,
          rule.GetRuleName().c_str(), match_results.size());

  return SUCCESS;
}

// Finding order:
// rule -> OuterOutputs -> OuterOutput -> InputAnchors -> InputAnchor ->
// PeerAnchors -> PeerAnchor -> node
FusionRuleNodePtr GraphMatcher::GetFirstOutputRuleNode() const {
  if (rule_ == nullptr) {
    return nullptr;
  }

  const vector<FusionRuleNodePtr> &outer_outputs = rule_->GetOutputInfo();
  if (outer_outputs.empty()) {
    return nullptr;
  }

  FusionRuleNodePtr outer_output = outer_outputs.front();
  if (outer_output == nullptr) {
    return nullptr;
  }

  const vector<FusionRuleAnchorPtr> &outer_input_anchors = outer_output->GetInputDataAnchors();
  if (outer_input_anchors.size() != DEFAULT_ANCHOR_NUM) {
    FE_LOGW("The outer output only supports one input anchor.");
    return nullptr;
  }

  FusionRuleAnchorPtr outer_input_anchor = outer_input_anchors.front();
  if (outer_input_anchor == nullptr) {
    return nullptr;
  }

  // The outer output may contain two input nodes, one inside the origin_graph
  // and the other in the fusion_graph.
  vector<FusionRuleAnchorPtr> peer_anchors = outer_input_anchor->GetPeerAnchors();
  for (FusionRuleAnchorPtr peer_anchor : peer_anchors) {
    if (peer_anchor != nullptr) {
      FusionRuleNodePtr rule_node = peer_anchor->GetOwnerNode();
      if (IsInOriginGraph(rule_node)) {
        return rule_node;
      }
    }
  }

  return nullptr;
}

bool GraphMatcher::MatchFromOutput(FusionRuleNodePtr output_rule_node, NodePtr output_graph_node,
                                   GraphMatchResult &match_result) {
  if (output_rule_node == nullptr || output_graph_node == nullptr || rule_ == nullptr) {
    FE_LOGW("outputRuleNode, output_graph_node or rule_ is nullptr");
    return false;
  }

  candidate_nodes_.clear();
  outer_anchor_idxs_.clear();

  // <output_rule_node, output_graph_node> pair is the first candidate.
  candidate_nodes_.push_back(make_pair(output_rule_node, output_graph_node));

  while (!candidate_nodes_.empty()) {
    // Pop the first node pair.
    auto node_pair = candidate_nodes_.front();
    candidate_nodes_.pop_front();

    FusionRuleNodePtr rule_node = node_pair.first;
    NodePtr graph_node = node_pair.second;
    FE_LOGD("start to match rule:%s, rule node:%s, graph node:%s.", rule_->GetRuleName().c_str(),
            rule_node->GetNodeName().c_str(), graph_node->GetName().c_str());
    if (IsInOriginGraph(rule_node)) {
      // Match graph_node.
      if (!MatchOriginNode(rule_node, graph_node, match_result)) {
        return false;
      }

      // Match successor nodes.
      if (!MatchOriginNodeOutputs(rule_node, graph_node, match_result)) {
        return false;
      }

      // Match predecessor nodes.
      if (!MatchOriginNodeInputs(rule_node, graph_node, match_result)) {
        return false;
      }
    } else if (IsInOuterInputs(rule_node)) {
      // Match successor nodes for outer input.
      if (!MatchOutputsForOuterInput(rule_node, graph_node, match_result)) {
        return false;
      }
    } else {
      // Since the outer output does not support specifying anchor index,
      // the inputs in the graph do not need to match.
    }
  }

  // Add matched origin nodes to prev_matched_origin_nodes_.
  for (auto node_pair : match_result.origin_nodes) {
    prev_matched_origin_nodes_.emplace(node_pair.second);
  }

  // Remove redundant outer outputs.
  if (!RemoveRedundantOuterOutputs(match_result)) {
    return false;
  }

  return VerifyMatchResult(match_result);
}

bool GraphMatcher::MatchOriginNode(FusionRuleNodePtr rule_node, NodePtr graph_node,
                                   GraphMatchResult &match_result) const {
  if (rule_node == nullptr || graph_node == nullptr) {
    FE_LOGW("ruleNode or graph_node is nullptr.");
    return false;
  }

  if (!IsNodeTypeMatch(rule_node, graph_node)) {
    return false;
  }

  // Save origin node.
  match_result.origin_nodes.emplace(rule_node, graph_node);
  match_result.origin_nodes_set.insert(graph_node);
  return true;
}

bool GraphMatcher::MatchOriginNodeInPeers(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                                          GraphMatchResult &match_result) {
  if (rule_anchor->GetAnchorIdx() != graph_anchor->GetIdx()) {
    FE_LOGW(
        "The rule input anchor index[%d] is not matched graph idx[%d], "
        "rule[%s], rule node[%s].",
        rule_anchor->GetAnchorIdx(), graph_anchor->GetIdx(), rule_->GetRuleName().c_str(),
        rule_anchor->GetOwnerNode()->GetNodeName().c_str());
    return false;
  }

  const vector<FusionRuleAnchorPtr> peer_rule_anchors = GetPeerAnchorsNotInFusionGraph(rule_anchor);
  if (peer_rule_anchors.size() != DEFAULT_ANCHOR_NUM) {
    FE_LOGW("Rule peer count of input must be %lu, rule[%s], rule node[%s].", DEFAULT_ANCHOR_NUM,
            rule_->GetRuleName().c_str(), rule_anchor->GetOwnerNode()->GetNodeName().c_str());
    return false;
  }

  // Match input peer.
  auto peer_graph_out_anchors = graph_anchor->GetPeerAnchors();
  if (peer_graph_out_anchors.size() != DEFAULT_ANCHOR_NUM) {
    FE_LOGW("graph peer count of input must be %lu, rule[%s], graph node[%s].", DEFAULT_ANCHOR_NUM,
            rule_->GetRuleName().c_str(), graph_anchor->GetOwnerNode()->GetName().c_str());
    return false;
  }

  if (!MatchPeer(peer_rule_anchors.front(), peer_graph_out_anchors.at(0), graph_anchor, match_result)) {
    return false;
  }

  return true;
}
bool GraphMatcher::MatchOriginNodeInputs(FusionRuleNodePtr rule_node, NodePtr graph_node,
                                         GraphMatchResult &match_result) {
  if (rule_node == nullptr || graph_node == nullptr || rule_ == nullptr) {
    FE_LOGW("ruleNode, graph_node or rule_ is nullptr.");
    return false;
  }
  FE_LOGD("match origin node inputs rule:%s, rule node:%s, graph node:%s.", rule_->GetRuleName().c_str(),
          rule_node->GetNodeName().c_str(), graph_node->GetName().c_str());
  const vector<FusionRuleAnchorPtr> &rule_data_anchors = rule_node->GetInputDataAnchors();
  ge::Node::Vistor<ge::InDataAnchorPtr> graph_data_anchors = graph_node->GetAllInDataAnchors();
  if (rule_data_anchors.size() > graph_data_anchors.size()) {
    FE_LOGW("The number of in anchors of rule node should less than the number in the graph, rule[%s], rule node[%s].",
            rule_->GetRuleName().c_str(), rule_node->GetNodeName().c_str());
    return false;
  }

  FusionRuleAnchorPtr rule_ctrl_anchor = rule_node->GetInputCtrlAnchor();
  ge::InControlAnchorPtr graph_ctrl_anchor = graph_node->GetInControlAnchor();
  if (!IsCtrlAnchorExistMatched(rule_ctrl_anchor, graph_ctrl_anchor)) {
    FE_LOGD("InCtrlAnchor's Size is not matched. rule:%s, rule node[%s], graph node[%s].", rule_->GetRuleName().c_str(),
            rule_node->GetNodeName().c_str(), graph_node->GetName().c_str());
    return false;
  }

  // The rule anchor's index must be continuous, and the anchor is sorted by
  // index.
  for (size_t idx = 0; idx < rule_data_anchors.size(); idx++) {
    FusionRuleAnchorPtr rule_anchor = rule_data_anchors[idx];
    ge::InDataAnchorPtr graph_anchor = graph_data_anchors.at(idx);
    if (rule_anchor == nullptr || graph_anchor == nullptr) {
      FE_LOGW("The %lust input anchor of rule or graph is nullptr, rule[%s], rule node[%s].", idx,
              rule_->GetRuleName().c_str(), rule_node->GetNodeName().c_str());
      return false;
    }

    if (!MatchOriginNodeInPeers(rule_anchor, graph_anchor, match_result)) {
      return false;
    }
  }

  // ctrl anchor to match
  if (rule_ctrl_anchor) {
    const vector<FusionRuleAnchorPtr> peer_rule_anchors = rule_ctrl_anchor->GetPeerAnchors();
    ge::Anchor::Vistor<ge::AnchorPtr> peer_graph_anchors = graph_ctrl_anchor->GetPeerAnchors();
    if (!MatchInOutPeers(peer_rule_anchors, peer_graph_anchors, graph_ctrl_anchor, match_result)) {
      return false;
    }
  }
  // The rest of the graph anchors should not have predecessors.
  for (size_t idx = rule_data_anchors.size(); idx < graph_data_anchors.size(); idx++) {
    if (graph_data_anchors.at(idx) != nullptr && graph_data_anchors.at(idx)->GetPeerOutAnchor() != nullptr) {
      FE_LOGW("Some input anchors in graph have no corresponding rules, rule[%s], rule node[%s].",
              rule_->GetRuleName().c_str(), rule_node->GetNodeName().c_str());
      return false;
    }
  }
  return true;
}

bool GraphMatcher::MatchOriginNodeOutPeers(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                                           GraphMatchResult &match_result) {
  // Match output peers.
  // The nodes in the original graph cannot be connected to the nodes in the
  // fused graph,
  // which has been verified when the rules are loaded.
  const vector<FusionRuleAnchorPtr> peer_rule_anchors = rule_anchor->GetPeerAnchors();
  ge::Anchor::Vistor<ge::AnchorPtr> peer_graph_anchors = graph_anchor->GetPeerAnchors();

  if (HasOuterOutput(peer_rule_anchors)) {
    // rule_node`s next node is only one and type is outer_output
    bool outer_output_no_successor_flag = peer_rule_anchors.size() == 1 && IsOuterOutput(peer_rule_anchors[0]);
    if (outer_output_no_successor_flag) {
      if (peer_graph_anchors.empty()) {  // graph_node has no successors
        FusionRuleNodePtr rule_node = rule_anchor->GetOwnerNode();
        NodePtr graph_node = graph_anchor->GetOwnerNode();
        FE_LOGW("Output does not have an outer_output. Rule node [%s], graph node [%s].", rule_node->GetNodeName().c_str(),
                graph_node->GetName().c_str());
        match_result.outer_outputs.emplace(peer_rule_anchors[0], set<ge::AnchorPtr>{});
        return true;
      }
    }

    if (!MatchPeersWithOuterOutput(peer_rule_anchors, peer_graph_anchors, match_result)) {
      return false;
    }
  } else {
    if (!MatchInOutPeers(peer_rule_anchors, peer_graph_anchors, nullptr, match_result)) {
      return false;
    }
  }
  return true;
}

bool GraphMatcher::MatchOriginNodeOutputs(FusionRuleNodePtr rule_node, NodePtr graph_node,
                                          GraphMatchResult &match_result) {
  FE_CHECK(rule_node == nullptr, FE_LOGW("ruleNode is nullptr."), return false);
  FE_CHECK(graph_node == nullptr, FE_LOGW("graphNode is nullptr."), return false);
  FE_CHECK(rule_ == nullptr, FE_LOGW("rule_ is nullptr."), return false);
  FE_LOGD("Matched origin node outputs rule: %s, rule node: %s, graph node: %s.", rule_->GetRuleName().c_str(),
          rule_node->GetNodeName().c_str(), graph_node->GetName().c_str());

  const vector<FusionRuleAnchorPtr> &rule_data_anchors = rule_node->GetOutputDataAnchors();
  ge::Node::Vistor<ge::OutDataAnchorPtr> graph_data_anchors = graph_node->GetAllOutDataAnchors();
  if (rule_data_anchors.size() > graph_data_anchors.size()) {
    FE_LOGW("Size of out anchors of rule node should be less than the size in the graph, rule[%s], rule node[%s].",
            rule_->GetRuleName().c_str(), rule_node->GetNodeName().c_str());
    return false;
  }

  const FusionRuleAnchorPtr rule_ctrl_anchor = rule_node->GetOutputCtrlAnchor();
  ge::OutControlAnchorPtr graph_ctrl_anchor = graph_node->GetOutControlAnchor();
  if (!IsCtrlAnchorExistMatched(rule_ctrl_anchor, graph_ctrl_anchor)) {
    FE_LOGD("OutCtrlAnchor's Size is not matched, rule:%s, rule node[%s], graph node[%s].",
            rule_->GetRuleName().c_str(), rule_node->GetNodeName().c_str(), graph_node->GetName().c_str());
    return false;
  }

  // The rule anchor's index must be continuous, and the anchor is sorted by index.
  for (size_t idx = 0; idx < rule_data_anchors.size(); idx++) {
    FusionRuleAnchorPtr rule_anchor = rule_data_anchors[idx];
    FE_CHECK(rule_anchor == nullptr, FE_LOGW("rule_anchor is null."), return false);
    ge::AnchorPtr graph_anchor = graph_data_anchors.at(idx);
    if (graph_anchor == nullptr || rule_anchor->GetAnchorIdx() != graph_anchor->GetIdx()) {
      FE_LOGW("The %zu output anchor is nullptr or the anchor index [%d] is not matched, rule:%s, rule node[%s].",
              idx, rule_anchor->GetAnchorIdx(), rule_->GetRuleName().c_str(), rule_node->GetNodeName().c_str());
      return false;
    }

    if (!MatchOriginNodeOutPeers(rule_anchor, graph_anchor, match_result)) {
      return false;
    }
  }

  // The rest of the graph anchors should not have successors.
  if (HasPeers(graph_data_anchors, rule_data_anchors.size())) {
    FE_LOGW("Some output anchors in graph have no corresponding rules, rule[%s], rule node[%s].",
            rule_->GetRuleName().c_str(), rule_node->GetNodeName().c_str());
    return false;
  }

  // ctrl anchor to match
  if (rule_ctrl_anchor) {
    if (!MatchOriginNodeOutPeers(rule_ctrl_anchor, graph_ctrl_anchor, match_result)) {
      return false;
    }
    FE_LOGD("rule:%s, rule node[%s], match ctrl anchor successfully.", rule_->GetRuleName().c_str(),
            rule_node->GetNodeName().c_str());
  }

  return true;
}

bool GraphMatcher::MatchOuterInputOutPeers(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                                           GraphMatchResult &match_result) {
  // Match output peers for outer input.
  vector<FusionRuleAnchorPtr> peer_rule_origin_anchors = GetPeerOriginAnchors(rule_anchor);
  ge::Anchor::Vistor<ge::AnchorPtr> peer_graph_anchors = graph_anchor->GetPeerAnchors();
  if (peer_rule_origin_anchors.size() == peer_graph_anchors.size()) {
    if (!MatchInOutPeers(peer_rule_origin_anchors, peer_graph_anchors, nullptr, match_result)) {
      return false;
    }
  } else if (peer_rule_origin_anchors.size() > peer_graph_anchors.size()) {
    return false;
  }

  return true;
}

bool GraphMatcher::MatchOutputsForOuterInput(FusionRuleNodePtr rule_node, NodePtr graph_node,
                                             GraphMatchResult &match_result) {
  FE_CHECK(rule_node == nullptr || graph_node == nullptr || rule_ == nullptr,
           FE_LOGW("ruleNode, graph_node or rule is nullptr."), return false);

  FE_LOGD("Matched outer input to outputs rule: %s, rule node: %s, graph node: %s.", rule_->GetRuleName().c_str(),
          rule_node->GetNodeName().c_str(), graph_node->GetName().c_str());

  const FusionRuleAnchorPtr rule_ctrl_anchor = rule_node->GetOutputCtrlAnchor();
  ge::OutControlAnchorPtr graph_ctrl_anchor = graph_node->GetOutControlAnchor();
  if (!CheckOuterInputCtrlAnchor(rule_ctrl_anchor, graph_ctrl_anchor)) {
    FE_LOGD("InCtrlAnchor's Size is not matched, rule:%s, rule node[%s], graph node[%s].", rule_->GetRuleName().c_str(),
            rule_node->GetNodeName().c_str(), graph_node->GetName().c_str());
    return false;
  }

  const vector<FusionRuleAnchorPtr> &rule_anchors = rule_node->GetOutputDataAnchors();
  ge::Node::Vistor<ge::OutDataAnchorPtr> graph_anchors = graph_node->GetAllOutDataAnchors();

  for (const FusionRuleAnchorPtr &rule_anchor : rule_anchors) {
    if (rule_anchor == nullptr) {
      FE_LOGW("Anchor of outer input is nullptr, rule[%s].", rule_->GetRuleName().c_str());
      return false;
    }

    // Get anchor index.
    int anchor_idx = rule_anchor->GetAnchorIdx();
    if (anchor_idx == DEFAULT_ANCHOR_INDEX) {
      // The outer input has at most one output anchor with a value of
      // DEFAULT_ANCHOR_INDEX,
      // and this anchor has been added to the outer_anchor_idxs_.
      const auto &iter = outer_anchor_idxs_.find(rule_anchor);
      if (iter == outer_anchor_idxs_.cend()) {
        // if not found, it indicates the node is matched by ctrl anchor, wait
        // until matched by data anchor
        if (rule_ctrl_anchor) {
          return true;
        }
        return false;
      }
      anchor_idx = iter->second;
    }

    if (anchor_idx < 0 || static_cast<size_t>(anchor_idx) >= graph_anchors.size()) {
      FE_LOGW("Src index %d of outer input is not matched, rule[%s], outer input[%s].", anchor_idx,
              rule_->GetRuleName().c_str(), rule_anchor->GetAnchorName().c_str());
      return false;
    }

    ge::OutDataAnchorPtr graph_anchor = graph_anchors.at(anchor_idx);
    if (graph_anchor == nullptr || anchor_idx != graph_anchor->GetIdx()) {
      FE_LOGW("The %dst output anchor of node %s is not matched, rule[%s], rule node[%s].", anchor_idx,
              graph_node->GetName().c_str(), rule_->GetRuleName().c_str(), rule_anchor->GetAnchorName().c_str());
      return false;
    }

    // Match output peers for outer input.
    if (!MatchOuterInputOutPeers(rule_anchor, graph_anchor, match_result)) {
      return false;
    }
  }

  if (rule_ctrl_anchor) {
    // Match output peers for outer input.
    if (!MatchOuterInputOutPeers(rule_ctrl_anchor, graph_ctrl_anchor, match_result)) {
      FE_LOGD("OutCtrlAnchor for OuterInput is not matched, rule:%s, rule node[%s], graph node[%s].",
              rule_->GetRuleName().c_str(), rule_node->GetNodeName().c_str(), graph_node->GetName().c_str());
      return false;
    }
  }

  return true;
}

bool GraphMatcher::MatchInOutPeers(const vector<FusionRuleAnchorPtr> &peer_rule_anchors,
                                   const ge::Anchor::Vistor<ge::AnchorPtr> &peer_graph_anchors,
                                   ge::AnchorPtr graph_origin_anchor, GraphMatchResult &match_result) {
  if (peer_rule_anchors.size() != peer_graph_anchors.size()) {
    return false;
  }

  // The output order in the rule may not be same with the order in the graph.
  // Use sorting to ensure a correct match.
  vector<FusionRuleAnchorPtr> copied_peer_rule_anchors = peer_rule_anchors;
  ge::Anchor::Vistor<ge::AnchorPtr> copied_peer_graph_anchors = peer_graph_anchors;
  sort(copied_peer_rule_anchors.begin(), copied_peer_rule_anchors.end(), CompareRuleNodeType);
  sort(copied_peer_graph_anchors.begin(), copied_peer_graph_anchors.end(), CompareGraphNodeType);

  // Match input peers.
  for (size_t peer_idx = 0; peer_idx < copied_peer_rule_anchors.size(); peer_idx++) {
    if (!MatchPeer(copied_peer_rule_anchors[peer_idx], copied_peer_graph_anchors.at(peer_idx), graph_origin_anchor,
                   match_result)) {
      return false;
    }
  }

  return true;
}

bool GraphMatcher::MatchPeersWithOuterOutput(const vector<FusionRuleAnchorPtr> &peer_rule_anchors,
                                             const ge::Anchor::Vistor<ge::AnchorPtr> &peer_graph_data_anchors,
                                             GraphMatchResult &match_result) {
  // Separate origin anchors and outer output anchor of rule.
  vector<FusionRuleAnchorPtr> peer_rule_origin_anchors;
  FusionRuleAnchorPtr peer_rule_outer_output_anchor = nullptr;
  for (const FusionRuleAnchorPtr &peer_anchor : peer_rule_anchors) {
    if (IsOuterOutput(peer_anchor)) {
      if (peer_rule_outer_output_anchor == nullptr) {
        peer_rule_outer_output_anchor = peer_anchor;
      } else {
        FE_LOGW("The output of rule node can only be connected to one outer output.");
        return false;
      }
    } else {
      peer_rule_origin_anchors.push_back(peer_anchor);
    }
  }

  // The output order in the rule may not be same with the order in the graph.
  // Use sorting to ensure a correct match.
  ge::OutDataAnchor::Vistor<ge::AnchorPtr> copied_peer_graph_data_anchors = peer_graph_data_anchors;
  sort(peer_rule_origin_anchors.begin(), peer_rule_origin_anchors.end(), CompareRuleNodeType);
  sort(copied_peer_graph_data_anchors.begin(), copied_peer_graph_data_anchors.end(), CompareGraphNodeType);

  // Separate origin anchors and outer output anchors of graph.
  map<FusionRuleAnchorPtr, ge::AnchorPtr> peer_origin_anchor_map;
  vector<ge::AnchorPtr> peer_graph_outer_output_anchors;
  SeparateOriginAndOuterOutputAnchors(copied_peer_graph_data_anchors, peer_rule_origin_anchors, peer_origin_anchor_map,
                                      peer_graph_outer_output_anchors);

  // Match peer origin nodes.
  // If the peer nodes are of the same type, they may not match.
  for (auto anchor_pair : peer_origin_anchor_map) {
    if (!MatchPeer(anchor_pair.first, anchor_pair.second, nullptr, match_result)) {
      return false;
    }
  }

  // Match peer outer anchors.
  for (ge::AnchorPtr graph_anchor : peer_graph_outer_output_anchors) {
    if (!MatchPeer(peer_rule_outer_output_anchor, graph_anchor, nullptr, match_result)) {
      return false;
    }
  }

  return true;
}

bool GraphMatcher::MatchPeer(FusionRuleAnchorPtr peer_rule_anchor, ge::AnchorPtr peer_graph_anchor,
                             ge::AnchorPtr graph_origin_anchor, GraphMatchResult &match_result) {
  if (peer_rule_anchor == nullptr || peer_graph_anchor == nullptr) {
    return false;
  }

  // Check anchor index.
  if (peer_rule_anchor->GetAnchorIdx() != DEFAULT_ANCHOR_INDEX &&
      peer_rule_anchor->GetAnchorIdx() != peer_graph_anchor->GetIdx()) {
    return false;
  }

  // Avoid repeat match if the node have been matched in other matchs.
  NodePtr peer_graph_node = peer_graph_anchor->GetOwnerNode();
  bool check_repeat_matched = (prev_matched_origin_nodes_.find(peer_graph_node) != prev_matched_origin_nodes_.cend() &&
                               !IsOuterInput(peer_rule_anchor) && !IsOuterOutput(peer_rule_anchor));
  if (check_repeat_matched) {
    return false;
  }

  // Check if the node have been matched in this match.
  FusionRuleNodePtr peer_rule_node = peer_rule_anchor->GetOwnerNode();
  const auto matched_origin_node_iter = match_result.origin_nodes.find(peer_rule_node);
  const auto matched_outer_input_iter = match_result.outer_inputs.find(peer_rule_anchor);

  if (IsOriginAnchor(peer_rule_anchor)) {
    if (matched_outer_input_iter != match_result.outer_inputs.cend()) {
      return false;
    }

    if (matched_origin_node_iter != match_result.origin_nodes.cend()) {
      return matched_origin_node_iter->second == peer_graph_node;
    }
  } else {
    // If the outer node does not specify an anchor index, use the index in the
    // graph.
    if (peer_rule_anchor->GetAnchorIdx() == DEFAULT_ANCHOR_INDEX) {
      outer_anchor_idxs_.emplace(peer_rule_anchor, peer_graph_anchor->GetIdx());
    }

    if (IsOuterInput(peer_rule_anchor)) {
      // The matched original node cannot be used as an outer input.
      if (matched_origin_node_iter != match_result.origin_nodes.cend()) {
        return false;
      }

      // Verify the matched outer input.
      if (matched_outer_input_iter != match_result.outer_inputs.cend()) {
        return matched_outer_input_iter->second == peer_graph_anchor;
      }

      // Save outer input to match_result.outer_inputs.
      match_result.outer_inputs.emplace(peer_rule_anchor, peer_graph_anchor);
      FE_CHECK(graph_origin_anchor == nullptr,
               REPORT_FE_ERROR("[GraphOpt][RunFusionRule][MatchPeer] The input origin anchor is null"),
               return false);
      match_result.origin_outer_inputs.emplace(peer_rule_anchor, graph_origin_anchor);
    } else if (IsOuterOutput(peer_rule_anchor)) {
      // Save outer output to match_result.outer_outputs.
      // We will continue to verify outer outputs within function
      // RemoveRedundantOuterOutputs.
      AddOuterOutputToResult(peer_rule_anchor, peer_graph_anchor, match_result);
    }
  }

  // Add the peer node to the candidate list.
  candidate_nodes_.push_back(make_pair(peer_rule_node, peer_graph_node));

  return true;
}

bool GraphMatcher::IsNodeTypeMatch(FusionRuleNodePtr rule_node, NodePtr graph_node) const {
  if (rule_node == nullptr || graph_node == nullptr) {
    FE_LOGW("ruleNode or graph_node is nullptr.");
    return false;
  }

  const vector<string> &rule_types = rule_node->GetNodeType();
  return find(rule_types.begin(), rule_types.end(), graph_node->GetType()) != rule_types.end();
}

// Determine whether the outer outputs matches based on the type of predecessor
// nodes.
bool GraphMatcher::IsOuterOutputMatch(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                                      const GraphMatchResult &match_result) const {
  if (rule_anchor == nullptr || graph_anchor == nullptr) {
    FE_LOGW("ruleAnchor or graph_anchor is nullptr.");
    return false;
  }

  // Get peer rule anchor.
  const vector<FusionRuleAnchorPtr> peer_rule_anchors = GetPeerAnchorsNotInFusionGraph(rule_anchor);
  if (peer_rule_anchors.size() != DEFAULT_ANCHOR_NUM) {
    FE_LOGW("Peer count of input must be %lu, rule:%s.", DEFAULT_ANCHOR_NUM, rule_->GetRuleName().c_str());
    return false;
  }

  // Get predecessor rule node.
  FusionRuleNodePtr pred_rule_node = nullptr;
  FusionRuleAnchorPtr peer_rule_anchor = peer_rule_anchors.front();
  if (peer_rule_anchor != nullptr) {
    pred_rule_node = peer_rule_anchor->GetOwnerNode();
  }

  // Get matched predecessor graph node.
  NodePtr matched_graph_node = nullptr;
  auto iter = match_result.origin_nodes.find(pred_rule_node);
  if (iter != match_result.origin_nodes.end()) {
    matched_graph_node = iter->second;
  }

  // Get the predecessor graph node.
  ge::Anchor::Vistor<ge::AnchorPtr> peer_graph_anchors = graph_anchor->GetPeerAnchors();
  for (ge::AnchorPtr const &peer_anchor : peer_graph_anchors) {
    if (peer_anchor != nullptr) {
      if (peer_anchor->GetOwnerNode() == matched_graph_node) {
        return true;
      }
    }
  }

  return false;
}

bool GraphMatcher::IsInOuterInputs(FusionRuleNodePtr rule_node) const {
  if (rule_node == nullptr || rule_ == nullptr) {
    FE_LOGW("ruleNode or rule_ is nullptr.");
    return false;
  }

  const vector<FusionRuleNodePtr> &nodes = rule_->GetInputInfo();
  return find(nodes.begin(), nodes.end(), rule_node) != nodes.end();
}

bool GraphMatcher::IsInOuterOutputs(FusionRuleNodePtr rule_node) const {
  if (rule_node == nullptr || rule_ == nullptr) {
    FE_LOGW("ruleNode or rule_ is nullptr.");
    return false;
  }

  const vector<FusionRuleNodePtr> &nodes = rule_->GetOutputInfo();
  return find(nodes.begin(), nodes.end(), rule_node) != nodes.end();
}

bool GraphMatcher::IsInOriginGraph(FusionRuleNodePtr rule_node) const {
  if (rule_node == nullptr || rule_ == nullptr) {
    FE_LOGW("ruleNode or rule_ is nullptr.");
    return false;
  }

  const set<FusionRuleNodePtr> &nodes = rule_->GetOriginRuleNodes();
  return nodes.find(rule_node) != nodes.end();
}

bool GraphMatcher::IsInFusionGraph(FusionRuleNodePtr rule_node) const {
  if (rule_node == nullptr || rule_ == nullptr) {
    FE_LOGW("ruleNode or rule_ is nullptr.");
    return false;
  }

  const set<FusionRuleNodePtr> &nodes = rule_->GetFusionRuleNodes();
  return nodes.find(rule_node) != nodes.end();
}

bool GraphMatcher::IsOuterInput(FusionRuleAnchorPtr anchor) const {
  return (anchor != nullptr) && IsInOuterInputs(anchor->GetOwnerNode());
}

bool GraphMatcher::IsOuterOutput(FusionRuleAnchorPtr anchor) const {
  return (anchor != nullptr) && IsInOuterOutputs(anchor->GetOwnerNode());
}

bool GraphMatcher::IsOriginAnchor(FusionRuleAnchorPtr anchor) const {
  return (anchor != nullptr) && IsInOriginGraph(anchor->GetOwnerNode());
}

bool GraphMatcher::IsCtrlAnchorExistMatched(FusionRuleAnchorPtr rule_ctrl_anchor,
                                            ge::ControlAnchorPtr graph_ctrl_anchor) const {
  bool ctrl_anchor_exist = rule_ctrl_anchor && (graph_ctrl_anchor->GetPeerAnchorsSize() != 0);
  bool ctrl_anchor_not_exist = !rule_ctrl_anchor && (graph_ctrl_anchor->GetPeerAnchorsSize() == 0);
  if (ctrl_anchor_exist || ctrl_anchor_not_exist) {
    return true;
  }
  return false;
}

bool GraphMatcher::CheckOuterInputCtrlAnchor(FusionRuleAnchorPtr rule_ctrl_anchor,
                                             ge::ControlAnchorPtr graph_ctrl_anchor) const {
  if (rule_ctrl_anchor) {
    if (graph_ctrl_anchor->GetPeerAnchorsSize() == 0) {
      return false;
    }
  }

  return true;
}

bool GraphMatcher::HasOuterOutput(const vector<FusionRuleAnchorPtr> &anchors) const {
  for (const FusionRuleAnchorPtr &anchor : anchors) {
    if (IsOuterOutput(anchor)) {
      return true;
    }
  }

  return false;
}

bool GraphMatcher::HasPeers(const ge::Node::Vistor<ge::OutDataAnchorPtr> &graph_anchors, size_t start_idx) const {
  for (size_t idx = start_idx; idx < graph_anchors.size(); idx++) {
    ge::OutDataAnchorPtr graph_anchor = graph_anchors.at(idx);
    if (graph_anchor != nullptr) {
      auto peer_anchors = graph_anchor->GetPeerInDataAnchors();
      if (!peer_anchors.empty()) {
        return true;
      }
    }
  }

  return false;
}

vector<FusionRuleAnchorPtr> GraphMatcher::GetPeerAnchorsNotInFusionGraph(FusionRuleAnchorPtr rule_anchor) const {
  vector<FusionRuleAnchorPtr> anchors;
  if (rule_anchor != nullptr) {
    vector<FusionRuleAnchorPtr> peer_anchors = rule_anchor->GetPeerAnchors();
    for (FusionRuleAnchorPtr const &peer_anchor : peer_anchors) {
      if (peer_anchor != nullptr) {
        if (!IsInFusionGraph(peer_anchor->GetOwnerNode())) {
          anchors.push_back(peer_anchor);
        }
      }
    }
  }

  return anchors;
}

vector<FusionRuleAnchorPtr> GraphMatcher::GetPeerOriginAnchors(FusionRuleAnchorPtr rule_anchor) const {
  vector<FusionRuleAnchorPtr> peer_origin_anchors;
  const vector<FusionRuleAnchorPtr> peer_anchors = rule_anchor->GetPeerAnchors();
  for (FusionRuleAnchorPtr const &peer_anchor : peer_anchors) {
    if (peer_anchor != nullptr && IsInOriginGraph(peer_anchor->GetOwnerNode())) {
      peer_origin_anchors.push_back(peer_anchor);
    }
  }

  return peer_origin_anchors;
}

void GraphMatcher::SeparateOriginAndOuterOutputAnchors(const ge::Anchor::Vistor<ge::AnchorPtr> &peer_graph_anchors,
                                                       const vector<FusionRuleAnchorPtr> &peer_rule_origin_anchors,
                                                       map<FusionRuleAnchorPtr, ge::AnchorPtr> &peer_origin_anchor_map,
                                                       vector<ge::AnchorPtr> &peer_graph_outer_output_anchors) const {
  // Init peer_origin_anchor_map use nullptr.
  for (FusionRuleAnchorPtr const &peer_rule_anchor : peer_rule_origin_anchors) {
    peer_origin_anchor_map.emplace(peer_rule_anchor, nullptr);
  }

  // Fill peer_graph_outer_output_anchors.
  if (peer_rule_origin_anchors.empty()) {
    std::copy(peer_graph_anchors.begin(), peer_graph_anchors.end(),
              std::back_inserter(peer_graph_outer_output_anchors));
  }

  // Fill peer_origin_anchor_map and peer_graph_outer_output_anchors.
  size_t graph_next_idx = 0;
  for (FusionRuleAnchorPtr const &peer_rule_anchor : peer_rule_origin_anchors) {
    for (size_t graph_idx = graph_next_idx; graph_idx < peer_graph_anchors.size(); graph_idx++) {
      ge::AnchorPtr peer_graph_anchor = peer_graph_anchors.at(graph_idx);
      if (peer_rule_anchor != nullptr && peer_graph_anchor != nullptr) {
        if (IsNodeTypeMatch(peer_rule_anchor->GetOwnerNode(), peer_graph_anchor->GetOwnerNode())) {
          peer_origin_anchor_map[peer_rule_anchor] = peer_graph_anchor;
          graph_next_idx = graph_idx + 1;
          break;
        }
      }
      peer_graph_outer_output_anchors.push_back(peer_graph_anchor);
    }
  }
}

void GraphMatcher::AddOuterOutputToResult(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                                          GraphMatchResult &result) const {
  const auto iter = result.outer_outputs.find(rule_anchor);
  if (iter != result.outer_outputs.cend()) {
    iter->second.emplace(graph_anchor);
  } else {
    set<ge::AnchorPtr> outer_outputs = {graph_anchor};
    result.outer_outputs.emplace(rule_anchor, outer_outputs);
  }
}

bool GraphMatcher::RemoveRedundantOuterOutputs(GraphMatchResult &match_result) const {
  // Remove the anchors of the origin node from match_result.outer_outputs.
  for (auto &out_pair : match_result.outer_outputs) {
    FusionRuleAnchorPtr rule_anchor = out_pair.first;
    set<ge::AnchorPtr> &graph_anchors = out_pair.second;

    if (graph_anchors.empty()) {
      continue;
    }

    set<ge::AnchorPtr> copied_graph_anchors;
    set<ge::AnchorPtr> origin_graph_anchors;
    for (ge::AnchorPtr const &graph_anchor : graph_anchors) {
      if (IsOuterOutputMatch(rule_anchor, graph_anchor, match_result)) {
        copied_graph_anchors.emplace(graph_anchor);
        origin_graph_anchors.emplace(graph_anchor->GetFirstPeerAnchor());
      }
    }

    if (copied_graph_anchors.empty()) {
      return false;
    }

    match_result.origin_outer_outputs.insert(std::make_pair(rule_anchor, origin_graph_anchors));
    graph_anchors = copied_graph_anchors;
  }

  return true;
}

// Check if the rule nodes and anchors are all matched.
bool GraphMatcher::VerifyMatchResult(const GraphMatchResult &match_result) const {
  if (match_result.outer_inputs.size() != rule_outer_input_count_ ||
      match_result.origin_nodes.size() != rule_->GetOriginRuleNodes().size() ||
      match_result.outer_outputs.size() != rule_outer_output_count_) {
    FE_LOGD(
        "Some nodes or anchors of rule %s are not matched, rule"
        "outerInputs:%lu, matched outer_inputs:%lu,"
        "rule origin_nodes:%lu, matched origin_nodes:%lu, rule"
        "outerOutputs:%lu, matched outer_outputs:%lu.",
        rule_->GetRuleName().c_str(), rule_outer_input_count_, match_result.outer_inputs.size(),
        rule_->GetOriginRuleNodes().size(), match_result.origin_nodes.size(), rule_outer_output_count_,
        match_result.outer_outputs.size());
    return false;
  }

  return true;
}

// Example:
// Matched 2 graphs for rule rule1:
//   Graph 1/2:
//     Input1 src: GraphNode1:0
//     Output1 dst: GraphNode4:0, GraphNode5:0
//     RuleNode1: GraphNode2
//     RuleNode2: GraphNode3
//   Graph 2/2:
//     Input1 src: GraphNode1:0
//     Output1 dst: GraphNode4:0, GraphNode5:0
//     RuleNode1: GraphNode2
//     RuleNode2: GraphNode3
void GraphMatcher::DumpMatcheResults(const vector<GraphMatchResult> &match_results) const {
  if (rule_ == nullptr) {
    FE_LOGW("rule_ is nullptr.");
    return;
  }

  size_t result_num = match_results.size();

  std::ostringstream oss_summary;
  oss_summary << "Matched " << result_num << " results for rule " << rule_->GetRuleName() << ":";
  FE_LOGD("%s", oss_summary.str().c_str());

  for (size_t idx = 0; idx < result_num; idx++) {
    std::ostringstream oss;
    oss << "  Graph " << (idx + 1) << "/" << result_num << ":";
    FE_LOGD("%s", oss.str().c_str());

    const GraphMatchResult &result = match_results[idx];
    DumpMatchedOuterInputs(result);
    DumpMatchedOuterOutputs(result);
    DumpMatchedOriginNodes(result);
  }
}

void GraphMatcher::DumpMatchedOuterInputs(const GraphMatchResult &result) const {
  for (const auto &anchor_pair : result.outer_inputs) {
    FusionRuleAnchorPtr rule_anchor = anchor_pair.first;
    ge::AnchorPtr graph_anchor = anchor_pair.second;
    if (rule_anchor != nullptr && graph_anchor != nullptr) {
      NodePtr graph_node = graph_anchor->GetOwnerNode();
      if (graph_node != nullptr) {
        auto iter = result.origin_outer_inputs.find(rule_anchor);
        if (iter == result.origin_outer_inputs.end()) {
          continue;
        }
        std::ostringstream oss;
        ge::AnchorPtr graph_origin_anchor = iter->second;
        NodePtr graph_origin_node = graph_origin_anchor->GetOwnerNode();
        oss << "    " << rule_anchor->GetAnchorName() << " src: " << graph_node->GetName() << ":"
            << graph_anchor->GetIdx() << " ori:" << graph_origin_node->GetName() << ":"
            << graph_origin_anchor->GetIdx();
        FE_LOGD("%s", oss.str().c_str());
      }
    }
  }
}

void GraphMatcher::DumpMatchedOuterOutputs(const GraphMatchResult &result) const {
  for (const auto &anchor_pair : result.outer_outputs) {
    FusionRuleAnchorPtr rule_anchor = anchor_pair.first;
    if (rule_anchor != nullptr) {
      std::ostringstream oss;
      oss << "    " << rule_anchor->GetAnchorName() << " dst: ";
      const auto &graph_anchors = anchor_pair.second;
      for (ge::AnchorPtr const &graph_anchor : graph_anchors) {
        if (graph_anchor != nullptr) {
          NodePtr graph_node = graph_anchor->GetOwnerNode();
          if (graph_node == nullptr) {
            continue;
          }

          if (graph_anchor->IsTypeOf<ge::DataAnchor>()) {
            auto data_anchor = ge::Anchor::DynamicAnchorCast<ge::DataAnchor>(graph_anchor);
            FE_CHECK(data_anchor == nullptr, REPORT_FE_ERROR("data_anchor is nullptr."), return);
            oss << graph_node->GetName() << ":" << data_anchor->GetIdx() << ", ";
          } else {
            oss << graph_node->GetName() << ":-1(ctrl), ";
          }
        }
      }

      auto iter = result.origin_outer_outputs.find(rule_anchor);
      if (iter == result.origin_outer_outputs.end()) {
        continue;
      }
      for (auto const &ori_anchor : iter->second) {
        NodePtr graph_node = ori_anchor->GetOwnerNode();
        oss << "ori:" << graph_node->GetName() << ":" << ori_anchor->GetIdx() << ", ";
      }
      FE_LOGD("%s", oss.str().c_str());
    }
  }
}

void GraphMatcher::DumpMatchedOriginNodes(const GraphMatchResult &result) const {
  for (const auto &node_pair : result.origin_nodes) {
    FusionRuleNodePtr rule_node = node_pair.first;
    NodePtr graph_node = node_pair.second;
    if (rule_node != nullptr && graph_node != nullptr) {
      std::ostringstream oss;
      oss << "    " << rule_node->GetNodeName() << ": " << graph_node->GetName();
      FE_LOGD("%s", oss.str().c_str());
    }
  }
}
}  // namespace fe
