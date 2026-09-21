/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_GRAPH_MATCHER_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_GRAPH_MATCHER_H_
#include <list>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/node.h"

#include "common/fe_inner_error_codes.h"
#include "fusion_rule_manager/fusion_rule_data/fusion_rule_pattern.h"
#include "graph_optimizer/fusion_common/fusion_statistic_recorder.h"

namespace fe {

struct CompareKeyValue {
  bool operator() (const std::shared_ptr<FusionRuleNode> &key1, const std::shared_ptr<FusionRuleNode> &key2) const {
    if ((key1 == nullptr) || (key2 == nullptr)) {
      return false;
    }
    return (key1->GetNodeName() < key2->GetNodeName());
  }
};
using FusionRuleNodeMapping = std::map<FusionRuleNodePtr, ge::NodePtr, CompareKeyValue>;
// Match result.
struct GraphMatchResult {
  // Matched outer inputs.
  std::map<FusionRuleAnchorPtr, ge::AnchorPtr> outer_inputs;
  std::map<FusionRuleAnchorPtr, ge::AnchorPtr> origin_outer_inputs;
  std::map<FusionRuleAnchorPtr, std::set<ge::AnchorPtr>> outer_inputs_set;

  // Matched original nodes.
  FusionRuleNodeMapping origin_nodes;
  std::unordered_set<ge::NodePtr> origin_nodes_set;

  // Matched outer outputs.
  std::map<FusionRuleAnchorPtr, std::set<ge::AnchorPtr>> outer_outputs;
  std::map<FusionRuleAnchorPtr, std::set<ge::AnchorPtr>> origin_outer_outputs;

  /** Fusion result graph nodes */
  std::vector<ge::NodePtr> fusion_nodes;
  bool valid_flag = true;
};

// Use a rule to match subgraphs in the compute graph.
class GraphMatcher {
 public:
  GraphMatcher();
  virtual ~GraphMatcher();

  // Use rule to match graph and save the results to match_results.
  Status Match(const FusionRulePattern &rule, const ge::ComputeGraph &graph,
               std::vector<GraphMatchResult> &match_results);

 private:
  using CandidateList = std::list<std::pair<FusionRuleNodePtr, ge::NodePtr>>;

  GraphMatcher(const GraphMatcher &) = delete;
  GraphMatcher &operator=(const GraphMatcher &) = delete;

  FusionRuleNodePtr GetFirstOutputRuleNode() const;

  bool MatchFromOutput(FusionRuleNodePtr output_rule_node, ge::NodePtr output_graph_node,
                       GraphMatchResult &match_result);

  bool MatchOriginNode(FusionRuleNodePtr rule_node, ge::NodePtr graph_node, GraphMatchResult &match_result) const;
  bool MatchOriginNodeInputs(FusionRuleNodePtr rule_node, ge::NodePtr graph_node, GraphMatchResult &match_result);
  bool MatchOriginNodeOutputs(FusionRuleNodePtr rule_node, ge::NodePtr graph_node, GraphMatchResult &match_result);
  bool MatchOutputsForOuterInput(FusionRuleNodePtr rule_node, ge::NodePtr graph_node, GraphMatchResult &match_result);
  bool MatchOriginNodeOutPeers(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                               GraphMatchResult &match_result);
  bool MatchOriginNodeInPeers(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                              GraphMatchResult &match_result);
  bool MatchOuterInputOutPeers(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                               GraphMatchResult &match_result);
  bool MatchInOutPeers(const std::vector<FusionRuleAnchorPtr> &peer_rule_anchors,
                       const ge::Anchor::Vistor<ge::AnchorPtr> &peer_graph_anchors, ge::AnchorPtr graph_origin_anchor,
                       GraphMatchResult &match_result);
  bool MatchPeersWithOuterOutput(const std::vector<FusionRuleAnchorPtr> &peer_rule_anchors,
                                 const ge::Anchor::Vistor<ge::AnchorPtr> &peer_graph_data_anchors,
                                 GraphMatchResult &match_result);
  bool MatchPeer(FusionRuleAnchorPtr peer_rule_anchor, ge::AnchorPtr peer_graph_anchor,
                 ge::AnchorPtr graph_origin_anchor, GraphMatchResult &match_result);

  bool IsNodeTypeMatch(FusionRuleNodePtr rule_node, ge::NodePtr graph_node) const;
  bool IsOuterOutputMatch(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                          const GraphMatchResult &match_result) const;

  bool IsInOuterInputs(FusionRuleNodePtr rule_node) const;
  bool IsInOuterOutputs(FusionRuleNodePtr rule_node) const;
  bool IsInOriginGraph(FusionRuleNodePtr rule_node) const;
  bool IsInFusionGraph(FusionRuleNodePtr rule_node) const;

  bool IsOuterInput(FusionRuleAnchorPtr anchor) const;
  bool IsOuterOutput(FusionRuleAnchorPtr anchor) const;
  bool IsOriginAnchor(FusionRuleAnchorPtr anchor) const;
  bool IsCtrlAnchorExistMatched(FusionRuleAnchorPtr rule_ctrl_anchor, ge::ControlAnchorPtr graph_ctrl_anchor) const;
  bool CheckOuterInputCtrlAnchor(FusionRuleAnchorPtr rule_ctrl_anchor, ge::ControlAnchorPtr graph_ctrl_anchor) const;
  bool HasOuterOutput(const vector<FusionRuleAnchorPtr> &anchors) const;
  bool HasPeers(const ge::Node::Vistor<ge::OutDataAnchorPtr> &graph_anchors, size_t start_idx) const;

  std::vector<FusionRuleAnchorPtr> GetPeerOriginAnchors(FusionRuleAnchorPtr rule_anchor) const;
  std::vector<FusionRuleAnchorPtr> GetPeerAnchorsNotInFusionGraph(FusionRuleAnchorPtr rule_anchor) const;

  void SeparateOriginAndOuterOutputAnchors(const ge::Anchor::Vistor<ge::AnchorPtr> &peer_graph_anchors,
                                           const std::vector<FusionRuleAnchorPtr> &peer_rule_origin_anchors,
                                           std::map<FusionRuleAnchorPtr, ge::AnchorPtr> &peer_origin_anchor_map,
                                           std::vector<ge::AnchorPtr> &peer_graph_outer_output_anchors) const;
  void AddOuterOutputToResult(FusionRuleAnchorPtr rule_anchor, ge::AnchorPtr graph_anchor,
                              GraphMatchResult &result) const;
  bool RemoveRedundantOuterOutputs(GraphMatchResult &match_result) const;

  bool VerifyMatchResult(const GraphMatchResult &match_result) const;

  Status GetRuleOuterInOutAnchorCount();

  void DumpMatcheResults(const std::vector<GraphMatchResult> &match_results) const;
  void DumpMatchedOuterInputs(const GraphMatchResult &result) const;
  void DumpMatchedOuterOutputs(const GraphMatchResult &result) const;
  void DumpMatchedOriginNodes(const GraphMatchResult &result) const;

 private:
  const FusionRulePattern *rule_{nullptr};
  size_t rule_outer_input_count_{0};
  size_t rule_outer_output_count_{0};
  std::set<ge::NodePtr> prev_matched_origin_nodes_;

  CandidateList candidate_nodes_;
  std::map<FusionRuleAnchorPtr, int> outer_anchor_idxs_;
};

}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_GRAPH_MATCHER_H_
