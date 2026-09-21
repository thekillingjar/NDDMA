/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stack>
#include <queue>
#include <utility>
#include "ge/fusion/pattern_matcher.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "common/checker.h"
#include "node_matcher.h"
#include "base/common/plugin/ge_make_unique_util.h"
#include "fusion_utils.h"

namespace ge {
namespace fusion {
using OpType2Nodes = std::unordered_map<std::string, std::vector<Node *>>;
using PDataAnchor2TDataAnchor = std::pair<OutDataAnchorPtr, OutDataAnchorPtr>;
namespace {
bool IsGraphData(const NodePtr &node) {
  return strcmp(node->GetTypePtr(), DATA) == 0;
}

std::string GetOpType(const NodePtr &node) {
  if (ConstantUtils::IsConstant(node)) {
    return CONSTANT;
  }
  return node->GetTypePtr();
}

bool IsGraphNetOutput(const NodePtr &node) {
  return strcmp(node->GetTypePtr(), NETOUTPUT) == 0;
}

bool IsControlEdgeExist(const NodePtr &node, std::stringstream &detail) {
  if ((node->GetInControlNodesSize() != 0U || node->GetOutControlNodesSize() != 0U)) {
    detail << "Node[" << node->GetNamePtr() << "] has control edge";
    return true;
  }
  return false;
}

bool IsDynamicIONode(const NodePtr &node, std::stringstream &detail) {
  const auto ir_outputs = node->GetOpDesc()->GetIrOutputs();
  for (const auto &output_name_2_output_type : ir_outputs) {
    if (output_name_2_output_type.second == IrOutputType::kIrOutputDynamic) {
      detail << "Node[" << node->GetNamePtr() << "] output name:[" << output_name_2_output_type.first
             << "] is dynamic output.";
      return true;
    }
  }

  const auto ir_inputs = node->GetOpDesc()->GetIrInputs();
  for (const auto &input_name_2_input_type : ir_inputs) {
    if (input_name_2_input_type.second == IrInputType::kIrInputDynamic) {
      detail << "Node[" << node->GetNamePtr() << "] input name:[" << input_name_2_input_type.first
             << "] is dynamic input.";
      return true;
    }
  }
  return false;
}

Status ValidatePattern(const ComputeGraphPtr &pattern_graph) {
  std::stringstream invalid_reason;
  invalid_reason << "Pattern[" << pattern_graph->GetName() << "] is invalid. Reason: ";
  if (!pattern_graph->GetAllSubgraphs().empty()) {
    invalid_reason << "It has subgraph.";
    GELOGE(FAILED, "%s", invalid_reason.str().c_str());
    return FAILED;
  }

  for (const auto &node : pattern_graph->GetDirectNode()) {
    if (IsGraphData(node) || IsGraphNetOutput(node)) {
      continue;
    }
    GE_ASSERT_TRUE(!IsControlEdgeExist(node, invalid_reason), invalid_reason.str().c_str());
    GE_ASSERT_TRUE(!IsDynamicIONode(node, invalid_reason), invalid_reason.str().c_str());
  }
  GE_ASSERT_SUCCESS(pattern_graph->TopologicalSorting(),
                    "Failed to topo sort on pattern[%s],please check if there is cycle in pattern graph.",
                    pattern_graph->GetName().c_str());
  return SUCCESS;
}

NodeIo ToNodeIo(const OutDataAnchorPtr &out_data_anchor) {
  GNode owner_node = NodeAdapter::Node2GNode(out_data_anchor->GetOwnerNode());
  return {owner_node, out_data_anchor->GetIdx()};
}

// pattern output匹配到的node称为匹配坐标MatchCoordinate
struct MatchCoordinate {
 public:
  bool IsValid() const {
    return node->GetOwnerComputeGraph() != nullptr;
  }
  MatchCoordinate(const NodePtr &target_node, size_t pattern_output_index)
      : node(target_node), pattern_output_idx(pattern_output_index) {}
  NodePtr node;
  size_t pattern_output_idx; // 对应在pattern中的第几个输出
};
using MatchCoordinatePtr = std::shared_ptr<MatchCoordinate>;

struct MatchCoordinateSeq {
 public:
  explicit MatchCoordinateSeq() : coordinates_sets_(), coordinates_(), sliding_cursor_(-1) {}

  explicit MatchCoordinateSeq(const std::vector<MatchCoordinatePtr> &coordinates)
      : coordinates_(coordinates), sliding_cursor_(-1) {
    for (const auto &cor : coordinates) {
      coordinates_sets_.emplace(cor->node);
    }
  }

  void ResetSlidingCursor() {
    sliding_cursor_ = -1;
  }

  // should check whether has_next outside
  MatchCoordinatePtr NextMatchCoordinate() {
    return HasNext() ? coordinates_[++sliding_cursor_] : nullptr;
  }

  void AppendCoordinate(const MatchCoordinatePtr &cor) {
    if (coordinates_sets_.emplace(cor->node).second) {
      coordinates_.emplace_back(cor);
    }
  }

 private:
  bool HasNext() const {
    return (sliding_cursor_ + 1) < static_cast<int64_t>(coordinates_.size());
  }

  std::unordered_set<NodePtr> coordinates_sets_;
  std::vector<MatchCoordinatePtr> coordinates_;
  int64_t sliding_cursor_ = -1L;
};

std::vector<OutDataAnchorPtr> GetAllOutDataAnchors(const ComputeGraphPtr &compute_graph) {
  std::vector<OutDataAnchorPtr> output_anchors;
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (!IsGraphNetOutput(node)) {
      continue;
    }
    for (const auto &in_data_anchor : node->GetAllInDataAnchorsPtr()) {
      if (in_data_anchor == nullptr) {
        continue;
      }
      output_anchors.emplace_back(in_data_anchor->GetPeerOutAnchor());
    }
  }
  return output_anchors;
}

void CollectDirectOpTypeToNodes(const ge::ComputeGraphPtr &compute_graph, OpType2Nodes &optype_2_nodes) {
  for (const auto node : compute_graph->GetDirectNodePtr()) {
    optype_2_nodes[GetOpType(node->shared_from_this())].emplace_back(node);
  }
}

/*
 * 若返回false表示其输入node不匹配
 */
bool ElectQualifiedInputsCandidate(const NodePtr &p_node, const NodePtr &t_node,
                                   std::queue<PDataAnchor2TDataAnchor> &anchor_pairs_queue) {
  const auto p_input_nodes_2_out_anchor = p_node->GetInDataNodesAndAnchors();
  const auto t_input_nodes_2_out_anchor = t_node->GetInDataNodesAndAnchors();
  // todo need check in data sequence is fine
  // todo data may match more than one node
  if (p_input_nodes_2_out_anchor.size() != t_input_nodes_2_out_anchor.size()) {
    GELOGD("[MATCH][SKIP]p node [%s] input nodes size[%zu], t node [%s] input nodes size[%zu]", p_node->GetTypePtr(),
           p_node->GetInDataNodesSize(), t_node->GetNamePtr(), t_node->GetInDataNodesSize());
    return false;
  }
  for (size_t i = 0U; i < p_input_nodes_2_out_anchor.size(); ++i) {
    const auto p_peer_in_out_anchor = p_input_nodes_2_out_anchor.at(i).second;
    const auto t_peer_in_out_anchor = t_input_nodes_2_out_anchor.at(i).second;
    if (!IsGraphData(p_peer_in_out_anchor->GetOwnerNode())) {
      if (p_peer_in_out_anchor->GetIdx() != t_peer_in_out_anchor->GetIdx()) {
        return false;
      }
    }
    anchor_pairs_queue.emplace(p_peer_in_out_anchor, t_peer_in_out_anchor);
  }
  return true;
}

bool IsValidMatch(const MatchResult &match) {
  InnerSubgraphBoundary inner_boundary;
  std::string boundary_invalid_reason;
  if (inner_boundary.Init(*(match.ToSubgraphBoundary()), boundary_invalid_reason) != SUCCESS) {
    GELOGW("%s", boundary_invalid_reason.c_str());
    return false;
  }
  return true;
}

void FetchNextMainCoordinate(MatchCoordinateSeq &main_cor_seq, std::stack<MatchCoordinatePtr> &candidates) {
  while (!candidates.empty()) {
    candidates.pop();
  }
  auto main_coor =  main_cor_seq.NextMatchCoordinate();
  if (main_coor != nullptr) {
    candidates.emplace(main_coor);
  }
}
} // namespace

class PatternMatcherImpl {
 public:
  PatternMatcherImpl() = delete;
  explicit PatternMatcherImpl(std::unique_ptr<Pattern> pattern, ComputeGraphPtr target_graph)
      : config_(PatternMatcherConfigBuilder().Build()), pattern_(std::move(pattern)), target_graph_(std::move(target_graph)) {}
  PatternMatcherImpl(std::unique_ptr<Pattern> pattern, ComputeGraphPtr target_graph, std::unique_ptr<PatternMatcherConfig> matcher_config)
      : config_(std::move(matcher_config)),
        pattern_(std::move(pattern)),
        target_graph_(std::move(target_graph)) {}

  /**
   * init pattern matcher
   * 1. check pattern valid
   * 2. get all matched out nodes in target graph
   * @return
   */
  Status Initialize() {
    if (!has_inited) {
      GE_ASSERT_SUCCESS(InitNodeMatchers());
      auto pattern_graph = pattern_->GetGraph();
      auto pattern_compute_graph = GraphUtilsEx::GetComputeGraph(pattern_graph);
      GE_ASSERT_SUCCESS(ValidatePattern(pattern_compute_graph));

      pattern_out_anchors_ = GetAllOutDataAnchors(pattern_compute_graph);
      if (pattern_out_anchors_.empty()) {
        GELOGW("Pattern graph %s has no output which is invalid pattern graph.",
               pattern_compute_graph->GetName().c_str());
        return FAILED;
      }
    }
    // coordinate需要更新，需要根据目标图实时刷新
    GE_ASSERT_SUCCESS(UpdateAllMatchCoordinates(target_graph_, pattern_out_anchors_, idx_2_coordinate_seqs_));
    GE_ASSERT_TRUE(idx_2_coordinate_seqs_.size() == pattern_out_anchors_.size());

    has_inited = true;
    return SUCCESS;
  }
  /**
   * 匹配算法
   * 以多输出的pattern graph为例
   *     A               A'         A''
   *    / \             / \        / \
   *   B  C            B'  C'     B''  C''
   * 以第0个输出的matched nodes {B', B''}为匹配序列
   *
   *
   * @param target_graph
   * @return
   */
  std::unique_ptr<MatchResult> MatchNext() {
    GE_ASSERT_SUCCESS(Initialize(), "Failed to init pattern matcher.");
    // 算法以第0个输出匹配到nodes为匹配序列，第0个输出对应的匹配坐标称为 主要匹配坐标 MainMatchCoordinate
    // idx_2_coordinate_seqs_ at least has 1 element, checked in Initialize
    auto &main_coordinate_seq = idx_2_coordinate_seqs_[0];
    const auto &cur_main_coordinate = main_coordinate_seq.NextMatchCoordinate();
    if (cur_main_coordinate == nullptr) {
      return nullptr;
    }
    auto match = MakeUnique<MatchResult>(pattern_.get());
    GE_ASSERT_NOTNULL(match);

    std::stack<MatchCoordinatePtr> candidates;
    candidates.emplace(cur_main_coordinate);
    GELOGD("[MATCH]Start match main coordinate [%s][%s]", cur_main_coordinate->node->GetNamePtr(),
           cur_main_coordinate->node->GetTypePtr());
    const size_t p_output_count = pattern_out_anchors_.size();
    while (!candidates.empty()) {
      const auto &match_coordinate = candidates.top();
      const auto curr_out_idx = match_coordinate->pattern_output_idx;  // 当前pattern图的图输出id
      GE_ASSERT_TRUE(curr_out_idx < pattern_out_anchors_.size());
      if (match_coordinate->IsValid() &&
          MatchBranchByOutTensor(match_coordinate->node, pattern_out_anchors_[curr_out_idx], *match)) {
        if (curr_out_idx >= (p_output_count - 1)) {
          // 最后一个pattern output匹配成功
          // found match， 非main_coordinate_seq的cursor置0，准备进入下一个图实例匹配
          // 校验stack里node个数符合pattern中输出个数
          GE_ASSERT_TRUE(candidates.size() == p_output_count);
          if (IsValidMatch(*match)) {
            GELOGD("[MATCH][FOUND] %s", match->ToAscendString().GetString());
            return match;
          }
          // 否则走下一个main候选坐标的匹配
          match = MakeUnique<MatchResult>(pattern_.get());
          GE_ASSERT_NOTNULL(match);
          FetchNextMainCoordinate(main_coordinate_seq, candidates);
          continue;
        } else {
          // 当前分支匹配成功，还有下一个pattern output待匹配
          const size_t next_out_idx = curr_out_idx + 1;
          const auto next_out_coordinate = idx_2_coordinate_seqs_[next_out_idx].NextMatchCoordinate();
          if (next_out_coordinate == nullptr) {
            candidates.pop();  // 当前pattern output的coordinate出栈，下一个coordinate候选者入栈
            continue;
          }
          candidates.emplace(next_out_coordinate);
          continue;
        }
      }
      candidates.pop();  // 当前图输出的coordinate出栈，下一个coordinate候选者入栈
      auto next_coordinate = idx_2_coordinate_seqs_[curr_out_idx].NextMatchCoordinate();
      if (next_coordinate == nullptr) {
        return nullptr;
      }
      // 若当前match idx没匹配上，需要在当前branch序列继续向下匹配
      candidates.emplace(next_coordinate);
    }
    return nullptr;
  }

 private:
  bool IsNodeMatchWith(const NodePtr &p_node, const NodePtr &t_node) const {
    return GetNodeMatcher(p_node)->IsMatch(p_node, t_node);
  }
  bool MatchBranchByOutTensor(const NodePtr &t_out_node, const OutDataAnchorPtr &p_out_anchor, MatchResult &match_ret) const;
  Status GetMatchCoordinatesByIdx(const ge::ComputeGraphPtr &t_graph, const NodePtr &p_output_node,
                                  const size_t p_output_idx, MatchCoordinateSeq &cor_seq) const;

  Status UpdateAllMatchCoordinates(const ComputeGraphPtr &t_graph,
                                   const std::vector<OutDataAnchorPtr> &p_idx_2_out_anchors,
                                   std::vector<MatchCoordinateSeq> &matched_cor_seq) const;
  Status InitNodeMatchers();
  const std::unique_ptr<NodeMatcher> &GetNodeMatcher(const NodePtr &node) const;
  bool has_inited = false;
  std::unique_ptr<PatternMatcherConfig> config_;
  std::unique_ptr<Pattern> pattern_;
  ComputeGraphPtr target_graph_;

  std::unique_ptr<NodeMatcher> data_matcher_{nullptr};
  std::unique_ptr<NodeMatcher> const_matcher_{nullptr};
  std::unique_ptr<NodeMatcher> normal_matcher_{nullptr};

  std::vector<OutDataAnchorPtr> pattern_out_anchors_;
  // pattern输出索引在target graph中的匹配到的坐标队列
  std::vector<MatchCoordinateSeq> idx_2_coordinate_seqs_;
};

Status PatternMatcherImpl::UpdateAllMatchCoordinates(const ComputeGraphPtr &t_graph,
                                                     const std::vector<OutDataAnchorPtr> &p_idx_2_out_anchors,
                                                     std::vector<MatchCoordinateSeq> &matched_cor_seq) const {
  // std::vector<MatchCoordinateSeq> t_idx_2_out_nodes;
  for (size_t i = 0U; i < p_idx_2_out_anchors.size(); ++i) {
    auto idx_of_output = i;
    auto &p_out_data_anchor = p_idx_2_out_anchors[i];
    auto p_out_node = p_out_data_anchor->GetOwnerNode();
    if (matched_cor_seq.size() > i) {
      // 已存在主坐标序列，更新
      GE_ASSERT_SUCCESS(GetMatchCoordinatesByIdx(t_graph, p_out_node, idx_of_output, matched_cor_seq[i]));  // 可以挪到matchbranch里面做
    } else {
      MatchCoordinateSeq cor_seq;
      GE_ASSERT_SUCCESS(GetMatchCoordinatesByIdx(t_graph, p_out_node, idx_of_output, cor_seq));  // 可以挪到matchbranch里面做
      matched_cor_seq.emplace_back(cor_seq);
    }
  }
  return SUCCESS;
}

Status PatternMatcherImpl::GetMatchCoordinatesByIdx(const ge::ComputeGraphPtr &t_graph, const NodePtr &p_output_node,
                                                    const size_t p_output_idx, MatchCoordinateSeq &cor_seq) const {
  OpType2Nodes t_optype_2_nodes;
  CollectDirectOpTypeToNodes(t_graph, t_optype_2_nodes);

  GE_ASSERT_NOTNULL(p_output_node);
  // todo add fuzzy node type
  auto op_type = GetOpType(p_output_node);
  auto iter = t_optype_2_nodes.find(op_type);
  if (iter == t_optype_2_nodes.cend()) {
    return SUCCESS;
  }
  for (const auto &node : iter->second) {
    if (IsNodeMatchWith(p_output_node, node->shared_from_this())) {
      MatchCoordinatePtr coordinate = std::make_shared<MatchCoordinate>(node->shared_from_this(), p_output_idx);
      GE_ASSERT_NOTNULL(coordinate);
      cor_seq.AppendCoordinate(coordinate);
      GELOGD("Got coordinate [%zu][%s][%s]", p_output_idx, node->GetTypePtr(), node->GetNamePtr());
    }
  }
  return SUCCESS;
}

bool PatternMatcherImpl::MatchBranchByOutTensor(const NodePtr &t_out_node, const OutDataAnchorPtr &p_out_anchor,
                                                MatchResult &match_ret) const {
  // 本函数为回溯算法中的一次尝试，因此需要将match_ret备份
  MatchResult tmp_match = match_ret;
  // 这里使用p out anchor 的index来做索引
  auto t_out_anchor = t_out_node->GetOutDataAnchor(p_out_anchor->GetIdx());
  if (t_out_anchor == nullptr) {
    return false;
  }
  std::queue<PDataAnchor2TDataAnchor> node_pairs_queue;
  node_pairs_queue.emplace(p_out_anchor, t_out_anchor);
  while (!node_pairs_queue.empty()) {
    const auto &p_anchor_2_t_anchor = node_pairs_queue.front();
    node_pairs_queue.pop();
    const auto p_node = p_anchor_2_t_anchor.first->GetOwnerNode();
    const auto t_node = p_anchor_2_t_anchor.second->GetOwnerNode();
    // 若pnode已被匹配过，说明pnode为图中汇点，若与当前匹配不一致，说明本次逆序匹配的match与之前的output tensor的逆序匹配match不是同一个
    GNode matched_node;
    if (tmp_match.GetMatchedNode(NodeAdapter::Node2GNode(p_node), matched_node) == SUCCESS) {
      if (NodeAdapter::GNode2Node(matched_node) != t_node) {
        return false;
      }
    }

    if (!IsNodeMatchWith(p_node, t_node)) {
      GELOGD("[MATCH][MISS] pattern node [%s][%s], target node [%s][%s].", p_node->GetNamePtr(), p_node->GetTypePtr(),
             t_node->GetNamePtr(), t_node->GetTypePtr());
      return false;
    }
    const NodeIo pnode_out_tensor_anchor = ToNodeIo(p_anchor_2_t_anchor.first);
    const NodeIo tnode_out_tensor_anchor = ToNodeIo(p_anchor_2_t_anchor.second);
    GE_ASSERT_SUCCESS(tmp_match.AppendNodeMatchPair(pnode_out_tensor_anchor, tnode_out_tensor_anchor));

    if (IsGraphData(p_node)) {
      GELOGD("[MATCH] pattern node [%s][%s] is pattern input", p_node->GetNamePtr(), p_node->GetTypePtr());
      continue;
    }
    if (!ElectQualifiedInputsCandidate(p_node, t_node, node_pairs_queue)) {
      return false;
    }
  }
  match_ret = tmp_match;
  return true;
}

Status PatternMatcherImpl::InitNodeMatchers() {
  data_matcher_ = MakeUnique<DataMatcher>();
  GE_ASSERT_NOTNULL(data_matcher_);
  const_matcher_ = MakeUnique<ConstantMatcher>(config_->IsEnableConstValueMatch(), false);
  GE_ASSERT_NOTNULL(const_matcher_);
  normal_matcher_ = MakeUnique<NormalNodeMatcher>(config_->IsEnableIrAttrMatch());
  GE_ASSERT_NOTNULL(normal_matcher_);
  return SUCCESS;
}
const std::unique_ptr<NodeMatcher> &PatternMatcherImpl::GetNodeMatcher(const NodePtr &node) const {
  if (strcmp(node->GetTypePtr(), DATA) == 0) {
    return data_matcher_;
  } else if (ConstantUtils::IsRealConst(node->GetOpDesc())) {
    return const_matcher_;
  } else {
    return normal_matcher_;
  }
}

PatternMatcher::PatternMatcher(std::unique_ptr<Pattern> pattern, const GraphPtr &target_graph) {
  auto compute_target_graph = GraphUtilsEx::GetComputeGraph(*target_graph);
  impl_ = MakeUnique<PatternMatcherImpl>(std::move(pattern), compute_target_graph);
}

PatternMatcher::PatternMatcher(std::unique_ptr<Pattern> pattern, const GraphPtr &target_graph,
                               std::unique_ptr<PatternMatcherConfig> matcher_config) {
  auto compute_target_graph = GraphUtilsEx::GetComputeGraph(*target_graph);
  impl_ = MakeUnique<PatternMatcherImpl>(std::move(pattern), compute_target_graph, std::move(matcher_config));
}

PatternMatcher::~PatternMatcher() = default;

std::unique_ptr<MatchResult> PatternMatcher::MatchNext() {
  return impl_ == nullptr ? nullptr : impl_->MatchNext();
}
}  // namespace fusion
} // namespace ge