/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ge/fusion/match_result.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/checker.h"

#include "base/common/plugin/ge_make_unique_util.h"
#include "framework/common/types.h"

namespace ge {
namespace fusion {
class MatchResultImpl {
 public:
  explicit MatchResultImpl(const Pattern *const pattern)
      : pattern_(const_cast<Pattern *>(pattern)), captured_tensors_(pattern) {}
  MatchResultImpl& operator=(const MatchResultImpl&) = delete;
  MatchResultImpl(const MatchResultImpl &other) : pattern_(other.pattern_), captured_tensors_(other.pattern_) {
    if (&other != this) {
      this->pattern_node_2_matched_node_ = other.pattern_node_2_matched_node_;
      this->in_idx_2_out_data_anchor_ = other.in_idx_2_out_data_anchor_;
      this->out_idx_2_out_data_anchor_ = other.out_idx_2_out_data_anchor_;
      this->matched_nodes_ = other.matched_nodes_;
      this->pattern_ = other.pattern_;
      this->pattern_outputs_ = other.pattern_outputs_;
      this->captured_tensors_ = other.captured_tensors_;
    }
  }

  NodePtr GetMatchedNode(const NodePtr &pattern_node) const {
    const auto iter = pattern_node_2_matched_node_.find(pattern_node);
    if (iter != pattern_node_2_matched_node_.cend()) {
      return iter->second;
    }
    return nullptr;
  }

  std::vector<NodePtr> GetMatchedNodes() const {
    std::vector<NodePtr> all_target_nodes_except_io;
    for (const auto &p_2_t : pattern_node_2_matched_node_) {
      if (OpTypeUtils::IsDataNode(p_2_t.first->GetTypePtr())) {
        continue;
      }
      all_target_nodes_except_io.emplace_back(p_2_t.second);
    }
    std::sort(
        all_target_nodes_except_io.begin(), all_target_nodes_except_io.end(),
        [](const NodePtr &a, const NodePtr &b) -> bool { return a->GetOpDesc()->GetId() < b->GetOpDesc()->GetId(); });
    return all_target_nodes_except_io;
  }

  Status AppendNodeMatchPair(const NodeIo &p_out_anchor, const NodeIo &t_out_anchor) {
    if (pattern_outputs_.IsEmpty()) {
      auto pattern_graph = pattern_->GetGraph();
      GE_ASSERT_SUCCESS(pattern_outputs_.CollectPatternOutput(GraphUtilsEx::GetComputeGraph(pattern_graph)));
    }

    const auto p_node = NodeAdapter::GNode2Node(p_out_anchor.node);
    const auto t_node = NodeAdapter::GNode2Node(t_out_anchor.node);
    pattern_node_2_matched_node_[p_node] = t_node;
    matched_nodes_.emplace(t_node);
    GELOGD("[MATCH][NODE]%s(%s), Pattern node:%s(%s).", t_node->GetNamePtr(), t_node->GetTypePtr(),
           p_node->GetNamePtr(), p_node->GetTypePtr());
    if (OpTypeUtils::IsDataNode(p_node->GetTypePtr())) {
      int32_t data_index = 0;
      GE_ASSERT_TRUE(AttrUtils::GetInt(p_node->GetOpDesc(), ATTR_NAME_INDEX, data_index));
      GELOGD("[MATCH][NODE] %s(%s) as %d input of match ret.", t_node->GetNamePtr(), t_node->GetTypePtr(), data_index);
      in_idx_2_out_data_anchor_[data_index] = t_out_anchor;
    } else if (pattern_outputs_.IsPatternOutput(p_node->GetOutDataAnchor(p_out_anchor.index))) {
      size_t pattern_out_idx = UINT64_MAX;
      GE_ASSERT_SUCCESS(pattern_outputs_.GetOutputIdx(p_node->GetOutDataAnchor(p_out_anchor.index), pattern_out_idx));
      out_idx_2_out_data_anchor_[pattern_out_idx] = t_out_anchor;
    }

    GE_ASSERT_SUCCESS(captured_tensors_.TryCaptureMatchedTensor(p_out_anchor, t_out_anchor));
    return SUCCESS;
  }

  std::string ToString() const {
    std::stringstream ss;
    AscendString pattern_name;
    pattern_->GetGraph().GetName(pattern_name);
    ss << "[PatternName:" << pattern_name.GetString() << "]";
    ss << "[MatchNodesPair]{";
    for (const auto &pnode_2_tnode : pattern_node_2_matched_node_) {
      ss << "{" << pnode_2_tnode.first->GetTypePtr() << ":" << pnode_2_tnode.second->GetNamePtr() << "}";
    }
    ss << "}";
    return ss.str();
  }

  [[nodiscard]] std::unique_ptr<SubgraphBoundary> ToSubgraphBoundary() const {
    std::unique_ptr<SubgraphBoundary> boundary = MakeUnique<SubgraphBoundary>();
    GE_ASSERT_NOTNULL(boundary);
    for (const auto &idx_2_out_anchor : in_idx_2_out_data_anchor_) {
      SubgraphInput subgraph_input;
      const auto producer_output = idx_2_out_anchor.second;
      auto out_data_anchor = NodeAdapter::GNode2Node(producer_output.node)->GetOutDataAnchor(producer_output.index);
      GE_ASSERT_NOTNULL(out_data_anchor);
      // todo 确认一下匹配的时候，单输出多引用场景是否有问题
      // todo 考虑直接把boundary作为match result的成员
      for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        const auto peer_in_node = peer_in_data_anchor->GetOwnerNode();
        if (matched_nodes_.count(peer_in_node) > 0) {
          GE_ASSERT_SUCCESS(
              subgraph_input.AddInput({NodeAdapter::Node2GNode(peer_in_node), peer_in_data_anchor->GetIdx()}));
        }
      }
      GE_ASSERT_TRUE(!subgraph_input.GetAllInputs().empty());
      boundary->AddInput(idx_2_out_anchor.first, std::move(subgraph_input));
    }

    for (const auto &idx_2_out_anchor : out_idx_2_out_data_anchor_) {
      SubgraphOutput subgraph_output(idx_2_out_anchor.second);
      const auto subgraph_output_idx = idx_2_out_anchor.first;
      GE_ASSERT_SUCCESS(boundary->AddOutput(subgraph_output_idx, std::move(subgraph_output)));
    }
    return boundary;
  }

  Status GetCapturedTensor(size_t capture_idx, NodeIo &node_output) const {
    return captured_tensors_.GetCapturedTensor(capture_idx, node_output);
  }

  const Graph &GetPatternGraph() const {
    return pattern_->GetGraph();
  }

 private:
  struct PatternOutputs {
   public:
    bool IsPatternOutput(const NodePtr &node) const {
      return output_nodes_.count(node) > 0;
    }
    bool IsPatternOutput(const OutDataAnchorPtr &out_anchor) const {
      auto iter = output_anchors_.find(out_anchor->GetOwnerNode());
      if (iter == output_anchors_.end()) {
        return false;
      }
      return (iter->second.count(out_anchor->GetIdx()) > 0);
    }

    bool IsEmpty() const {
      return output_anchors_.empty();
    }

    Status GetOutputIdx(const OutDataAnchorPtr &output_anchor, size_t &output_idx) {
      auto iter = out_data_anchor_2_out_idx_.find(output_anchor);
      if (iter == out_data_anchor_2_out_idx_.end()) {
        auto owner_node = output_anchor->GetOwnerNode();
        GE_ASSERT_NOTNULL(owner_node);
        GELOGE(FAILED, "Failed to find pattern output idx from Node[%s][%s] output idx[%d]", owner_node->GetNamePtr(),
               owner_node->GetTypePtr(), output_anchor->GetIdx());
        return FAILED;
      }
      output_idx = iter->second;
      return SUCCESS;
    }

    Status CollectPatternOutput(const ComputeGraphPtr &pattern_graph) {
      auto netoutput = pattern_graph->FindFirstNodeMatchType(NETOUTPUT);
      GE_ASSERT_NOTNULL(netoutput);
      for (size_t output_idx = 0U; output_idx < netoutput->GetAllInDataAnchorsSize(); ++output_idx) {
        auto peer_out_anchor= netoutput->GetInDataAnchor(output_idx)->GetPeerOutAnchor();
        out_data_anchor_2_out_idx_[peer_out_anchor] = output_idx;
        output_nodes_.emplace(peer_out_anchor->GetOwnerNode());
        output_anchors_[peer_out_anchor->GetOwnerNode()].emplace(peer_out_anchor->GetIdx());
      }
      return SUCCESS;
    }
   private:
    std::set<NodePtr> output_nodes_;
    std::map<NodePtr, std::set<size_t>> output_anchors_;
    std::map<OutDataAnchorPtr, size_t> out_data_anchor_2_out_idx_;
  };

  struct CapturedTensors {
   public:
    // init pattern captured tensor
    explicit CapturedTensors(const Pattern *const pattern) {
      std::vector<NodeIo> captured_tensors;
      pattern->GetCapturedTensors(captured_tensors);
      size_t captured_index = 0U;
      for (const auto &node_output : captured_tensors) {
        const auto node = NodeAdapter::GNode2Node(node_output.node);
        OutDataAnchorPtr out_data_anchor = nullptr;
        if (node != nullptr) {
          out_data_anchor = node->GetOutDataAnchor(node_output.index);
          pattern_captured_set_.emplace(out_data_anchor);
          pattern_captured_tensor_2_idx_[out_data_anchor] = captured_index;
        }
        captured_index++;
      }
      matched_captured_tensor_.resize(captured_index + 1);
    }

    Status TryCaptureMatchedTensor(const NodeIo &p_node_output, const NodeIo &matched_node_output) {
      const auto pattern_node = NodeAdapter::GNode2Node(p_node_output.node);
      GE_ASSERT_NOTNULL(pattern_node);
      const auto p_out_data_anchor = pattern_node->GetOutDataAnchor(p_node_output.index);
      if (pattern_captured_set_.count(p_out_data_anchor) == 0) {
        return SUCCESS;
      }

      const auto matched_node = NodeAdapter::GNode2Node(matched_node_output.node);
      GE_ASSERT_NOTNULL(matched_node);
      const auto matched_out_data_anchor = matched_node->GetOutDataAnchor(matched_node_output.index);
      auto captured_index = pattern_captured_tensor_2_idx_[p_out_data_anchor];
      GE_ASSERT_TRUE(captured_index < matched_captured_tensor_.size());
      matched_captured_tensor_[captured_index] = matched_out_data_anchor;
      GELOGD("[MATCH][CAPTURED]Found captured tensor [%s][%s]output[%d], capture index: %zu.",
             matched_node->GetNamePtr(), matched_node->GetTypePtr(), matched_out_data_anchor->GetIdx(), captured_index);
      return SUCCESS;
    }

    Status GetCapturedTensor(size_t captured_index, NodeIo &node_output) const {
      GE_ASSERT_TRUE(captured_index < matched_captured_tensor_.size());
      auto out_data_anchor = matched_captured_tensor_[captured_index];
      GE_ASSERT_NOTNULL(out_data_anchor);
      node_output = {NodeAdapter::Node2GNode(out_data_anchor->GetOwnerNode()), out_data_anchor->GetIdx()};
      return SUCCESS;
    }
   private:
    std::set<OutDataAnchorPtr> pattern_captured_set_;
    std::map<OutDataAnchorPtr, size_t> pattern_captured_tensor_2_idx_;

    std::vector<OutDataAnchorPtr> matched_captured_tensor_;
  };

  Pattern *pattern_;
  PatternOutputs pattern_outputs_;

  std::set<NodePtr> matched_nodes_;
  std::map<NodePtr, NodePtr> pattern_node_2_matched_node_;
  std::map<size_t, NodeIo> in_idx_2_out_data_anchor_; // match ret的input边界
  std::map<size_t, NodeIo> out_idx_2_out_data_anchor_; // match ret的output边界
  CapturedTensors captured_tensors_;
};

MatchResult::MatchResult(const Pattern *pattern) {
  impl_ = MakeUnique<MatchResultImpl>(pattern);
}

Status MatchResult::GetMatchedNode(const GNode &pattern_node, GNode &matched_node) const {
  GE_ASSERT_NOTNULL(impl_, "Match result is invalid, impl_ is null.");
  auto p_node = NodeAdapter::GNode2Node(pattern_node);
  AscendString pattern_node_name, pattern_node_type;
  pattern_node.GetName(pattern_node_name);
  pattern_node.GetType(pattern_node_type);
  GE_ASSERT_NOTNULL(p_node, "Failed to get node of Gnode %s[%s]", pattern_node_name.GetString(),
                    pattern_node_type.GetString());

  auto m_node_ptr = impl_->GetMatchedNode(p_node);
  if (m_node_ptr == nullptr) {
    // this func may used as check func, so this condition can not considered as an error
    GELOGD("Failed to get matched node of pattern node %s[%s]", pattern_node_name.GetString(),
           pattern_node_type.GetString());
    return FAILED;
  }
  GE_ASSERT_NOTNULL(m_node_ptr, "Failed to get matched node of pattern node %s[%s]", pattern_node_name.GetString(),
                    pattern_node_type.GetString());
  matched_node = NodeAdapter::Node2GNode(m_node_ptr);
  return SUCCESS;
}

std::vector<GNode> MatchResult::GetMatchedNodes() const {
  std::vector<GNode> matched_nodes;
  if (impl_ == nullptr) {
    return matched_nodes;
  }
  for (const auto &node : impl_->GetMatchedNodes()) {
    matched_nodes.emplace_back(NodeAdapter::Node2GNode(node));
  }
  return matched_nodes;
}

Status MatchResult::AppendNodeMatchPair(const NodeIo &pattern_node_out_tensor,
                                        const NodeIo &target_node_out_tensor) {
  return (impl_ != nullptr) ? impl_->AppendNodeMatchPair(pattern_node_out_tensor, target_node_out_tensor) : FAILED;
}

AscendString MatchResult::ToAscendString() const {
  return (impl_ != nullptr) ? AscendString(impl_->ToString().c_str())
                            : "MatchResult is not valid because of it was not fully constructed.";
}

MatchResult::MatchResult(const MatchResult &other) noexcept {
  if (other.impl_ != nullptr) {
    impl_ = MakeUnique<MatchResultImpl>(*other.impl_);
  }
}

MatchResult &MatchResult::operator=(const MatchResult &other) noexcept {
  if (this != &other) {
    impl_ = (other.impl_ != nullptr) ? MakeUnique<MatchResultImpl>(*other.impl_) : nullptr;
  }
  return *this;
}

MatchResult::MatchResult(MatchResult &&other) noexcept : impl_(std::move(other.impl_)) {}

MatchResult &MatchResult::operator=(MatchResult &&other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

std::unique_ptr<SubgraphBoundary> MatchResult::ToSubgraphBoundary() const {
  GE_ASSERT_NOTNULL(impl_);
  return impl_->ToSubgraphBoundary();
}

Status MatchResult::GetCapturedTensor(size_t captured_idx, NodeIo &node_output) const {
  GE_ASSERT_NOTNULL(impl_);
  return impl_->GetCapturedTensor(captured_idx, node_output);
}

const Graph &MatchResult::GetPatternGraph() const {
  if (impl_ == nullptr) {
    static Graph invalid_graph("invalid_pattern_graph");
    return invalid_graph;
  }
  return impl_->GetPatternGraph();
}

MatchResult::~MatchResult() = default;
} // namespace fusion
} // namespace ge