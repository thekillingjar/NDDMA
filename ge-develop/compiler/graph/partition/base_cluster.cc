/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/base_cluster.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "graph_metadef/common/ge_common/util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/partition/base_partitioner.h"

namespace ge {
thread_local size_t ge::BaseCluster::unique_id_ = 0;
using ClusterPtr = std::shared_ptr<BaseCluster>;

namespace {
  NodePtr AddPartitionedCallKeepTopo(const ComputeGraphPtr &compute_graph,
                                     const std::vector<NodePtr> &ori_nodes, const OpDescPtr &partitioned_call_op) {
    auto min_id_node = NodeUtils::GetNodeWithMinimalId(ori_nodes);
    GE_ASSERT_NOTNULL(min_id_node);
    auto ret_nodes = compute_graph->InsertNodes(min_id_node, {partitioned_call_op});
    GE_ASSERT_TRUE(!ret_nodes.empty());
    return ret_nodes[0];
  }
} // namespace
PartitionNodeAttrNameManager &PartitionNodeAttrNameManager::GetInstance() {
  static PartitionNodeAttrNameManager instance;
  return instance;
}

void PartitionNodeAttrNameManager::RegisterNodeAttrName(const PartitionNodeAttrName &attr_name) {
  const std::lock_guard<std::mutex> lock(mutex_);
  attr_array_.emplace_back(attr_name);
}

PartitionNodeAttrRegister::PartitionNodeAttrRegister(const std::string &attr_name, bool is_support_tensor,
                                                     const bool is_need_copy, const SetAttrFn set_fn,
                                                     const GetAttrFn get_fn) noexcept {
  PartitionNodeAttrName partition_attr_name(attr_name, is_support_tensor, is_need_copy, set_fn, get_fn);
  PartitionNodeAttrNameManager::GetInstance().RegisterNodeAttrName(partition_attr_name);
}

std::string BaseCluster::DebugString() const {
  std::stringstream ss;
  if (type_index_ < static_cast<int32_t>(partitioner_->type_index_to_type_.size())) {
    ss << partitioner_->type_index_to_type_[type_index_];
  }

  ss << "[" << id_ << "]" << "[" << type_index_ << "]" << "(size:" << nodes_.size() << ")";
  ss << "(" << min_ << "," << max_ << ")(";
  for (const auto &cluster : in_clusters_) {
    ss << cluster->id_ << ",";
  }
  ss << ")->(";
  for (const auto &cluster : out_clusters_) {
    ss << cluster->id_ << ",";
  }
  ss << ")|";
  for (const auto &node : nodes_) {
    ss << (node->GetName() + "|");
  }
  return ss.str();
}

size_t BaseCluster::MinId() const { return min_; }
size_t BaseCluster::Id() const { return id_; }
size_t BaseCluster::UniqueId() const { return unique_id_; }
void BaseCluster::UpdateRank(size_t rank) {
  max_ = rank;
  min_ = rank;
};

bool BaseCluster::IsData() const { return partitioner_->type_index_to_type_[type_index_] == kClusterData; }
bool BaseCluster::IsIndependent() const { return partitioner_->type_index_to_type_[type_index_] == kClusterStage; }
bool BaseCluster::IsNetOutput() const { return partitioner_->type_index_to_type_[type_index_] == kClusterNetoutput; }
bool BaseCluster::IsInputNode() const { return partitioner_->type_index_to_type_[type_index_] == kClusterInputNode; }

bool BaseCluster::IsRefVariable() const {
  if ((nodes_.size() == 1) && OpTypeUtils::IsVarLikeNode(nodes_[0]->GetType())) {
    std::string ref_variable_name;
    return (AttrUtils::GetStr(nodes_[0]->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_variable_name) &&
            !ref_variable_name.empty());
  }
  return false;
}

void BaseCluster::AddInput(ClusterPtr in) {
  if (std::find(in_clusters_.begin(), in_clusters_.end(), in.get()) != in_clusters_.end()) {
    return;
  }
  in_clusters_.insert(in_clusters_.end(), in.get());
  if (std::find(in->out_clusters_.begin(), in->out_clusters_.end(), this) != in->out_clusters_.end()) {
    return;
  }
  in->out_clusters_.insert(in->out_clusters_.end(), this);
};
void BaseCluster::RemoveInput(ClusterPtr in) {
  in_clusters_.erase(std::remove(in_clusters_.begin(), in_clusters_.end(), in.get()), in_clusters_.end());
  in->out_clusters_.erase(std::remove(in->out_clusters_.begin(), in->out_clusters_.end(), this),
                          in->out_clusters_.end());
};
void BaseCluster::AddOutput(ClusterPtr out) {
  if (std::find(out_clusters_.begin(), out_clusters_.end(), out.get()) != out_clusters_.end()) {
    return;
  }
  out_clusters_.insert(out_clusters_.end(), out.get());
  if (std::find(out->in_clusters_.begin(), out->in_clusters_.end(), this) != out->in_clusters_.end()) {
    return;
  }
  out->in_clusters_.insert(out->in_clusters_.end(), this);
};
void BaseCluster::RemoveOutput(ClusterPtr out) {
  out_clusters_.erase(std::remove(out_clusters_.begin(), out_clusters_.end(), out.get()), out_clusters_.end());
  out->in_clusters_.erase(std::remove(out->in_clusters_.begin(), out->in_clusters_.end(), this),
                          out->in_clusters_.end());
};

void BaseCluster::Merge(ClusterPtr other) {
  if (other->IsIndependent()) { return; }
  nodes_.insert(nodes_.end(), other->nodes_.begin(), other->nodes_.end());
  other->in_clusters_.erase(std::remove(other->in_clusters_.begin(), other->in_clusters_.end(), this),
                            other->in_clusters_.end());
  other->out_clusters_.erase(std::remove(other->out_clusters_.begin(), other->out_clusters_.end(), this),
                             other->out_clusters_.end());
  in_clusters_.erase(std::remove(in_clusters_.begin(), in_clusters_.end(), other.get()), in_clusters_.end());
  out_clusters_.erase(std::remove(out_clusters_.begin(), out_clusters_.end(), other.get()), out_clusters_.end());
  auto in_clusters = other->in_clusters_;
  for (const auto &cluster : in_clusters) {
    cluster->RemoveOutput(other);
    cluster->AddOutput(shared_from_this());
  }
  auto out_clusters = other->out_clusters_;
  for (const auto &cluster : out_clusters) {
    cluster->RemoveInput(other);
    cluster->AddInput(shared_from_this());
  }
  if (other->max_ > max_) {
    max_ = other->max_;
  }
  if (other->min_ < min_) {
    min_ = other->min_;
  }
}

bool BaseCluster::TryMerge(ClusterPtr other) {
  std::queue<BaseCluster *> forward_reached;
  forward_reached.push(other.get());

  // Try merge other cluster to this cluster, ONLY if will not leads to a ring
  std::unordered_set<BaseCluster *> visited;
  while (!forward_reached.empty()) {
    auto current_cluster = forward_reached.front();
    forward_reached.pop();
    for (const auto &cluster : current_cluster->out_clusters_) {
      if ((cluster->max_ == max_) && (current_cluster != other.get())) {
        return false;
      } else if (cluster->min_ < max_) {
        if (visited.insert(cluster).second) {
          forward_reached.push(cluster);
        }
      }
    }
  }
  Merge(other);
  return true;
}

void BaseCluster::MergeInOutClusters(const std::unordered_set<BaseCluster *> &path_cluster_set,
                                     const std::vector<BaseCluster *> &ordered_path_clusters) {
  std::vector<BaseCluster *> merge_in_clusters;
  std::vector<BaseCluster *> merge_out_clusters;
  std::unordered_set<BaseCluster *> unique_path_in_clusters;
  std::unordered_set<BaseCluster *> unique_path_out_clusters;
  // insert path cluster input & output
  for (auto &path_cluster : ordered_path_clusters) {
    for (const auto &cluster_input : path_cluster->in_clusters_) {
      if ((path_cluster_set.find(cluster_input) == path_cluster_set.cend()) &&
          unique_path_in_clusters.insert(cluster_input).second) {
        merge_in_clusters.emplace_back(cluster_input);
      }
    }
    for (const auto &cluster_output : path_cluster->out_clusters_) {
      if ((path_cluster_set.find(cluster_output) == path_cluster_set.cend()) &&
          unique_path_out_clusters.insert(cluster_output).second) {
        merge_out_clusters.emplace_back(cluster_output);
      }
    }
    // update max
    max_ = (path_cluster->max_ > max_) ? path_cluster->max_ : max_;
    // update min
    min_ = (path_cluster->min_ < min_) ? path_cluster->min_ : min_;
  }
  in_clusters_ = std::move(merge_in_clusters);
  out_clusters_ = std::move(merge_out_clusters);
}

void BaseCluster::UpdateInOutClusters(const std::unordered_set<BaseCluster *> &path_cluster_set) {
  // update in clusters's output
  for (auto &in_cluster : in_clusters_) {
    const auto &it = in_cluster;
    for (const auto &out_cluster : path_cluster_set) {
      it->out_clusters_.erase(std::remove(it->out_clusters_.begin(), it->out_clusters_.end(), out_cluster),
                              it->out_clusters_.end());
    }
    if (std::find(it->out_clusters_.begin(), it->out_clusters_.end(), this) == it->out_clusters_.end()) {
      it->out_clusters_.emplace_back(this);
    }
  }
  // update out clusters's input
  for (auto &out_cluster : out_clusters_) {
    const auto &it = out_cluster;
    for (const auto &in_cluster : path_cluster_set) {
      it->in_clusters_.erase(std::remove(it->in_clusters_.begin(), it->in_clusters_.end(), in_cluster),
                             it->in_clusters_.end());
    }
    if (std::find(it->in_clusters_.begin(), it->in_clusters_.end(), this) == it->in_clusters_.end()) {
      it->in_clusters_.emplace_back(this);
    }
  }
}

void BaseCluster::MergeAllPath(const std::vector<ClusterPtr> &path_clusters) {
  // merge all cluster to set
  constexpr int32_t kUnknownShapeTypeIndex = 5;
  std::unordered_set<BaseCluster *> path_cluster_set;
  std::vector<BaseCluster *> ordered_path_clusters;
  ordered_path_clusters.reserve(path_clusters.size() + 1U);
  path_cluster_set.insert(this);
  ordered_path_clusters.emplace_back(this);
  size_t node_size = nodes_.size();
  for (const auto &path_cluster : path_clusters) {
    if (path_cluster->IsIndependent()) {
      continue;
    }
    if (path_cluster->type_index_ == kUnknownShapeTypeIndex) {
      type_index_ = kUnknownShapeTypeIndex;
    }
    path_cluster_set.insert(path_cluster.get());
    ordered_path_clusters.emplace_back(path_cluster.get());
    node_size += path_cluster->nodes_.size();
  }
  nodes_.reserve(node_size);
  for (const auto &path_cluster : path_clusters) {
    nodes_.insert(nodes_.end(), path_cluster->nodes_.begin(), path_cluster->nodes_.end());
  }
  MergeInOutClusters(path_cluster_set, ordered_path_clusters);
  path_cluster_set.erase(this);
  UpdateInOutClusters(path_cluster_set);
}

std::vector<ClusterPtr> BaseCluster::MergeAllPathFrom(const ClusterPtr &other) {
  std::queue<BaseCluster *> forward_reached_queue;
  std::queue<BaseCluster *> backward_reached_queue;

  std::unordered_set<BaseCluster *> forward_reached_clusters;
  std::unordered_set<BaseCluster *> backward_reached_clusters;
  std::vector<ClusterPtr> path_clusters;
  if (other->IsIndependent()) {
    return path_clusters;
  }

  path_clusters.push_back(other);
  forward_reached_queue.push(other.get());
  backward_reached_queue.push(this);
  while (!forward_reached_queue.empty()) {
    auto current_cluster = forward_reached_queue.front();
    forward_reached_queue.pop();
    for (const auto &cluster : current_cluster->out_clusters_) {
      if (cluster->min_ < max_ && cluster->max_ != max_ && forward_reached_clusters.count(cluster) == 0) {
        forward_reached_clusters.insert(cluster);
        forward_reached_queue.push(cluster);
      }
    }
  }
  while (!backward_reached_queue.empty()) {
    auto current_cluster = backward_reached_queue.front();
    backward_reached_queue.pop();
    for (const auto &cluster : current_cluster->in_clusters_) {
      if (cluster->max_ > other->min_ && cluster->max_ != other->max_ &&
          backward_reached_clusters.count(cluster) == 0) {
        backward_reached_clusters.insert(cluster);
        backward_reached_queue.push(cluster);
        if (forward_reached_clusters.count(cluster) != 0) {
          path_clusters.push_back(cluster->shared_from_this());
        }
      }
    }
  }
  MergeAllPath(path_clusters);
  return path_clusters;
}
std::vector<BaseCluster *> BaseCluster::Inputs() const { return in_clusters_; }
std::vector<BaseCluster *> BaseCluster::Outputs() const { return out_clusters_; }
const std::vector<NodePtr> &BaseCluster::Nodes() const { return nodes_; }

bool BaseCluster::AddFrameInput(InDataAnchorPtr anchor) {
  bool added = true;
  if (anchor != nullptr && anchor->GetPeerOutAnchor() != nullptr) {
    auto index = inputs_.size();
    if (merge_inputs_) {
      GELOGD("Merge inputs is enabled");
      auto src_node = anchor->GetPeerOutAnchor()->GetOwnerNode();
      std::string src_key = src_node->GetName() + ":" + std::to_string(anchor->GetPeerOutAnchor()->GetIdx());
      std::map<std::string, size_t>::const_iterator it = src_key_to_frame_input_.find(src_key);
      if (it != src_key_to_frame_input_.cend()) {
        index = it->second;
        GELOGD("[%s:%d] Reuse data index: %zu",
               anchor->GetOwnerNode()->GetName().c_str(),
               anchor->GetIdx(),
               it->second);
        added = false;
      } else {
        inputs_.push_back(anchor);
        inputs_index_[anchor] = index;
        GELOGD("[%s:%d] Assign data index: %zu",
               anchor->GetOwnerNode()->GetName().c_str(),
               anchor->GetIdx(),
               index);
        src_key_to_frame_input_[src_key] = index;
      }
    } else {
      inputs_index_[anchor] = index;
      inputs_.push_back(anchor);
    }
    data_to_node_inputs_[index].emplace_back(anchor);
  }
  return added;
}

void BaseCluster::AddFrameOutput(OutDataAnchorPtr anchor) {
  if (anchor != nullptr) {
    outputs_index_[anchor] = outputs_.size();
    outputs_.push_back(anchor);
  }
}

InDataAnchorPtr BaseCluster::GetFrameInDataAnchor(InDataAnchorPtr anchor) {
  return partition_node_->GetInDataAnchor(static_cast<int32_t>(inputs_index_[anchor]));
}

OutDataAnchorPtr BaseCluster::GetFrameOutDataAnchor(OutDataAnchorPtr anchor) {
  return partition_node_->GetOutDataAnchor(static_cast<int32_t>(outputs_index_[anchor]));
}

InControlAnchorPtr BaseCluster::GetFrameInControlAnchor() { return partition_node_->GetInControlAnchor(); };

OutControlAnchorPtr BaseCluster::GetFrameOutControlAnchor() { return partition_node_->GetOutControlAnchor(); };

Status BaseCluster::BuildFrame() {
  if (partitioner_->IsNeedBuildPartitionFrame(*this)) {
    return BuildPartitionFrame();
  } else {
    auto node = nodes_.front();
    auto in_control_anchor = node->GetInControlAnchor();
    if (in_control_anchor != nullptr) {
      for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
        auto src_cluster = partitioner_->node_2_cluster_[peer_out_control_anchor->GetOwnerNode()];
        if (src_cluster->id_ != id_) {
          GE_CHK_GRAPH_STATUS_RET(
              GraphUtils::RemoveEdge(peer_out_control_anchor, in_control_anchor),
              "[Remove][Edge] from node %s index %d to node %s failed, index %d.",
              peer_out_control_anchor->GetOwnerNode()->GetName().c_str(), AnchorUtils::GetIdx(peer_out_control_anchor),
              in_control_anchor->GetOwnerNode()->GetName().c_str(), AnchorUtils::GetIdx(in_control_anchor));
          control_inputs_.insert(src_cluster);
          src_cluster->control_outputs_.insert(peer_out_control_anchor);
        }
      }
    }
    if (IsData() || IsIndependent()) {
      for (const auto &anchor : node->GetAllOutDataAnchors()) {
        AddFrameOutput(anchor);
      }
    } else {
      for (const auto &anchor : node->GetAllInDataAnchors()) {
        (void)AddFrameInput(anchor);
      }
    }
    partition_node_ = node;
  }
  return SUCCESS;
}

Status BaseCluster::AddPartitionedCallNodeOutput(const NodePtr &node, const OpDescPtr &partition_op) {
  for (const auto &anchor : node->GetAllOutDataAnchors()) {
    auto peer_in_anchors = anchor->GetPeerInDataAnchors();
    for (const auto &peer_in_anchor : peer_in_anchors) {
      auto src_cluster = partitioner_->node_2_cluster_[peer_in_anchor->GetOwnerNode()];
      if (src_cluster->id_ != id_) {
        AddFrameOutput(anchor);
        GE_CHK_GRAPH_STATUS_RET(partition_op->AddOutputDesc(node->GetOpDesc()->GetOutputDesc(anchor->GetIdx())),
                                "[Add][OutputDesc] to op:%s failed.", partition_op->GetName().c_str());
        auto tensor = partition_op->MutableOutputDesc(partition_op->GetAllOutputsDescPtr().size() - 1U);
        GE_CHK_GRAPH_STATUS_RET(UpdateFrameTensor(node->GetOpDesc(), tensor),
                                "[Call][UpdateFrameOutTensor] failed, from %s to %s:%d.", node->GetName().c_str(),
                                partition_op->GetName().c_str(), anchor->GetIdx());
        break;
      }
    }
  }
  return SUCCESS;
}

Status BaseCluster::RemoveNodeFromRoot(const ge::ComputeGraphPtr &graph) {
  for (auto &node : nodes_) {
    GE_ASSERT_NOTNULL(subgraph_->AddNode(node), "[Add][Node] failed.");
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveJustNode(graph, node), "[Remove][JustNode] failed, graph:%s, node:%s.",
                            graph->GetName().c_str(), node->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(node->SetOwnerComputeGraph(subgraph_), "[Set][OwnerComputeGraph] %s for node:%s failed.",
                            subgraph_->GetName().c_str(), node->GetName().c_str());
  }
  return SUCCESS;
}

Status BaseCluster::BuildPartitionNodes(const OpDescPtr &partition_op) {
  for (auto &node : nodes_) {
    for (const auto &anchor : node->GetAllInDataAnchors()) {
      auto peer_out_anchor = anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;  // Skip overhang input.
      }
      auto src_cluster = partitioner_->node_2_cluster_[peer_out_anchor->GetOwnerNode()];
      if (src_cluster->id_ != id_) {
        if (AddFrameInput(anchor)) {
          GE_CHK_GRAPH_STATUS_RET(partition_op->AddInputDesc(node->GetOpDesc()->GetInputDesc(anchor->GetIdx())),
                                  "[Add][InputDesc] to op:%s failed.", partition_op->GetName().c_str());
          auto tensor = partition_op->MutableInputDesc(partition_op->GetAllInputsSize() - 1U);
          GE_CHK_GRAPH_STATUS_RET(UpdateFrameTensor(node->GetOpDesc(), tensor),
                                  "[Call][UpdateFrameInTensor] failed, from %s to %s:%zu.", node->GetName().c_str(),
                                  partition_op->GetName().c_str(), partition_op->GetAllInputsSize() - 1U);
        }
      }
    }
    auto in_control_anchor = node->GetInControlAnchor();
    if (in_control_anchor != nullptr) {
      for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
        if (peer_out_control_anchor == nullptr) {
          continue;
        }
        auto src_cluster = partitioner_->node_2_cluster_[peer_out_control_anchor->GetOwnerNode()];
        if (src_cluster->id_ != id_) {
          GE_CHK_GRAPH_STATUS_RET(
              GraphUtils::RemoveEdge(peer_out_control_anchor, in_control_anchor),
              "[Remove][Edge] from %s:%d to %s:%d failed.", peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
              peer_out_control_anchor->GetIdx(), node->GetName().c_str(), in_control_anchor->GetIdx());
          control_inputs_.insert(src_cluster);
          src_cluster->control_outputs_.insert(peer_out_control_anchor);
        }
      }
    }
    GE_CHK_GRAPH_STATUS_RET(AddPartitionedCallNodeOutput(node, partition_op),
                            "[Add][PartitionedCallNodeOutput] failed, node[%s], partitioned_call[%s].",
                            node->GetName().c_str(), partition_op->GetName().c_str());
  }
  return SUCCESS;
}

Status BaseCluster::BuildPartitionFrame() {
  auto graph = partitioner_->root_graph_;
  std::string sub_graph_name = partitioner_->GetSubgraphName(*this);
  subgraph_ = MakeShared<ComputeGraph>(sub_graph_name);
  GE_ASSERT_NOTNULL(subgraph_, "[New][Memory] for subgraph failed, name:%s.", sub_graph_name.c_str());
  auto partitioned_op =
      MakeShared<OpDesc>(sub_graph_name + "_PartitionedCall_" + std::to_string(unique_id_++), "PartitionedCall");
  GE_ASSERT_NOTNULL(partitioned_op, "[New][Memory] for partition op failed.");
  GE_CHK_GRAPH_STATUS_RET(partitioned_op->AddSubgraphName(subgraph_->GetName()), "[Add][SubgraphName] %s for op:%s.",
                          subgraph_->GetName().c_str(), partitioned_op->GetName().c_str());
  GE_CHK_GRAPH_STATUS_RET(partitioned_op->SetSubgraphInstanceName(0, subgraph_->GetName()),
                          "[Call][SetSubgraphInstanceName] for op:%s failed, index:0, name:%s.",
                          partitioned_op->GetName().c_str(), subgraph_->GetName().c_str());
  GE_CHK_STATUS_RET(BuildPartitionNodes(partitioned_op),
                    "[Call][SetSubgraphInstanceName] for op:%s failed, index:0, name:%s.",
                    partitioned_op->GetName().c_str(), subgraph_->GetName().c_str());
  partition_node_ = AddPartitionedCallKeepTopo(graph, nodes_, partitioned_op);
  GE_ASSERT_NOTNULL(partition_node_, "[Add][Node] %s to graph:%s failed.", partitioned_op->GetName().c_str(),
                    graph->GetName().c_str());
  GE_CHK_GRAPH_STATUS_RET(partition_node_->SetOwnerComputeGraph(graph),
                          "[Set][OwnerComputeGraph] %s for node:%s failed.", graph->GetName().c_str(),
                          partitioned_op->GetName().c_str());
  GE_CHK_STATUS_RET(RemoveNodeFromRoot(graph),
                    "[Call][SetSubgraphInstanceName] for op:%s failed, index:0, name:%s.",
                    partitioned_op->GetName().c_str(), subgraph_->GetName().c_str());
  subgraph_->SetParentNode(partition_node_);
  subgraph_->SetParentGraph(graph);
  // all partition add subgraph to root_graph
  auto root_graph = GraphUtils::FindRootGraph(graph);
  GE_CHK_GRAPH_STATUS_RET(root_graph->AddSubgraph(subgraph_), "[Add][Subgraph] %s to root graph:%s failed.",
                          subgraph_->GetName().c_str(), graph->GetName().c_str());
  std::string session_graph_id;
  GE_ASSERT_TRUE(AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                 "[Get][Attr] %s on root graph:%s failed.", ATTR_NAME_SESSION_GRAPH_ID.c_str(),
                 graph->GetName().c_str());
  GE_ASSERT_TRUE(AttrUtils::SetStr(*subgraph_, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                 "[Set][Attr] %s on subgraph:%s failed.", ATTR_NAME_SESSION_GRAPH_ID.c_str(),
                 subgraph_->GetName().c_str());
  return SUCCESS;
}

Status BaseCluster::CombinePartitionFrame() {
  size_t index = 0U;
  for (const auto &anchor : inputs_) {
    auto peer_out_anchor = anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    auto src_cluster = partitioner_->node_2_cluster_[peer_out_anchor->GetOwnerNode()];
    auto src_anchor = src_cluster->GetFrameOutDataAnchor(peer_out_anchor);
    auto dst_anchor = GetFrameInDataAnchor(anchor);
    for (const auto &node_anchor : data_to_node_inputs_[index]) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(peer_out_anchor, node_anchor),
                              "[Remove][Edge] from %s:%d to %s:%d fail.",
                              peer_out_anchor->GetOwnerNode()->GetName().c_str(), peer_out_anchor->GetIdx(),
                              node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx());
    }
    GE_ASSERT_NOTNULL(src_anchor);
    GE_ASSERT_NOTNULL(dst_anchor);
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(src_anchor, dst_anchor), "[Add][Edge] from %s:%d to %s:%d failed.",
                            src_anchor->GetOwnerNode()->GetName().c_str(), src_anchor->GetIdx(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetIdx());
    ++index;
  }
  for (const auto &src_cluster : control_inputs_) {
    auto src_anchor = src_cluster->GetFrameOutControlAnchor();
    auto dst_anchor = GetFrameInControlAnchor();
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(src_anchor, dst_anchor), "[Add][Edge] from %s:%d to %s:%d failed.",
                            src_anchor->GetOwnerNode()->GetName().c_str(), src_anchor->GetIdx(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetIdx());
  }
  return SUCCESS;
}

Status BaseCluster::BuildPartitionSubgraph() {
  if (IsData() || IsNetOutput() || IsIndependent()) {
    return SUCCESS;
  }
  int64_t parent_node_index = 0;
  for (const auto &anchor : inputs_) {
    GE_CHECK_NOTNULL(subgraph_);
    auto data_op =
        MakeShared<OpDesc>(subgraph_->GetName() + std::string("_Data_") + std::to_string(parent_node_index), ge::DATA);
    GE_ASSERT_NOTNULL(data_op, "[New][Memory] for data op failed.");
    auto input_desc = anchor->GetOwnerNode()->GetOpDesc()->GetInputDesc(anchor->GetIdx());
    GE_CHK_GRAPH_STATUS_RET(data_op->AddInputDesc(input_desc), "[Add][InputDesc] to op:%s failed, graph %s.",
                            data_op->GetName().c_str(), subgraph_->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(data_op->AddOutputDesc(input_desc), "[Add][OutputDesc] to op:%s failed, graph %s.",
                            data_op->GetName().c_str(), subgraph_->GetName().c_str());
    GE_ASSERT_TRUE(AttrUtils::SetInt(data_op, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index),
                   "[Set][Attr] %s on subgraph data node:%s failed.", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                   data_op->GetName().c_str());
    // node and opdesc can not be nullptr
    GE_CHK_STATUS_RET(CopyOpAttr(anchor->GetOwnerNode()->GetOpDesc(), data_op), "[Call][CopyOpAttr] data op failed.");
    auto data_node = subgraph_->AddNode(data_op);
    GE_ASSERT_NOTNULL(data_node, "[Add][Node] %s to subgraph:%s failed.", data_op->GetName().c_str(),
                      subgraph_->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(data_node->SetOwnerComputeGraph(subgraph_), "[Set][OwnerGraph] %s of data node:%s failed.",
                            subgraph_->GetName().c_str(), data_op->GetName().c_str());
    for (const auto &node_anchor : data_to_node_inputs_[parent_node_index]) {
      GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_anchor),
                              "[Call][AddEdge] Failed add data input edge to %s:%d",
                              node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx());
    }
    parent_node_index++;
  }
  if (outputs_.empty() && control_outputs_.empty()) {
    return SUCCESS;
  }

  auto net_output_op = MakeShared<OpDesc>(subgraph_->GetName() + "_" + NODE_NAME_NET_OUTPUT, ge::NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output_op, "[New][Memory] for netoutput op failed.");
  GE_CHK_STATUS_RET(SetAttrToNetoutput(net_output_op), "[Call][SetAttrToNetoutput] net_output_op failed.");
  for (size_t i = 0; i < outputs_.size(); ++i) {
    GeTensorDesc input_desc;
    GE_CHK_GRAPH_STATUS_RET(net_output_op->AddInputDesc(input_desc), "[Add][InputDesc] to op:%s failed.",
                            net_output_op->GetName().c_str());
  }
  auto net_output_node = subgraph_->AddNode(net_output_op);
  GE_ASSERT_NOTNULL(net_output_node, "[Call][AddNode] Failed add netoutput node:%s to subgraph:%s.",
                    net_output_op->GetName().c_str(), subgraph_->GetName().c_str());
  GE_CHK_GRAPH_STATUS_RET(net_output_node->SetOwnerComputeGraph(subgraph_),
                          "[Set][OwnerGraph] %s of netoutput node:%s failed.", subgraph_->GetName().c_str(),
                          net_output_node->GetName().c_str());
  parent_node_index = 0;
  for (const auto &anchor : outputs_) {
    auto output_desc = anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(anchor->GetIdx()));
    GE_ASSERT_TRUE(AttrUtils::SetInt(output_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index),
                   "[Set][Attr] parent_node_index on subgraph node:%s netoutput's input failed.",
                   anchor->GetOwnerNode()->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(net_output_op->UpdateInputDesc(parent_node_index, output_desc),
                            "[Update][InputDesc] of netoutput node:%s failed.", net_output_op->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(anchor, net_output_node->GetInDataAnchor(parent_node_index)),
                            "[Add][Edge] from %s:%d to netoutput node:%s failed.",
                            anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx(),
                            net_output_op->GetName().c_str());
    parent_node_index++;
  }
  for (const auto &anchor : control_outputs_) {
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(anchor, net_output_node->GetInControlAnchor()),
                            "[Add][ControlEdge] from %s:%d to netoutput node:%s failed.",
                            anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx(),
                            net_output_op->GetName().c_str());
  }
  return SUCCESS;
}

void BaseCluster::Clear() {
  in_clusters_.clear();
  out_clusters_.clear();
  nodes_.clear();
  inputs_index_.clear();
  outputs_index_.clear();
  inputs_.clear();
  outputs_.clear();
  control_inputs_.clear();
  control_outputs_.clear();
  partition_node_.reset();
  subgraph_.reset();
  unique_id_ = 0;
}

void BaseCluster::SetMergeInputs(bool merge_inputs) {
  merge_inputs_ = merge_inputs;
}

Status BaseCluster::SetGraphId(const uint32_t graph_id) const {
  if (subgraph_ == nullptr) {
    GELOGW("Subgraph is null, can not set graph id, graph_id=%u.", graph_id);
    return SUCCESS;
  }
  subgraph_->SetGraphID(graph_id);
  return SUCCESS;
}

std::vector<NodePtr> &BaseCluster::GetMutableDirectNode() {
  return nodes_;
}

Status BaseCluster::UpdateFrameTensor(const OpDescPtr &op_desc, GeTensorDescPtr &tensor) {
  GE_ASSERT_NOTNULL(op_desc, "[Check][Valid] op_desc is invalid");
  GE_ASSERT_NOTNULL(tensor, "[Check][Valid] tensor is invalid");
  const auto &node_attr_names = PartitionNodeAttrNameManager::GetInstance().GetAllPartitionNodeAttrNames();
  for (const auto &item : node_attr_names) {
    if (item.IsSupportTensor()) {
      GE_CHK_STATUS_RET(CopyAttr(op_desc, tensor, item), "[Call][CopyAttr] failed, from %s to %s.",
                        op_desc->GetName().c_str(), tensor->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status BaseCluster::CopyOpAttr(const OpDescPtr &from, OpDescPtr &to) {
  if (from == nullptr) {
    return SUCCESS;
  }
  const auto &node_attr_names = PartitionNodeAttrNameManager::GetInstance().GetAllPartitionNodeAttrNames();
  for (const auto &item : node_attr_names) {
    if (item.IsNeedCopy()) {
      GE_CHK_STATUS_RET(CopyAttr(from, to, item), "[Call][CopyAttr] failed, from %s to %s.", from->GetName().c_str(),
                        to->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status BaseCluster::CopyAttr(AttrUtils::ConstAttrHolderAdapter &&from, AttrUtils::AttrHolderAdapter &&to,
                             const PartitionNodeAttrName &item) {
  const auto get_fn = item.GetAttrFunction();
  const auto set_fn = item.SetAttrFunction();
  GE_CHECK_NOTNULL(get_fn);
  GE_CHECK_NOTNULL(set_fn);
  AnyValue val;
  const auto has_attr = get_fn(from.get(), item.GetAttrName(), val);
  if (has_attr) {
    GE_ASSERT_TRUE(set_fn(to.get(), item.GetAttrName(), val), "[Call][set_fn] failed, attr %s",
                   item.GetAttrName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge
