/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cluster.h"
#include <sstream>
namespace optimize {
void Cluster::AddInput(Cluster &input) {
  if (std::find(inputs_.cbegin(), inputs_.cend(), &input) != inputs_.cend()) {
    return;
  }
  inputs_.insert(&input);
  if (std::find(input.outputs_.cbegin(), input.outputs_.cend(), this) != input.outputs_.cend()) {
    return;
  }
  input.outputs_.insert(this);
}

void Cluster::AddOutput(Cluster &output) {
  if (std::find(outputs_.cbegin(), outputs_.cend(), &output) != outputs_.cend()) {
    return;
  }
  outputs_.insert(&output);
  if (std::find(output.inputs_.cbegin(), output.inputs_.cend(), this) != output.inputs_.cend()) {
    return;
  }
  output.inputs_.insert(this);
}

void Cluster::MergeFrom(Cluster &from) {
  nodes_.insert(nodes_.cend(), from.nodes_.cbegin(), from.nodes_.cend());
  node_set_.insert(from.node_set_.begin(), from.node_set_.end());
  from.inputs_.erase(this);
  from.outputs_.erase(this);
  inputs_.erase(&from);
  outputs_.erase(&from);
  meta_data_.ins_num += from.meta_data_.ins_num;

  auto in_clusters = from.inputs_;
  for (const auto &cluster : in_clusters) {
    cluster->RemoveOutput(from);
    cluster->AddOutput(*this);
  }
  auto out_clusters = from.outputs_;
  for (const auto &cluster : out_clusters) {
    cluster->RemoveInput(from);
    cluster->AddInput(*this);
  }
  auto connected_nodes = Cluster::FindConnectedNodes(from, *this);
  in_nodes_ = Cluster::CalculateMergedInNodes(from, *this, connected_nodes);
  out_nodes_ = Cluster::CalculateMergedOutNodes(from, *this);
}

void Cluster::RemoveInput(Cluster &input) {
  inputs_.erase(&input);
  input.outputs_.erase(this);
}

void Cluster::RemoveOutput(Cluster &output) {
  outputs_.erase(&output);
  output.inputs_.erase(this);
}

const std::list<ge::AscNodePtr> &Cluster::Nodes() const {
  return nodes_;
}

std::unordered_set<ge::AscNodePtr> Cluster::FindConnectedNodes(const Cluster &pre, const Cluster &post) {
  std::unordered_set<ge::AscNodePtr> connected_nodes;
  for (const auto &node : pre.out_nodes_) {
    if (post.in_nodes_.count(node) > 0UL) {
      connected_nodes.insert(node);
    }
  }
  return connected_nodes;
}

std::unordered_set<ge::AscNodePtr> Cluster::CalculateMergedInNodes(
    const Cluster &pre, const Cluster &post, const std::unordered_set<ge::AscNodePtr> &connected_nodes) {
  std::unordered_set<ge::AscNodePtr> merged_in;

  for (const auto &node : pre.in_nodes_) {
    merged_in.insert(node);
  }

  for (const auto &node : post.in_nodes_) {
    if (connected_nodes.count(node) == 0UL) {
      merged_in.insert(node);
    }
  }

  return merged_in;
}

std::unordered_set<ge::AscNodePtr> Cluster::CalculateMergedOutNodes(const Cluster &pre, const Cluster &post) {
  std::unordered_set<ge::AscNodePtr> merged_out;

  for (const auto &node: post.out_nodes_) {
    merged_out.insert(node);
  }
  for (const auto &node: pre.out_nodes_) {
    for (const auto &out_node: node->GetOutDataNodes()) {
      auto asc_out_node = std::dynamic_pointer_cast<ge::AscNode>(out_node);
      if ((asc_out_node != nullptr) && !post.ContainsNode(asc_out_node)) {
        merged_out.insert(node);
      }
    }
  }

  return merged_out;
}

std::string Cluster::DebugString() const {
  std::stringstream ss;
  ss << "id:" << id_ << ", node_size:" << nodes_.size() << ", enable_vf:" << meta_data_.enable_vf << ",";
  ss << " inputs:[";
  for (const auto &cluster : inputs_) {
    ss << cluster->id_ << ",";
  }
  ss << "] outputs:[";
  for (const auto &cluster : outputs_) {
    ss << cluster->id_ << ",";
  }
  ss << "] nodes:|";
  for (const auto &node : nodes_) {
    ss << (node->GetName() + "|");
  }
  return ss.str();
}
}  // namespace optimize