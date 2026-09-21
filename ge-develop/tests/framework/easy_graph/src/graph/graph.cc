/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_graph/graph/graph.h"
#include "easy_graph/graph/graph_visitor.h"
#include "easy_graph/layout/graph_layout.h"
#include <algorithm>

EG_NS_BEGIN

Graph::Graph(const std::string &name) : name_(name) {}

std::string Graph::GetName() const {
  return name_;
}

Node *Graph::AddNode(const Node &node) {
  auto result = nodes_.emplace(node.GetId(), node);
  return &(result.first->second);
}

Edge *Graph::AddEdge(const Edge &edge) {
  auto result = edges_.emplace(edge);
  return &(const_cast<Edge &>(*(result.first)));
}

Node *Graph::FindNode(const NodeId &id) {
  auto it = nodes_.find(id);
  if (it == nodes_.end()) {
    return nullptr;
  }
  return &(it->second);
}

const Node *Graph::FindNode(const NodeId &id) const {
  return const_cast<Graph &>(*this).FindNode(id);
}

std::pair<const Node *, const Node *> Graph::FindNodePair(const Edge &edge) const {
  return std::make_pair(FindNode(edge.GetSrc().getNodeId()), FindNode(edge.GetDst().getNodeId()));
}

std::pair<Node *, Node *> Graph::FindNodePair(const Edge &edge) {
  return std::make_pair(FindNode(edge.GetSrc().getNodeId()), FindNode(edge.GetDst().getNodeId()));
}

void Graph::Accept(GraphVisitor &visitor) const {
  visitor.Visit(*this);
  std::for_each(nodes_.begin(), nodes_.end(), [&visitor](const auto &node) { visitor.Visit(node.second); });
  std::for_each(edges_.begin(), edges_.end(), [&visitor](const auto &edge) { visitor.Visit(edge); });
}

Status Graph::Layout(const LayoutOption *option) const {
  return GraphLayout::GetInstance().Layout(*this, option);
}
void Graph::AddOutputAnchor(NodeId id, int32_t out_index) {
  output_points_.emplace_back(std::move(id), out_index);
}
const std::vector<std::pair<NodeId, int32_t>> &Graph::GetOutputEndPoints() const {
  return output_points_;
}
void Graph::AddTarget(NodeId id) {
  targets_.emplace_back(std::move(id));
}
const std::vector<NodeId> &Graph::GetTargets() const {
  return targets_;
}

EG_NS_END
