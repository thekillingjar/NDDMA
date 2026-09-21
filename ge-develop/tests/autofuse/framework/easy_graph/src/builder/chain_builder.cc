/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_graph/builder/chain_builder.h"
#include "easy_graph/builder/graph_builder.h"

EG_NS_BEGIN

ChainBuilder::ChainBuilder(GraphBuilder &graphBuilder, EdgeType defaultEdgeType)
    : linker(*this, defaultEdgeType), graph_builder_(graphBuilder) {}

ChainBuilder::LinkBuilder *ChainBuilder::operator->() {
  return &linker;
}

ChainBuilder &ChainBuilder::LinkTo(const Node &node, const Link &link) {
  Node *currentNode = graph_builder_.BuildNode(node);
  if (prev_node_) {
    graph_builder_.BuildEdge(*prev_node_, *currentNode, link);
  }
  prev_node_ = currentNode;
  return *this;
}

const Node *ChainBuilder::FindNode(const NodeId &id) const {
  return graph_builder_->FindNode(id);
}

ChainBuilder::LinkBuilder::LinkBuilder(ChainBuilder &chain, EdgeType defaultEdgeType)
    : chain_(chain), default_edge_type_(defaultEdgeType), from_link_(defaultEdgeType) {}

ChainBuilder &ChainBuilder::LinkBuilder::Node(const NodeObj &node) {
  chain_.LinkTo(node, from_link_);
  from_link_.Reset(default_edge_type_);
  return chain_;
}

ChainBuilder &ChainBuilder::LinkBuilder::startLink(const Link &link) {
  this->from_link_ = link;
  return chain_;
}

ChainBuilder &ChainBuilder::LinkBuilder::Ctrl(const std::string &label) {
  return this->Edge(EdgeType::CTRL, UNDEFINED_PORT_ID, UNDEFINED_PORT_ID, label);
}

ChainBuilder &ChainBuilder::LinkBuilder::Data(const std::string &label) {
  return this->Edge(EdgeType::DATA, UNDEFINED_PORT_ID, UNDEFINED_PORT_ID, label);
}

ChainBuilder &ChainBuilder::LinkBuilder::Data(PortId srcId, PortId dstId, const std::string &label) {
  return this->Edge(EdgeType::DATA, srcId, dstId, label);
}

ChainBuilder &ChainBuilder::LinkBuilder::Edge(EdgeType type, PortId srcPort, PortId dstPort, const std::string &label) {
  return this->startLink(Link(type, label, srcPort, dstPort));
}

EG_NS_END
