/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_graph/graph/node.h"
#include "easy_graph/graph/graph_visitor.h"
#include <algorithm>

EG_NS_BEGIN

__DEF_EQUALS(Node) {
  return id_ == rhs.id_;
}

__DEF_COMP(Node) {
  return id_ < rhs.id_;
}

NodeId Node::GetId() const {
  return id_;
}

Node &Node::Packing(const BoxPtr &box) {
  this->box_ = box;
  return *this;
}

Node &Node::AddSubgraph(const Graph &graph) {
  subgraphs_.push_back(&graph);
  return *this;
}

void Node::Accept(GraphVisitor &visitor) const {
  std::for_each(subgraphs_.begin(), subgraphs_.end(), [&visitor](const auto &graph) { visitor.Visit(*graph); });
}

EG_NS_END
