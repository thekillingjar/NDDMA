/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H813EC8C1_3850_4320_8AC0_CE071C89B871
#define H813EC8C1_3850_4320_8AC0_CE071C89B871

#include "easy_graph/graph/node.h"

#include "easy_graph/graph/edge.h"
#include "easy_graph/infra/status.h"
#include <string>
#include <set>
#include <map>

EG_NS_BEGIN

struct GraphVisitor;
struct LayoutOption;

struct Graph {
  explicit Graph(const std::string &name);

  std::string GetName() const;

  Node *AddNode(const Node &);
  Edge *AddEdge(const Edge &);

  Node *FindNode(const NodeId &);
  const Node *FindNode(const NodeId &) const;

  std::pair<const Node *, const Node *> FindNodePair(const Edge &) const;
  std::pair<Node *, Node *> FindNodePair(const Edge &);

  void AddOutputAnchor(NodeId id, int32_t out_index);
  const std::vector<std::pair<NodeId, int32_t>> &GetOutputEndPoints() const;

  void AddTarget(NodeId id);
  const std::vector<NodeId> &GetTargets() const;

  void Accept(GraphVisitor &) const;

  Status Layout(const LayoutOption *option = nullptr) const;

 private:
  std::string name_;
  std::map<NodeId, Node> nodes_;
  std::set<Edge> edges_;
  std::vector<std::pair<NodeId, int32_t>> output_points_;
  std::vector<NodeId> targets_;
};

EG_NS_END

#endif
