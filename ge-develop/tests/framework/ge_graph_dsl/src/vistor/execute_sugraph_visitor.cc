/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_graph_dsl/vistor/execute_sugraph_visitor.h"
#include "easy_graph/graph/node.h"
#include "easy_graph/graph/graph.h"
#include "easy_graph/graph/edge.h"
#include "ge_graph_dsl/op_desc/op_box.h"

GE_NS_BEGIN

ExecuteSubgraphVisitor::ExecuteSubgraphVisitor(ExecuteGraphPtr &root_graph, const ::EG_NS::Node &node)
    : root_graph_(root_graph), node_(node) {}

::EG_NS::Status ExecuteSubgraphVisitor::BuildGraphRelations() {
  node_.Accept(*this);
  auto nod_builder = node_.Unpacking<OpBox>();
  if (nod_builder == nullptr) {
    return EG_FAILURE;
  } else {
    auto op_desc = nod_builder->Build(node_.GetId());
    return BuildGraphRelations(op_desc);
  }
}

::EG_NS::Status ExecuteSubgraphVisitor::BuildGraphRelations(OpDescPtr &op_desc) {
  auto node = root_graph_->AddNode(op_desc);
  int graph_index = 0;
  for (auto subgraph : subgraphs_) {
    op_desc->AddSubgraphName(subgraph->GetName());
    op_desc->SetSubgraphInstanceName(graph_index++, subgraph->GetName());
    subgraph->SetParentNode(node);
    subgraph->SetParentGraph(root_graph_.get());
    root_graph_->AddSubGraph(subgraph);
  }
  return EG_SUCCESS;
}

::EG_NS::Status ExecuteSubgraphVisitor::Visit(const ::EG_NS::Graph &graph) {
  auto subgraph = std::make_shared<ExecuteGraph>(graph.GetName());
  cur_graph_visitor_.reset(subgraph);
  graph.Accept(cur_graph_visitor_);
  subgraphs_.push_back(subgraph);
  return EG_SUCCESS;
}

::EG_NS::Status ExecuteSubgraphVisitor::Visit(const ::EG_NS::Node &node) {
  ::EG_NS::GraphVisitor &visitor = cur_graph_visitor_;
  return visitor.Visit(node);
}

::EG_NS::Status ExecuteSubgraphVisitor::Visit(const ::EG_NS::Edge &edge) {
  ::EG_NS::GraphVisitor &visitor = cur_graph_visitor_;
  return visitor.Visit(edge);
}

GE_NS_END
