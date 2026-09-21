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
#include "easy_graph/graph/graph.h"
#include "easy_graph/graph/edge.h"
#include "ge_graph_dsl/op_desc/op_box.h"
#include "ge_graph_dsl/vistor/ge_subgraph_vistor.h"

GE_NS_BEGIN

GeSubgraphVisitor::GeSubgraphVisitor(ComputeGraphPtr &rootGraph, const ::EG_NS::Node &node)
    : root_graph_(rootGraph), node_(node) {}

::EG_NS::Status GeSubgraphVisitor::BuildGraphRelations() {
  node_.Accept(*this);
  auto nod_builder = node_.Unpacking<OpBox>();
  if (nod_builder == nullptr) {
    return EG_FAILURE;
  } else {
    auto opPtr = nod_builder->Build(node_.GetId());
    return BuildGraphRelations(opPtr);
  }
}

::EG_NS::Status GeSubgraphVisitor::BuildGraphRelations(OpDescPtr &opPtr) {
  auto node = root_graph_->AddNode(opPtr);
  int graph_index = 0;
  for (auto subGraph : subgraphs_) {
    opPtr->AddSubgraphName(subGraph->GetName());
    opPtr->SetSubgraphInstanceName(graph_index++, subGraph->GetName());
    subGraph->SetParentNode(node);
    subGraph->SetParentGraph(root_graph_);
    root_graph_->AddSubgraph(subGraph);
  }
  return EG_SUCCESS;
}

::EG_NS::Status GeSubgraphVisitor::Visit(const ::EG_NS::Graph &graph) {
  auto subgraph = std::make_shared<ComputeGraph>(graph.GetName());
  cur_graph_vistor_.reset(subgraph);
  graph.Accept(cur_graph_vistor_);
  subgraphs_.push_back(subgraph);
  return EG_SUCCESS;
}

::EG_NS::Status GeSubgraphVisitor::Visit(const ::EG_NS::Node &node) {
  ::EG_NS::GraphVisitor &vistor = cur_graph_vistor_;
  return vistor.Visit(node);
}

::EG_NS::Status GeSubgraphVisitor::Visit(const ::EG_NS::Edge &edge) {
  ::EG_NS::GraphVisitor &vistor = cur_graph_vistor_;
  return vistor.Visit(edge);
}

GE_NS_END
