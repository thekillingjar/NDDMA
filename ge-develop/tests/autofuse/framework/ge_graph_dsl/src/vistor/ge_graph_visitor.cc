/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_graph/graph/edge.h"
#include "easy_graph/graph/graph.h"
#include "easy_graph/graph/node.h"
#include "easy_graph/graph/edge_type.h"
#include "easy_graph/builder/box_builder.h"

#include "graph/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"

#include "common/ge_common/ge_types.h"
#include "ge_graph_dsl/op_desc/op_box.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "ge_graph_dsl/vistor/ge_graph_vistor.h"
#include "ge_graph_dsl/vistor/ge_subgraph_vistor.h"

using ::EG_NS::Status;

GE_NS_BEGIN

GeGraphVisitor::GeGraphVisitor() : build_graph_(std::make_shared<ComputeGraph>("")) {}

void GeGraphVisitor::reset(const ComputeGraphPtr &graph) { build_graph_ = graph; }

Graph GeGraphVisitor::BuildGeGraph() const {
  for (const auto &id_and_index : outputs_) {
    auto node = build_graph_->FindNode(id_and_index.first);
    build_graph_->AddOutputNodeByIndex(node, id_and_index.second);
  }
  std::vector<NodePtr> target_nodes;
  for (const auto &id : targets_) {
    target_nodes.emplace_back(build_graph_->FindNode(id));
  }
  build_graph_->SetGraphTargetNodesInfo(target_nodes);
  return GraphUtilsEx::CreateGraphFromComputeGraph(build_graph_);
}

ComputeGraphPtr GeGraphVisitor::BuildComputeGraph() const { return build_graph_; }

Status GeGraphVisitor::Visit(const ::EG_NS::Graph &graph) {
  build_graph_->SetName(graph.GetName());
  outputs_ = graph.GetOutputEndPoints();
  targets_ = graph.GetTargets();
  return EG_SUCCESS;
}

Status GeGraphVisitor::Visit(const ::EG_NS::Node &node) {
  GeSubgraphVisitor vistor(build_graph_, node);
  return vistor.BuildGraphRelations();
}

Status GeGraphVisitor::Visit(const ::EG_NS::Edge &edge) {
  auto src_node = build_graph_->FindNode(edge.GetSrc().getNodeId());
  auto dst_node = build_graph_->FindNode(edge.GetDst().getNodeId());

  if (edge.GetType() == ::EG_NS::EdgeType::CTRL) {
    GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
    return EG_SUCCESS;
  }

  if (src_node->GetAllOutDataAnchorsSize() <= edge.GetSrc().getPortId()) {
    NodeUtils::AppendOutputAnchor(src_node, edge.GetSrc().getPortId() + 1U);
  }
  if (dst_node->GetAllInDataAnchorsSize() <= edge.GetDst().getPortId()) {
    NodeUtils::AppendInputAnchor(dst_node, edge.GetDst().getPortId() + 1U);
  }

  if (src_node->GetAllOutDataAnchorsSize() <= edge.GetSrc().getPortId() ||
      dst_node->GetAllInDataAnchorsSize() <= edge.GetDst().getPortId()) {
    return EG_FAILURE;
  }
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(edge.GetSrc().getPortId()),
                      dst_node->GetInDataAnchor(edge.GetDst().getPortId()));
  return EG_SUCCESS;
}

GE_NS_END
