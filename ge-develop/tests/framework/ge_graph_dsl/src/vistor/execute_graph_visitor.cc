/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_graph_dsl/vistor/execute_graph_visitor.h"

#include "easy_graph/graph/edge.h"
#include "easy_graph/graph/graph.h"
#include "easy_graph/graph/node.h"
#include "easy_graph/graph/edge_type.h"
#include "easy_graph/builder/box_builder.h"

#include "ge_graph_dsl/op_desc/op_box.h"
#include "ge_graph_dsl/vistor/execute_sugraph_visitor.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/fast_node_utils.h"

using ::EG_NS::Status;

GE_NS_BEGIN

ExecuteGraphVisitor::ExecuteGraphVisitor() : build_graph_(std::make_shared<ExecuteGraph>("")) {}

void ExecuteGraphVisitor::reset(const ExecuteGraphPtr &graph) { build_graph_ = graph; }

ExecuteGraphPtr ExecuteGraphVisitor::BuildExecuteGraph() const { return build_graph_; }

Status ExecuteGraphVisitor::Visit(const ::EG_NS::Graph &graph) {
  build_graph_->SetName(graph.GetName());
  return EG_SUCCESS;
}

Status ExecuteGraphVisitor::Visit(const ::EG_NS::Node &node) {
  ExecuteSubgraphVisitor visitor(build_graph_, node);
  return visitor.BuildGraphRelations();
}

Status ExecuteGraphVisitor::Visit(const ::EG_NS::Edge &edge) {
  // FindNodeFromAllNodes是在根图的所有节点上查找，如果是子图这里存在问题
  auto src_node = ExecuteGraphUtils::FindNodeFromAllNodes(build_graph_.get(), edge.GetSrc().getNodeId().c_str());
  auto dst_node = ExecuteGraphUtils::FindNodeFromAllNodes(build_graph_.get(), edge.GetDst().getNodeId().c_str());

  if (edge.GetType() == ::EG_NS::EdgeType::CTRL) {
    build_graph_->AddEdge(src_node, kControlEdgeIndex, dst_node, kControlEdgeIndex);
    return EG_SUCCESS;
  }

  if (src_node->GetDataOutNum() <= edge.GetSrc().getPortId()) {
    FastNodeUtils::AppendOutputEdgeInfo(src_node, edge.GetSrc().getPortId() + 1U);
  }
  if (dst_node->GetDataInNum() <= edge.GetDst().getPortId()) {
    FastNodeUtils::AppendInputEdgeInfo(dst_node, edge.GetDst().getPortId() + 1U);
  }

  if ((src_node->GetDataOutNum() <= edge.GetSrc().getPortId()) ||
      (dst_node->GetDataInNum() <= edge.GetDst().getPortId())) {
    return EG_FAILURE;
  }
  build_graph_->AddEdge(src_node, edge.GetSrc().getPortId(), dst_node, edge.GetDst().getPortId());
  return EG_SUCCESS;
}

GE_NS_END
