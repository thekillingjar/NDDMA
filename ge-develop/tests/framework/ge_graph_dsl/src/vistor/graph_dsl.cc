/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_graph/builder/graph_dsl.h"
#include "easy_graph/graph/graph.h"
#include "graph/graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/vistor/ge_graph_vistor.h"
#include "ge_graph_dsl/vistor/execute_graph_visitor.h"

GE_NS_BEGIN

Graph ToGeGraph(const ::EG_NS::Graph &graph) {
  graph.Layout();
  GeGraphVisitor geVistor;
  graph.Accept(geVistor);
  return geVistor.BuildGeGraph();
}

ComputeGraphPtr ToComputeGraph(const ::EG_NS::Graph &graph) {
  graph.Layout();
  GeGraphVisitor geVistor;
  graph.Accept(geVistor);
  return geVistor.BuildComputeGraph();
}

ExecuteGraphPtr ToExecuteGraph(const ::EG_NS::Graph &graph) {
  graph.Layout();
  ExecuteGraphVisitor execute_graph_visitor;
  graph.Accept(execute_graph_visitor);
  return execute_graph_visitor.BuildExecuteGraph();
}

GE_NS_END
