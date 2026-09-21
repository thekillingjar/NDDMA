/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_GRAPH_DSL_INCLUDE_GE_GRAPH_DSL_VISITOR_EXECUTE_SUBGRAPH_VISITOR_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_GRAPH_DSL_INCLUDE_GE_GRAPH_DSL_VISITOR_EXECUTE_SUBGRAPH_VISITOR_H_

#include "ge_graph_dsl/vistor/execute_graph_visitor.h"
#include "easy_graph/infra/status.h"
#include "ge_graph_dsl/ge.h"
#include "graph/fast_graph/execute_graph.h"

GE_NS_BEGIN

struct ExecuteSubgraphVisitor : ::EG_NS::GraphVisitor {
  ExecuteSubgraphVisitor(ExecuteGraphPtr &root_graph, const ::EG_NS::Node &node);
  ::EG_NS::Status BuildGraphRelations();

 private:
  ::EG_NS::Status Visit(const ::EG_NS::Graph &) override;
  ::EG_NS::Status Visit(const ::EG_NS::Node &) override;
  ::EG_NS::Status Visit(const ::EG_NS::Edge &) override;

 private:
  ::EG_NS::Status BuildGraphRelations(OpDescPtr &);

 private:
  ExecuteGraphPtr &root_graph_;
  const ::EG_NS::Node &node_;
  ExecuteGraphVisitor cur_graph_visitor_;
  std::vector<ExecuteGraphPtr> subgraphs_;
};

GE_NS_END

#endif /* AIR_CXX_TESTS_FRAMEWORK_GE_GRAPH_DSL_INCLUDE_GE_GRAPH_DSL_VISITOR_EXECUTE_SUBGRAPH_VISITOR_H_ */
