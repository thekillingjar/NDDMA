/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_SHARE_GRAPH_H
#define METADEF_CXX_SHARE_GRAPH_H

#include "graph_builder_utils.h"

namespace ge {
template<typename GraphT, typename BuilderT>
class SharedGraph {
 public:
  static GraphT BuildGraphWithSubGraph();
  static GraphT BuildGraphWithConst();
  static GraphT BuildGraphWithControlEdge();

 private:
  static BuilderT CreateBuilder(const std::string &name) {
    return BuilderT(name);
  }
};

template<typename GraphT>
class BuilderUtils {
 public:
  static void SetParentGraph(GraphT &graph, GraphT &parent_graph) {
    graph->SetParentGraph(parent_graph);
  }
};

template<>
class BuilderUtils<ExecuteGraphPtr> {
 public:
  // ExecuteGraph requires passing a raw pointer, so explicitly specialize template
  static void SetParentGraph(ExecuteGraphPtr &graph, ExecuteGraphPtr &parent_graph) {
    graph->SetParentGraph(parent_graph.get());
  }
};

}  // namespace ge

#endif  //METADEF_CXX_SHARE_GRAPH_H
