/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H5FED5F58_167D_4536_918A_D5FE8F28DD9C
#define H5FED5F58_167D_4536_918A_D5FE8F28DD9C

#include "easy_graph/graph/graph.h"

EG_NS_BEGIN

struct Link;

struct GraphBuilder {
  GraphBuilder(const std::string &name);

  Node *BuildNode(const Node &);
  Edge *BuildEdge(const Node &src, const Node &dst, const Link &);

  Graph &operator*() {
    return graph_;
  }

  const Graph &operator*() const {
    return graph_;
  }

  Graph *operator->() {
    return &graph_;
  }

  const Graph *operator->() const {
    return &graph_;
  }

 private:
  struct NodeInfo {
    PortId inPortMax{0};
    PortId outPortMax{0};
  };

  NodeInfo *FindNode(const NodeId &);
  const NodeInfo *FindNode(const NodeId &) const;

 private:
  std::map<NodeId, NodeInfo> nodes_;
  Graph graph_;
};

EG_NS_END

#endif
