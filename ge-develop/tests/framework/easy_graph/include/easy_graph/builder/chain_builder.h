/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HDF50E564_F050_476A_A479_F82B20F35C84
#define HDF50E564_F050_476A_A479_F82B20F35C84

#include "easy_graph/builder/link.h"
#include "easy_graph/graph/node_id.h"
#include "easy_graph/graph/node.h"

EG_NS_BEGIN

struct GraphBuilder;
struct Graph;
struct Edge;

struct ChainBuilder {
  ChainBuilder(GraphBuilder &graphBuilder, EdgeType defaultEdgeType);

  struct LinkBuilder {
    using NodeObj = ::EG_NS::Node;
    using EdgeObj = ::EG_NS::Edge;

    LinkBuilder(ChainBuilder &chain, EdgeType defaultEdgeType);

    ChainBuilder &Node(const NodeObj &node);

    template<typename... PARAMS>
    ChainBuilder &Node(const NodeId &id, const PARAMS &... params) {
      auto node = chain_.FindNode(id);
      if (node) {
        return this->Node(*node);
      }
      return this->Node(NodeObj(id, params...));
    }

    ChainBuilder &Ctrl(const std::string &label = "");
    ChainBuilder &Data(const std::string &label = "");

    ChainBuilder &Data(PortId srcId = UNDEFINED_PORT_ID, PortId dstId = UNDEFINED_PORT_ID,
                       const std::string &label = "");

    ChainBuilder &Edge(EdgeType type, PortId srcId = UNDEFINED_PORT_ID, PortId dstId = UNDEFINED_PORT_ID,
                       const std::string &label = "");

   private:
    ChainBuilder &startLink(const Link &);

   private:
    ChainBuilder &chain_;
    EdgeType default_edge_type_;
    Link from_link_;
  } linker;

  LinkBuilder *operator->();

 private:
  ChainBuilder &LinkTo(const Node &, const Link &);
  const Node *FindNode(const NodeId &) const;

 private:
  Node *prev_node_{nullptr};
  GraphBuilder &graph_builder_;
};

EG_NS_END

#endif
