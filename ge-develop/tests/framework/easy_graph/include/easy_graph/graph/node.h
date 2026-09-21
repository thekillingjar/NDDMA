/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HF37ACE88_F726_4AA3_8599_ED7A888AA623
#define HF37ACE88_F726_4AA3_8599_ED7A888AA623

#include <vector>
#include "easy_graph/graph/node_id.h"
#include "easy_graph/infra/operator.h"
#include "easy_graph/infra/ext_traits.h"
#include "easy_graph/graph/box.h"

EG_NS_BEGIN

struct GraphVisitor;
struct Graph;

struct Node {
  template<typename... GRAPHS, SUBGRAPH_CONCEPT(GRAPHS, Graph)>
  Node(const NodeId &id, const GRAPHS &... graphs) : id_(id), subgraphs_{&graphs...} {}

  template<typename... GRAPHS, SUBGRAPH_CONCEPT(GRAPHS, Graph)>
  Node(const NodeId &id, const BoxPtr &box, const GRAPHS &... graphs) : id_(id), box_(box), subgraphs_{&graphs...} {}

  __DECL_COMP(Node);

  NodeId GetId() const;

  Node &Packing(const BoxPtr &);

  template<typename Anything>
  Anything *Unpacking() const {
    if (!box_)
      return nullptr;
    return BoxUnpacking<Anything>(box_);
  }

  Node &AddSubgraph(const Graph &);
  void Accept(GraphVisitor &) const;

 private:
  NodeId id_;
  BoxPtr box_;
  std::vector<const Graph *> subgraphs_;
};

EG_NS_END

#endif
