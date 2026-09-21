/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H46D07001_D54E_497C_B1BA_878A47164DA5
#define H46D07001_D54E_497C_B1BA_878A47164DA5

#include "easy_graph/builder/graph_builder.h"
#include "easy_graph/builder/chain_builder.h"
#include "easy_graph/builder/box_builder.h"
#include "easy_graph/infra/macro_traits.h"

EG_NS_BEGIN

////////////////////////////////////////////////////////////////
namespace detail {
template <typename GRAPH_BUILDER>
Graph BuildGraph(const char *name, GRAPH_BUILDER builderInDSL) {
  GraphBuilder builder(name);
  builderInDSL(builder);
  return std::move(*builder);
}

struct GraphDefiner {
  GraphDefiner(const char *defaultName, const char *specifiedName = nullptr) {
    name = specifiedName ? specifiedName : defaultName;
  }

  template <typename USER_BUILDER>
  auto operator|(USER_BUILDER &&userBuilder) {
    GraphBuilder graphBuilder{name};
    std::forward<USER_BUILDER>(userBuilder)(graphBuilder);
    return *graphBuilder;
  }

 private:
  const char *name;
};

inline void AddOutput(GraphBuilder &builder, Node &&node, int32_t index) {
  builder.BuildNode(node);
  builder->AddOutputAnchor(node.GetId(), index);
};

inline void AddTarget(GraphBuilder &builder, Node &&node) {
  builder.BuildNode(node);
  builder->AddTarget(node.GetId());
};
}  // namespace detail

#define DEF_GRAPH(G, ...) ::EG_NS::Graph G = ::EG_NS::detail::GraphDefiner(#G, ##__VA_ARGS__) | [&](auto &&BUILDER)
#define DATA_CHAIN(...) ::EG_NS::ChainBuilder(BUILDER, ::EG_NS::EdgeType::DATA)->__VA_ARGS__
#define CTRL_CHAIN(...) ::EG_NS::ChainBuilder(BUILDER, ::EG_NS::EdgeType::CTRL)->__VA_ARGS__
#define CHAIN(...) DATA_CHAIN(__VA_ARGS__)
#define ADD_OUTPUT(node, index) ::EG_NS::detail::AddOutput(BUILDER, ::GE_NS::OpDescNodeBuild(node), index)
#define ADD_TARGET(node)  ::EG_NS::detail::AddTarget(BUILDER, ::GE_NS::OpDescNodeBuild(node))

EG_NS_END

#endif
