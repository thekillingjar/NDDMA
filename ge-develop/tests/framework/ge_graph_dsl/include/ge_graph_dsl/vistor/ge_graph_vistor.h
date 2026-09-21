/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HD31125D4_0EB8_494C_B83D_3B8B923A914D
#define HD31125D4_0EB8_494C_B83D_3B8B923A914D

#include "easy_graph/graph/graph_visitor.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "ge_graph_dsl/ge.h"

GE_NS_BEGIN

struct GeGraphVisitor : ::EG_NS::GraphVisitor {
  GeGraphVisitor();
  void reset(const ComputeGraphPtr &graph);
  Graph BuildGeGraph() const;
  ComputeGraphPtr BuildComputeGraph() const;

 private:
  ::EG_NS::Status Visit(const ::EG_NS::Graph &) override;
  ::EG_NS::Status Visit(const ::EG_NS::Node &) override;
  ::EG_NS::Status Visit(const ::EG_NS::Edge &) override;

 private:
  ComputeGraphPtr build_graph_;
  std::vector<std::pair<eg::NodeId, int32_t>> outputs_;
  std::vector<eg::NodeId> targets_;
};

GE_NS_END

#endif /* TESTS_ST_EASY_GRAPH_GELAYOUT_GRAPH_GE_VISTOR_H_ */
