/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HF900DC04_D202_42ED_992A_35DD7C940CE6
#define HF900DC04_D202_42ED_992A_35DD7C940CE6

#include "easy_graph/infra/status.h"
#include "graph/gnode.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/vistor/ge_graph_vistor.h"

GE_NS_BEGIN

struct GeSubgraphVisitor : ::EG_NS::GraphVisitor {
  GeSubgraphVisitor(ComputeGraphPtr &, const ::EG_NS::Node &);
  ::EG_NS::Status BuildGraphRelations();

 private:
  ::EG_NS::Status Visit(const ::EG_NS::Graph &) override;
  ::EG_NS::Status Visit(const ::EG_NS::Node &) override;
  ::EG_NS::Status Visit(const ::EG_NS::Edge &) override;

 private:
  ::EG_NS::Status BuildGraphRelations(OpDescPtr &);

 private:
  ComputeGraphPtr &root_graph_;
  const ::EG_NS::Node &node_;
  GeGraphVisitor cur_graph_vistor_;
  std::vector<ComputeGraphPtr> subgraphs_;
};

GE_NS_END

#endif /* HF900DC04_D202_42ED_992A_35DD7C940CE6 */
