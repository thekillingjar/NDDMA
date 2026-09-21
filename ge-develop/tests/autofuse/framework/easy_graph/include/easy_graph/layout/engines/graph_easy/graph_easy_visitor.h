/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HB6783151_C24E_4DA3_B969_46C2298FF43F
#define HB6783151_C24E_4DA3_B969_46C2298FF43F

#include <string>
#include "easy_graph/graph/graph_visitor.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_layout_context.h"

EG_NS_BEGIN

struct GraphEasyOption;

struct GraphEasyVisitor : GraphVisitor {
  GraphEasyVisitor(const GraphEasyOption &);

  std::string GetLayout() const;

 private:
  Status Visit(const Graph &) override;
  Status Visit(const Node &) override;
  Status Visit(const Edge &) override;

 private:
  std::string layout_;
  GraphEasyLayoutContext ctxt_;
};

EG_NS_END

#endif
