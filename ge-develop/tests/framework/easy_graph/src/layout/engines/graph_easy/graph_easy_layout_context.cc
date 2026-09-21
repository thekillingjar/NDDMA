/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "easy_graph/layout/engines/graph_easy/graph_easy_layout_context.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"
#include "easy_graph/graph/graph.h"

EG_NS_BEGIN

GraphEasyLayoutContext::GraphEasyLayoutContext(const GraphEasyOption &options) : options_(options) {}

const Graph *GraphEasyLayoutContext::GetCurrentGraph() const {
  if (graphs_.empty())
    return nullptr;
  return graphs_.back();
}

void GraphEasyLayoutContext::EnterGraph(const Graph &graph) {
  graphs_.push_back(&graph);
}

void GraphEasyLayoutContext::ExitGraph() {
  graphs_.pop_back();
}

void GraphEasyLayoutContext::LinkBegin() {
  is_linking_ = true;
}

void GraphEasyLayoutContext::LinkEnd() {
  is_linking_ = false;
}

bool GraphEasyLayoutContext::InLinking() const {
  return is_linking_;
}

std::string GraphEasyLayoutContext::GetGroupPath() const {
  if (graphs_.empty())
    return "";
  std::string result("");
  std::for_each(graphs_.begin(), graphs_.end(),
                [&result](const auto &graph) { result += (std::string("/") + graph->GetName()); });
  return (result + "/");
}

const GraphEasyOption &GraphEasyLayoutContext::GetOptions() const {
  return options_;
}

EG_NS_END
