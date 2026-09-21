/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_graph/layout/graph_layout.h"
#include "easy_graph/layout/layout_executor.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_executor.h"
#include "easy_graph/graph/graph.h"

EG_NS_BEGIN

namespace {
GraphEasyExecutor default_executor;
}

void GraphLayout::Config(LayoutExecutor &executor, const LayoutOption *opts) {
  this->executor_ = &executor;
  options_ = opts;
}

Status GraphLayout::Layout(const Graph &graph, const LayoutOption *opts) {
  const LayoutOption *options = opts ? opts : this->options_;
  if (!executor_) return static_cast<LayoutExecutor &>(default_executor).Layout(graph, options);
  return executor_->Layout(graph, options);
}

EG_NS_END
