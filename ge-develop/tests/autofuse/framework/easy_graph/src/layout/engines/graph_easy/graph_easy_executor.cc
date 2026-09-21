/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_graph/layout/engines/graph_easy/graph_easy_executor.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_visitor.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"
#include "layout/engines/graph_easy/utils/shell_executor.h"
#include "easy_graph/layout/layout_option.h"
#include "easy_graph/graph/graph.h"

EG_NS_BEGIN

namespace {
const GraphEasyOption *GraphEasyOptionCast(const LayoutOption *opts) {
  if (!opts)
    return &(GraphEasyOption::GetDefault());
  auto options = dynamic_cast<const GraphEasyOption *>(opts);
  if (options)
    return options;
  return &(GraphEasyOption::GetDefault());
}
}  // namespace

Status GraphEasyExecutor::Layout(const Graph &graph, const LayoutOption *opts) {
  auto options = GraphEasyOptionCast(opts);
  GraphEasyVisitor visitor(*options);
  graph.Accept(visitor);

  std::string script =
      std::string("echo \"") + visitor.GetLayout() + "\" | graph-easy " + options->GetLayoutCmdArgs(graph.GetName());
  return ShellExecutor::execute(script);
}

EG_NS_END
