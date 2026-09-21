/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "asc_graph_dumper_context.h"
#include "ascir_utils.h"
namespace ascir {
AscGraphDumperContext &AscGraphDumperContext::GetThreadLocalCtx() {
  static thread_local AscGraphDumperContext ctx;
  return ctx;
}
void AscGraphDumperContext::ClearAllWatchGraphs() {
  orderd_graphs_.clear();
  watched_graphs_.clear();
}
void AscGraphDumperContext::AddWatchGraph(const string &suffix, const ge::AscGraph &graph) {
  const auto iter = watched_graphs_.find(&graph);
  if (iter == watched_graphs_.end()) {
    orderd_graphs_.emplace_back(suffix, graph);
    watched_graphs_.emplace(&graph, std::prev(orderd_graphs_.end()));
  } else {
    GELOGD("Update graph %s with suffix from %s to %s", graph.GetName().c_str(), iter->second->first.c_str(),
           suffix.c_str());
    iter->second->first = suffix;  // 更新持有相同graph对象的最后一个suffix
    if (std::next(iter->second) != orderd_graphs_.end()) {
      orderd_graphs_.splice(orderd_graphs_.cend(), orderd_graphs_, iter->second);
    }
  }
}
void AscGraphDumperContext::DumpWatchedGraphs() {
  for (const auto &suffix_2_graph : orderd_graphs_) {
    utils::AlwaysDumpGraph(suffix_2_graph.second, suffix_2_graph.first);
  }
}
}  // namespace ascir