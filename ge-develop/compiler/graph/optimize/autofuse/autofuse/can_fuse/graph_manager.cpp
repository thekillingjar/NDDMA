
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_manager.h"
#include <iostream>
#include <stack>
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils.h"
#include "utils/autofuse_utils.h"

namespace ge {
Status FusionGraphManager::AddMergeGraphMap(const std::string &new_node, const std::string &node1,
                                            const std::string &node2) {
  merge_graph_to_subgraphs_[new_node] = std::make_pair(node1, node2);
  GELOGD("add merged graph map: new_node: %s, node1: %s, node2: %s", new_node.c_str(), node1.c_str(), node2.c_str());
  return SUCCESS;
}

Status FusionGraphManager::CacheGraph(const std::string &graph_name, const ComputeGraphPtr &graph) {
  GELOGD("cache graph: %s.", graph_name.c_str());
  ComputeGraphPtr cache_graph;
  GE_ASSERT_SUCCESS(AutofuseUtils::CreateComputeGraphWithGraphID(graph, graph_name, cache_graph));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::CopyComputeGraph(graph, cache_graph));
  cache_graph->CopyAttrsFrom(*graph);
  name_to_graph_[graph_name] = cache_graph;
  return SUCCESS;
}

void FusionGraphManager::CacheCurrentGraphName(const std::string &graph_name1, const std::string &graph_name2) {
  GELOGI("cache current graph name: graph_name1:%s, graph_name2:%s.", graph_name1.c_str(), graph_name2.c_str());
  current_merging_graphs_.clear();
  current_merging_graphs_.push_back(graph_name1);
  current_merging_graphs_.push_back(graph_name2);
}

void FusionGraphManager::CacheCurrentGraphName(const std::string &graph_name1, const std::string &graph_name2,
                                               const std::string &origin_graph_name) {
  GELOGI("cache current graph name: graph_name1:%s, graph_name2:%s, origin_graph_name:%s.", graph_name1.c_str(),
         graph_name2.c_str(), origin_graph_name.c_str());
  current_merging_graphs_.clear();
  current_merging_graphs_.push_back(graph_name1);
  current_merging_graphs_.push_back(graph_name2);
  current_merging_graphs_.push_back(origin_graph_name);
}

ComputeGraphPtr FusionGraphManager::GetCachedGraph(const std::string &graph_name) const {
  auto it = name_to_graph_.find(graph_name);
  if (it != name_to_graph_.end()) {
    return it->second;
  }
  return nullptr;
}

Status FusionGraphManager::DumpGraph(const std::string &graph_name, const std::string &path,
                                     const std::string &suffix) const {
  auto graph = GetCachedGraph(graph_name);
  if (graph != nullptr) {
    AutofuseUtils::DumpGraphToOnnx(*graph, path, graph_name + suffix);
  } else {
    GELOGW("dump cache graph(%s) failed, it not in graph manager (cache graph first).", graph_name.c_str());
  }
  return SUCCESS;
}

Status FusionGraphManager::DumpGraphAndSubgraphs(const std::vector<std::string> &target_graphs,
                                                 const std::string &path) const {
  std::vector<std::pair<std::string, std::string>> dump_order;
  std::unordered_set<std::string> visited;

  // 使用双端队列+批量处理优化缓存利用率
  std::deque<std::string> processing_queue;

  // 第一阶段：直接处理目标图（保证最先dump origin graph）
  if (target_graphs.size() == 1U) {
    // 只有一张输入图时是融合后或者后处理的图，在merge_graph_to_subgraphs_.find后dump，非融合后的图不dump
    processing_queue.push_back(target_graphs[0]);
  } else {
    for (const auto &graph : target_graphs) {
      dump_order.push_back(std::make_pair(graph, ""));
      processing_queue.push_back(graph);
    }
  }

  // 第二阶段：批量处理子图（BFS顺序）
  while (!processing_queue.empty()) {
    // 批量处理一层的节点（优化缓存局部性）
    size_t level_size = processing_queue.size();
    for (size_t i = 0U; i < level_size; ++i) {
      const auto &current_graph = processing_queue.front();
      const auto it = merge_graph_to_subgraphs_.find(current_graph);
      if (it != merge_graph_to_subgraphs_.end()) {
        const auto &subgraph1 = it->second.first;
        const auto &subgraph2 = it->second.second;
        dump_order.push_back(std::make_pair(current_graph, "_merged_by_" + subgraph1 + "_and_" + subgraph2));
        dump_order.push_back(std::make_pair(subgraph1, ""));
        processing_queue.push_back(subgraph1);
        dump_order.push_back(std::make_pair(subgraph2, ""));
        processing_queue.push_back(subgraph2);
      } else {
        GELOGD("can't find cache graph(%s), it not in merged graph map keys.", current_graph.c_str());
      }
      processing_queue.pop_front();  // 批量处理完成后统一pop
    }
  }

  for (const auto &name : dump_order) {
    GE_ASSERT_SUCCESS(DumpGraph(name.first, path, name.second));
  }
  return SUCCESS;
}

Status FusionGraphManager::DumpCurrentGraphAndSubgraphs(const std::string &path) {
  if (current_merging_graphs_.size() == kCurrentSubGraphNum) {  // 需要dump merged graph场景(FusedAscBackend场景)
    std::vector<std::string> origin_graphs;
    origin_graphs.push_back(current_merging_graphs_.back());
    current_merging_graphs_.pop_back();
    GE_ASSERT_SUCCESS(DumpGraphAndSubgraphs(current_merging_graphs_, path));
    GE_ASSERT_SUCCESS(DumpGraphAndSubgraphs(origin_graphs, path));
  } else {
    GE_ASSERT_SUCCESS(DumpGraphAndSubgraphs(current_merging_graphs_, path));
  }
  return SUCCESS;
}
}  // namespace ge