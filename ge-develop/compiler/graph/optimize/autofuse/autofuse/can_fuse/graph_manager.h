/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_BACKEND_GRAPH_MANAGER_H
#define AUTOFUSE_CAN_FUSE_BACKEND_GRAPH_MANAGER_H

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>
#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"

namespace ge {
const uint32_t kCurrentSubGraphNum = 3U; // FusedAscBackend子图融合场景除了dump node1和node2的子图还有FusedAscBackend的子图
class FusionGraphManager {
 public:
  // 获取单例实例
  static FusionGraphManager &GetInstance() {
    thread_local static FusionGraphManager instance;
    return instance;
  }

  // 删除拷贝构造函数和赋值操作符
  FusionGraphManager(const FusionGraphManager &) = delete;
  FusionGraphManager &operator=(const FusionGraphManager &) = delete;

  // 添加融合图和其子图的映射关系
  Status AddMergeGraphMap(const std::string &new_node, const std::string &node1, const std::string &node2);

  // 缓存图名称和图的映射
  Status CacheGraph(const std::string &graph_name, const ComputeGraphPtr &graph);
  void CacheCurrentGraphName(const std::string &graph_name1, const std::string &graph_name2);
  void CacheCurrentGraphName(const std::string &graph_name1, const std::string &graph_name2,
                             const std::string &origin_graph_name);
  // 获取缓存的图
  ComputeGraphPtr GetCachedGraph(const std::string &graph_name) const;

  // dump单个图
  Status DumpGraph(const std::string &graph_name, const std::string &path, const std::string &suffix) const;

  // dump图以及融合前的子图
  Status DumpGraphAndSubgraphs(const std::vector<std::string> &target_graphs, const std::string &path) const;

  // dump当前正在融合的图以及融合成当前图的所有子图
  Status DumpCurrentGraphAndSubgraphs(const std::string &path);  // Graph merging operations

 private:
  // 融合图和融合子图的映射
  std::unordered_map<std::string, std::pair<std::string, std::string>> merge_graph_to_subgraphs_;
  // 缓存图名字和图的映射
  std::unordered_map<std::string, ComputeGraphPtr> name_to_graph_;
  // 缓存当前正在处理的子图对
  std::vector<std::string> current_merging_graphs_;
  FusionGraphManager() = default;  // 私有构造函数
};
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_BACKEND_GRAPH_MANAGER_H