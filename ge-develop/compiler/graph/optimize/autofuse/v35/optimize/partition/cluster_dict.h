/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_PARTITION_CLUSTER_DICT_H
#define OPTIMIZE_PLATFORM_V2_PARTITION_CLUSTER_DICT_H

#include "graph/node.h"
#include "cluster.h"

namespace optimize {
// who manage partitioner clusters map
class ClusterDict {
 public:
  explicit ClusterDict() = default;
  ~ClusterDict() = default;
  void AddCluster(const ClusterPtr &cluster);
  void SetNodeClusterPair(const ge::NodePtr &node, const ClusterPtr &cluster);
  ClusterPtr GetNodeCluster(const ge::NodePtr &node) const;
  const std::vector<ClusterPtr> &GetAllClusters() const {
    return clusters_;
  }
  void SwapClusters(std::vector<ClusterPtr> &swap_clusters) {
    clusters_.swap(swap_clusters);
  }

 private:
  // defined for clusters merge
  std::vector<ClusterPtr> clusters_;
  std::unordered_map<ge::NodePtr, ClusterPtr> nodes_2_cluster_;
};
}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_V2_PARTITION_CLUSTER_DICT_H
