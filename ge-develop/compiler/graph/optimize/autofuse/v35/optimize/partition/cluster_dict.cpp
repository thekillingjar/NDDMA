/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cluster_dict.h"
namespace optimize {
void ClusterDict::AddCluster(const ClusterPtr &cluster) {
  clusters_.emplace_back(cluster);
}

void ClusterDict::SetNodeClusterPair(const ge::NodePtr &node, const ClusterPtr &cluster) {
  nodes_2_cluster_[node] = cluster;
}

ClusterPtr ClusterDict::GetNodeCluster(const ge::NodePtr &node) const {
  const auto &pair = nodes_2_cluster_.find(node);
  if (pair == nodes_2_cluster_.cend()) {
    return nullptr;
  }
  return pair->second;
}
}  // namespace optimize