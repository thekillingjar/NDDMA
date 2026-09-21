/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PARTITION_BASE_PARTITION_H_
#define GE_GRAPH_PARTITION_BASE_PARTITION_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/partition/base_cluster.h"

namespace ge {
const char_t *const kClusterData = "DATA";
const char_t *const kClusterInputNode = "INPUT_NODE";
const char_t *const kClusterNetoutput = "NETOUTPUT";
const char_t *const kClusterStage = "STAGE";
// sequence increase
constexpr int32_t kDataTypeIndex = 0;
constexpr int32_t kNetOutputTypeIndex = 1;
constexpr int32_t kInputNodeTypeIndex = 2;

class BasePartitioner {
 public:
  friend class BaseCluster;
  friend class HeterogeneousCluster;
  friend class FlowCluster;
  friend class DynamicShapeCluster;
  using ClusterFilter = std::function<bool(const std::shared_ptr<BaseCluster> &cluster)>;
  using ClusterCompareFunc =
      std::function<bool(const std::shared_ptr<BaseCluster> &a, const std::shared_ptr<BaseCluster> &b)>;
  explicit BasePartitioner(ge::ComputeGraphPtr graph) : root_graph_(std::move(graph)) {}
  virtual ~BasePartitioner() = default;
  // Debug function
  virtual void LogDebugString(bool is_failed) const;

  virtual Status Partition() = 0;

  virtual std::string GetSubgraphName(const BaseCluster &cluster) const = 0;

  virtual bool IsNeedBuildPartitionFrame(const BaseCluster &cluster) const = 0;

  virtual void ClearResource();

  void DumpGraph(const std::string &suffix) const;

  Status TopologicalSortClusters(const ClusterFilter &cluster_filter);
  std::vector<std::shared_ptr<BaseCluster>> GetUniqueClusters(const ClusterFilter &cluster_filter);
  Status PartitionImpl();
  // Establish the input-output anchors for each partition of the cluster and record links to other clusters
  virtual Status BuildPartitionFrame();
  // Establish connection between corresponding partitioned of clusters
  virtual Status CombinePartitionFrame() const;
  // Convert the nodes in cluster into a complete ComputeGraph
  virtual Status BuildPartitionSubgraph();
  virtual Status InitClusters() = 0;
  virtual Status MergeClusters() = 0;
  virtual Status ProcessUniqueClusters() = 0;
  virtual std::string GetPartitionName() const = 0;
  static Status CheckWritableVarNode(const ComputeGraphPtr &root_graph);
 protected:
  const ComputeGraphPtr &GetRootGraph() const { return root_graph_; }
  void SetRootGraph(const ComputeGraphPtr &root_graph) { root_graph_ = root_graph; }
  const std::shared_ptr<BaseCluster> &GetCluster(const NodePtr &node) { return node_2_cluster_[node]; }
  void SetCluster(const NodePtr &node, const shared_ptr<BaseCluster> &cluster) {
    node_2_cluster_[node] = cluster;
  }
  void RecordClusters(const shared_ptr<BaseCluster> &cluster) {
    clusters_.emplace(cluster);
  }

  void ClearClusters() {
    clusters_.clear();
    node_2_cluster_.clear();
    ordered_cluster_.clear();
    unique_clusters_.clear();
    sorted_unique_clusters_.clear();
    type_index_to_type_.clear();
  }
  // Sort unique cluster by it's topo order
  Status SortUniqueClusters(const ClusterCompareFunc &compare_func);
  const std::vector<std::shared_ptr<BaseCluster>> &GetOrderedClusters() const {
    return ordered_cluster_;
  }
  const std::unordered_set<std::shared_ptr<BaseCluster>> &GetUniqueClusters() const {
    return unique_clusters_;
  }
  const std::vector<std::shared_ptr<BaseCluster>> &GetSortedUniqueClusters() const {
    return sorted_unique_clusters_;
  }
  void InsertIndexType(const int32_t index, const std::string &type);
  std::string GetTypeByIndex(const int32_t index) const {
    const auto &it = type_index_to_type_.find(index);
    if (it == type_index_to_type_.end()) {
      return "";
    }
    return it->second;
  }

  std::vector<std::shared_ptr<BaseCluster>> sorted_unique_clusters_;

 private:
  ge::ComputeGraphPtr root_graph_;
  // Record nodes and the cluster it belongs to
  std::unordered_map<NodePtr, std::shared_ptr<BaseCluster>> node_2_cluster_;
  // Record all clusters to ensure that the lifecycle of cluster is long enough
  std::unordered_set<std::shared_ptr<BaseCluster>> clusters_;
  // topological sorted clusters, this field will change with the splitting.
  // When partitioning UNKNOWN_SHAPE cluster, it is a collection of all topological sorted UNKNOWN_SHAPE clusters
  // When partitioning KNOWN_SHAPE cluster, it is a collection of all topological sorted KNOWN_SHAPE clusters
  std::vector<std::shared_ptr<BaseCluster>> ordered_cluster_;
  // Unique clusters left after merged clusters
  std::unordered_set<std::shared_ptr<BaseCluster>> unique_clusters_;
  // Unique clusters left after merged clusters sorted by rank
  std::map<int32_t, std::string> type_index_to_type_;  // key:type index. value:cluster type, for debug
};
} // namespace ge

#endif // GE_GRAPH_PARTITION_BASE_PARTITION_H_
