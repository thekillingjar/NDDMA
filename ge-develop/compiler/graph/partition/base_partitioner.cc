/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/base_partitioner.h"
#include <vector>
#include <queue>
#include "graph/utils/graph_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"

namespace ge {
using Cluster = BaseCluster;
using ClusterPtr = std::shared_ptr<Cluster>;

void BasePartitioner::DumpGraph(const std::string &suffix) const {
  GraphUtils::DumpGEGraphToOnnx(*root_graph_, suffix + "_root_graph");
  uint32_t graph_id = 0U;
  for (const auto &sub_graph : root_graph_->GetAllSubgraphs()) {
    GraphUtils::DumpGEGraphToOnnx(*sub_graph, suffix + "_subgraph" + std::to_string(graph_id));
    graph_id++;
  }
}

std::vector<std::shared_ptr<BaseCluster>> BasePartitioner::GetUniqueClusters(
    const ClusterFilter &cluster_filter) {
  std::vector<ClusterPtr> unique_cluster;
  std::unordered_set<ClusterPtr> seen_clusters;
  for (auto &node : root_graph_->GetDirectNode()) {
    auto &cluster = node_2_cluster_[node];
    if (!seen_clusters.insert(cluster).second) {
      continue;
    }
    if (cluster_filter == nullptr || cluster_filter(cluster)) {
      unique_cluster.push_back(cluster);
    }
  }
  return unique_cluster;
}

Status BasePartitioner::TopologicalSortClusters(const ClusterFilter &cluster_filter) {
  ordered_cluster_.clear();
  // BFS topological sort clusters for known shape cluster
  std::queue<ClusterPtr> ready_clusters;
  std::unordered_map<ClusterPtr, size_t> cluster_pending_count;
  std::unordered_set<ClusterPtr> seen_clusters;
  for (auto &node : root_graph_->GetDirectNode()) {
    auto &cluster = node_2_cluster_[node];
    if (seen_clusters.count(cluster) != 0) {
      continue;
    }
    seen_clusters.insert(cluster);
    auto pending_count = cluster->Inputs().size();
    if (pending_count == 0) {
      ready_clusters.push(cluster);
    } else {
      cluster_pending_count[cluster] = pending_count;
    }
  }

  size_t rank = 0;
  while (!ready_clusters.empty()) {
    auto cluster = ready_clusters.front();
    ready_clusters.pop();
    cluster->UpdateRank(rank++);
    if (cluster_filter == nullptr || cluster_filter(cluster)) {
      ordered_cluster_.push_back(cluster);
    }
    for (const auto &out_cluster : cluster->Outputs()) {
      if ((cluster_pending_count[out_cluster->shared_from_this()] > 0) &&
          (--cluster_pending_count[out_cluster->shared_from_this()] == 0)) {
        ready_clusters.push(out_cluster->shared_from_this());
      }
    }
  }
  GE_ASSERT_TRUE(rank == seen_clusters.size(), "[Check][Rank] failed, rank = %zu, seen cluster size = %zu.", rank,
                 seen_clusters.size());
  return SUCCESS;
}

Status BasePartitioner::BuildPartitionFrame() {
  for (const auto &cluster : sorted_unique_clusters_) {
    GE_CHK_STATUS_RET(cluster->BuildFrame(), "[Build][Frame] of cluster[%lu] failed.", cluster->Id());
  }
  return SUCCESS;
}

Status BasePartitioner::CombinePartitionFrame() const {
  for (const auto &cluster : sorted_unique_clusters_) {
    GE_CHK_STATUS_RET(cluster->CombinePartitionFrame(), "[Combine][Frame] of cluster[%lu] failed.", cluster->Id());
  }
  return SUCCESS;
}

Status BasePartitioner::BuildPartitionSubgraph() {
  for (const auto &cluster : sorted_unique_clusters_) {
    GE_CHK_STATUS_RET(cluster->BuildPartitionSubgraph(), "[Build][SubGraph] of cluster[%lu] failed.", cluster->Id());
  }
  return SUCCESS;
}

void BasePartitioner::ClearResource() {
  for (const auto &cluster : unique_clusters_) {
    cluster->Clear();
  }
  node_2_cluster_.clear();
  ordered_cluster_.clear();
  unique_clusters_.clear();
  sorted_unique_clusters_.clear();
  type_index_to_type_.clear();
  root_graph_.reset();
}

Status BasePartitioner::PartitionImpl() {
  GE_CHK_STATUS_RET(InitClusters(), "[Call][InitClusters] failed, graph:%s.", root_graph_->GetName().c_str());
  GE_CHK_STATUS_RET(MergeClusters(), "[Call][MergeClusters] failed, graph:%s.", root_graph_->GetName().c_str());
  GE_CHK_STATUS_RET(ProcessUniqueClusters(), "[Call][ProcessUniqueClusters] failed, graph:%s.",
                    root_graph_->GetName().c_str());
  GE_CHK_STATUS_RET(BuildPartitionFrame(), "[Call][BuildPartitionFrameNoLinkParent] failed, graph:%s.",
                    root_graph_->GetName().c_str());
  GE_CHK_STATUS_RET(CombinePartitionFrame(), "[Call][CombinePartitionFrame] failed, graph:%s.",
                    root_graph_->GetName().c_str());
  GE_CHK_STATUS_RET(BuildPartitionSubgraph(), "[Call][BuildPartitionSubgraph] failed, graph:%s.",
                    root_graph_->GetName().c_str());
  return SUCCESS;
}

void BasePartitioner::LogDebugString(bool is_failed) const {
  if ((!is_failed) && (!IsLogEnable(GE_MODULE_NAME, DLOG_INFO))) {
    return;
  }
  std::vector<size_t> cluster_num(type_index_to_type_.size(), 0);
  std::stringstream ss;
  for (const auto &cluster : unique_clusters_) {
    ++cluster_num[cluster->GetTypeIndex()];
  }
  ss << "All clusters:" << unique_clusters_.size();
  for (const auto &iter : type_index_to_type_) {
    ss << "," << iter.second << ":" << cluster_num[iter.first];
  }
  if (is_failed) {
    GELOGE(FAILED, "%s.", ss.str().c_str());
  } else {
    GELOGI("%s.", ss.str().c_str());
  }

  for (const auto &cluster : unique_clusters_) {
    if (is_failed) {
      GELOGE(FAILED, "[%s] %s.", GetPartitionName().c_str(), cluster->DebugString().c_str());
    } else {
      GELOGI("[%s] %s.", GetPartitionName().c_str(), cluster->DebugString().c_str());
    }
  }
}

Status BasePartitioner::CheckWritableVarNode(const ComputeGraphPtr &root_graph) {
  constexpr int32_t kInvalidOutIndex = -1;
  for (const auto &node : root_graph->GetDirectNode()) {
    if (node->GetType() == VARIABLE) {
      const auto &op_desc = node->GetOpDesc();
      const auto &input_name_index = op_desc->GetAllInputName();
      for (const auto &name_index : input_name_index) {
        const int32_t out_index = op_desc->GetOutputIndexByName(name_index.first);
        GE_ASSERT_TRUE(out_index == kInvalidOutIndex, "[Check][Variable] node %s is ref of index %d.",
                       node->GetName().c_str(), out_index);
      }
    }
  }
  return SUCCESS;
}

Status BasePartitioner::SortUniqueClusters(const ClusterCompareFunc &compare_func) {
  for (auto &node : GetRootGraph()->GetDirectNode()) {
    const auto &cluster = GetCluster(node);
    if (unique_clusters_.count(cluster) != 0U) {
      continue;
    }
    if (unique_clusters_.insert(cluster).second) {
      sorted_unique_clusters_.emplace_back(cluster);
    }
  }
  std::sort(sorted_unique_clusters_.begin(), sorted_unique_clusters_.end(), compare_func);
  return SUCCESS;
}

void BasePartitioner::InsertIndexType(const int32_t index, const std::string &type) {
  type_index_to_type_.emplace(index, type);
}
}  // namespace ge

