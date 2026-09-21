/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PARTITION_DYNAMIC_SHAPE_PARTITION_H_
#define GE_GRAPH_PARTITION_DYNAMIC_SHAPE_PARTITION_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/partition/base_partitioner.h"

namespace ge {
class DynamicShapeCluster : public BaseCluster {
 public:
  DynamicShapeCluster(size_t rank, int32_t type_index, NodePtr node, BasePartitioner *partitioner)
      : BaseCluster(rank, type_index, std::move(node), partitioner) {}
  ~DynamicShapeCluster() override = default;
  Status BuildPartitionFrame() override;
  Status SetUnknownAttr();
  bool IsKnownShape() const;
  bool IsUnknownShape() const;
  Status SetAttrToNetoutput(const OpDescPtr &op) override;
  void Merge(std::shared_ptr<BaseCluster> other) override;
};

class DynamicShapePartitioner : public BasePartitioner {
 public:
  explicit DynamicShapePartitioner(ge::ComputeGraphPtr graph, const bool merge_known_first = false)
      : BasePartitioner(std::move(graph)), merge_known_first_(merge_known_first) {}
  ~DynamicShapePartitioner() override = default;

  Status Partition() override;

 private:
  // Collect nodes that satisfy the unknowshape rules:
  // 1) The Tensor shape of any input or output is unknow shape(dim_size = -1) or unknow rank(dim_size=-2)
  // 2) Subgraphs of the node has an operator that satisfies rule 1)
  Status MarkUnknownShapeNodes();
  // For each node a Cluster structure, and connected according to the connection relationship of the nodes
  // An cluster means set of nodes that can be merged in same partition,
  // Corresponding relationship between cluster type and node:
  // DATA:DATA, UNKNOWN_SHAPE:unknowshape, KNOWN_SHAPE:knowshape, NETOUTPUT:NETOUTPUT
  Status InitClusters() override;
  // Merge clusters according to the following rules:
  // 1) Iterate through the UNKNOWN_SHAPE clusters, if the input is UNKNOWN_SHAPE,
  //    merge all the clusters in the path(s) between the two clusters
  // 2) Iterate through the KNOWN_SHAPE clusters, if the input is KNOWN_SHAPE, and
  //    and there's only one path between the two clusters , merge the two clusters
  // 3) Iterate through the INPUT_DATA clusters, merge all INPUT_DATA
  Status MergeClustersNormal();
  Status MergeClustersInput();
  Status MergeClustersWithConsistantId();
  Status MergeIdConsistantCluster();
  Status MergeRefVariableCluster();
  void MergeClusters(std::shared_ptr<BaseCluster> &merged_cluster,
                     std::shared_ptr<BaseCluster> &cluster);
  Status MergeClusters() override;
  Status InitClusterType();
  Status MergeClustersUnknownShape();
  Status TryMergeClusters(const ClusterFilter &cluster_filter);
  Status MergeClustersInputData();
  Status ProcessUniqueClusters() override;
  bool IsNeedMarkDynamicTilingDepend(const NodePtr &node) const;
  void MarkDynamicTilingDependNoe(const ComputeGraphPtr &compute_graph) const;
  // Deduplicate merged clusters
  Status PruneUniqueClusters();
  // Clear resource and break circular dependency
  void ClearResource() override;
  // Debug functions
  std::string DebugString() const;
  bool JudgeUnknowShapeWithAttr(const OpDescPtr &opdesc) const;
  // Util functions
  Status CollectSpreadUnknownShapeNodes(NodePtr node);
  Status IsUnknownShapeGraph(const ge::ComputeGraphPtr &graph, bool &is_unknown);
  Status JudgeUnknownShapeForTilingDependNode(const NodePtr &node, bool &is_dynamic) const;
  bool IsNodeSupportAddrRefresh(const NodePtr &node) const;
  Status IsUnknownShapeNode(ge::NodePtr node, bool &is_unknown);
  Status CtrlEdgeTransfer() const;
  std::string GetSubgraphName(const BaseCluster &cluster) const override;
  bool IsNeedBuildPartitionFrame(const BaseCluster &cluster) const override;
  bool IsNodeSupportNoTiling(const ConstNodePtr &node);
  Status MarkOpNoTiling(const NodePtr &node, bool no_tiling) const;
  void RevertOpNoTilingAttr(const NodePtr &node) const;
  Status ReDynamicShapePartitioner();
  void ClearReDataFlowResource();
  Status GenerateCluster();
  Status MarkMemoryDiscontiguousAllocation() const;
  void MergeClustersControlFlow();
  Status IsSingleOpGraph(bool &is_single_op) const;
  Status IsGraphNeedUnknownShapePartition(bool &need_unknown_shape_partition);
  void SetRootGraphUnknown() const;
  Status ChangeSmallClusterType(const size_t threshold) const;
  bool IsSpecialNode(const OpDescPtr &op_desc) const;
  std::string GetPartitionName() const override;
  Status MarkSubgraphUnknownStatus(ComputeGraphPtr graph) const;
  Status CheckIfSubgraphUnknown(const ComputeGraphPtr &graph,
                                bool &is_unknown_shape) const;
  Status BuildPartitionFrame() override;
  Status Initialize();
  Status GetMultiBatchIndependCompileGraphs(const ComputeGraphPtr &compute_graph,
      std::vector<ComputeGraphPtr> &independ_graphs);
  bool IsSubgraphMultiDims() const;
  // Nodes of root_graph_ that satisfy the knowshape rules
  std::unordered_set<NodePtr> known_shape_nodes_;
  // Nodes of root_graph_ that satisfy the unknowshape rules
  std::unordered_set<NodePtr> unknown_shape_nodes_;
  // Nodes of root_graph_ that satisfy the unknowshape rules and support no tiling
  std::unordered_set<NodePtr> unknown_shape_no_tiling_nodes_;
  std::map<int64_t, std::vector<NodePtr>> control_nodes_;
  int64_t static_model_ops_lower_limit_ = 4L;
  bool merge_known_first_{false};
  bool has_special_node_{false};
  bool has_no_tiling_{false};
};

class PartitionerPass {
 public:
  PartitionerPass() = default;
  virtual ~PartitionerPass() = default;
  virtual Status Run(const ge::ComputeGraphPtr &graph,
                     const std::vector<std::shared_ptr<BaseCluster>> &sorted_unique_clusters, bool &result) = 0;
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_DYNAMIC_SHAPE_PARTITION_H_
