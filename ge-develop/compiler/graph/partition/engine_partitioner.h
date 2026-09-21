/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PARTITION_GRAPH_PARTITION_H_
#define GE_GRAPH_PARTITION_GRAPH_PARTITION_H_

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/operator_reg.h"
#include "graph/partition/engine_place.h"

namespace ge {
using PartitionMap = std::unordered_map<ComputeGraphPtr, std::string>;
using NodetoNodeMap = std::unordered_map<NodePtr, NodePtr>;
using NodeNameToNodeMap = std::unordered_map<std::string, NodePtr>;
using EnginetoGraphMap = std::unordered_map<std::string, ComputeGraphPtr>;
using EdgeMap = std::set<std::pair<AnchorPtr, AnchorPtr>>;
using ClusterSet = std::set<size_t>;
class Cluster {
 public:
  size_t index_;              // corresponding to rank of node
  ClusterSet in_clu_;         // inClusters index
  ClusterSet out_clu_;        // outClusters index
  std::list<NodePtr> nodes_;  // including node of this cluster
  std::string engine_name_;   // data like must be a specific engine
  std::string stream_label_;
  std::string user_stream_label_;
  explicit Cluster(size_t index, std::string engine, std::string stream, std::string user_stream = "")
      : index_(index),
        engine_name_(std::move(engine)),
        stream_label_(std::move(stream)),
        user_stream_label_(std::move(user_stream)) {
  }
  ~Cluster() = default;
};
using ClusterPtr = std::shared_ptr<Cluster>;

class EnginePartitioner {
 public:
  /// Partition() can only be called in Partition mode.
  /// MergeAfterSubGraphOptimization() can only be called in Merge mode.
  /// After Partition(), change to Merge mode. After MergeAfterSubGraphOptimization(), change to Partition mode
  enum class Mode {
    kAtomicEnginePartitioning,
    kCompositeEnginePartitioning,
    kSecondPartitioning,
    kMerging
  };
  EnginePartitioner() : partition_times_(0) {};
  ~EnginePartitioner() = default;

  // the main method that partitions the graph
  // input_size and output_size are the number of inputs and outputs in the original graph
  Status Partition(const ComputeGraphPtr &compute_graph, Mode mode);

  // after partition, all SubGraph will be merged back based on end<->pld.
  Status MergeAfterSubGraphOptimization(ComputeGraphPtr &output_merged_compute_graph,
                                        const ComputeGraphPtr &original_compute_graph,
                                        EnginePartitioner::Mode mode = EnginePartitioner::Mode::kAtomicEnginePartitioning);
  // Return all subgraphs
  const Graph2SubGraphInfoList &GetSubGraphMap();

  const Graph2InputNodesSubGraphInfo &GetSubGraphInfoMap() {return graph_2_input_subgraph_; }

  EnginePlacer &GetEnginePlacer() { return engine_placer_; }

 private:
  struct GraphPartitionInfo {
    PartitionMap partitions_;  // sub-graphs after partition <sub-graph-id, ComputeGraphPtr>
    std::unordered_map<ComputeGraphPtr, size_t> partitions_2_rank_;  // <subGraph, rank>
    std::vector<ComputeGraphPtr> rank_2_partitions_;                 // <rank, subGraph>
    NodeNameToNodeMap corresponding_node_in_partitions_;             // mapping between a node in the original graph and
    uint32_t num_of_pld_end_;                                        // a counter to track 'place holder' and 'end'
    size_t input_size_;
    size_t output_size_;
    std::string output_name_;
    NodetoNodeMap end_2_pld_;                // mapping between each 'end; and 'placeHolder' node
    NodetoNodeMap pld_2_end_;                // mapping between each 'placeHolder' and 'end' node
    std::map<size_t, NodePtr> index_2_end_;  // order mapping between peerindex and 'end' node
    Mode mode_;
    std::unordered_map<size_t, ClusterPtr> clusters_;                       // index to cluster ptr, contains all nodes
    std::unordered_map<NodePtr, std::shared_ptr<Cluster>> node_2_cluster_;  // node map to cluster
    std::unordered_map<std::shared_ptr<Cluster>, ComputeGraphPtr> cluster_2_partition_;  // cluster map to subgraph
    GraphPartitionInfo(Mode mode = Mode::kAtomicEnginePartitioning)
        : num_of_pld_end_(0), input_size_(0), output_size_(0), mode_(mode) {}
    ~GraphPartitionInfo() = default;
  };

  Status FindOverflowAttr(const ge::ComputeGraphPtr &sub_graph, ge::ComputeGraphPtr &original_graph);
  Status MergeOverflowAttr(const ge::ComputeGraphPtr &sub_graph, ge::ComputeGraphPtr &root_graph) const;
  Status MergeSubGraph(ge::ComputeGraphPtr &output_merged_compute_graph,
                       const ge::ComputeGraphPtr &original_compute_graph);
  Status PartitionSubGraph(const ge::ComputeGraphPtr &compute_graph, Mode mode);
  Status InheritOriginalAttr(const ComputeGraphPtr &original_compute_graph,
                             ComputeGraphPtr &output_merged_compute_graph) const;
  Status MergeAllSubGraph(ComputeGraphPtr &output_merged_compute_graph,
                          const std::vector<SubGraphInfoPtr> &sub_graph_list,
                          const GraphPartitionInfo &graph_info) const;
  Status CheckValidIfEnd2PldEmpty(const GraphPartitionInfo &graph_info,
                                  ComputeGraphPtr &output_merged_compute_graph) const;

  // Run engine placer, assign engine, check support amd init all clusters
  Status Initialize(const ComputeGraphPtr &compute_graph, Mode mode);
  Status InitializeInputClusters(const NodePtr &node, const ClusterPtr &cluster, size_t index);

  /// add pld and end nodes between two sub-graphs for the specific anchors
  /// all anchors are in original graph
  Status AddPlaceHolderEnd(const AnchorPtr &out_anchor, const AnchorPtr &in_anchor);
  void AddNewGraphToPartition(const ComputeGraphPtr &input_graph, const std::string &engine_name);
  Status AddPartitionsToGraphNode(std::vector<SubGraphInfoPtr> &output_subgraphs, ComputeGraphPtr compute_graph);

  // check if the node has no input
  bool HasNoInput(const NodePtr &node) const;

  // check if the node is data-like. Currently data-like means: data, variable, const
  bool IsDataLike(NodePtr node) const;
  graphStatus MakeEndOpNode(const AnchorPtr &out_anchor,
                            const ge::ComputeGraphPtr &end_graph,
                            NodePtr &new_end_node);
  graphStatus SetPldOpAttr(const NodePtr &src_node,
                           const NodePtr &new_end_node,
                           const ge::ComputeGraphPtr &end_graph,
                           const AnchorPtr &out_anchor,
                           const OpDescPtr &pld_op_desc) const;
  graphStatus SetEndOpAttr(const NodePtr &dst_node, const OpDescPtr &end_op_desc) const;
  // add place holder and end node in src and dst graph
  graphStatus AddPlaceHolderEndInSrcDstGraph(const AnchorPtr &out_anchor, const AnchorPtr &peer_in_anchor,
                                             const ComputeGraphPtr &pld_graph, const ComputeGraphPtr &end_graph);
  graphStatus MakePldOpNode(const AnchorPtr &peer_in_anchor,
                            const NodePtr &src_node,
                            const ge::ComputeGraphPtr &pld_graph,
                            NodePtr &new_pld_node);
  Status LinkInput2EndRemoveOrginalLink(const NodePtr &input_node, const ComputeGraphPtr &src_graph,
                                        const ComputeGraphPtr &dst_graph);

  /// After partition, put input nodes in srcGraph to dstGraph. Data will be linked to 'end';
  /// the other end will be linked to 'placeholder'
  Status PutInputNodesInSubGraph(const ComputeGraphPtr &src_graph, const ComputeGraphPtr &dst_graph);

  // Sort all subGraphs topologically, store the info in sorted_partitions_ <computeGraph, rank>
  Status SortSubGraphs(const ComputeGraphPtr &compute_graph);
  AnchorPtr GetEndInAnchor(const AnchorPtr &src_anchor, const NodePtr &end_node) const;
  AnchorPtr GetPldOutAnchor(const NodePtr &pld_node, const AnchorPtr &dst_anchor) const;
  Status RemoveNodeAndEdgeBetweenEndPld(ComputeGraphPtr &output_merged_compute_graph,
                                        const std::vector<SubGraphInfoPtr> &sub_graph_list,
                                        const GraphPartitionInfo &graph_info);
  void AddEndPldInformationToSubGraphInfo(SubGraphInfoPtr &subgraph_info);
  bool IsMergeable(size_t parent_cluster, size_t child_cluster, size_t upper_bound);

  // Link from->to
  void InsertEdge(size_t from, size_t to);

  // Remove parent cluster's out and child cluster's in
  void RemoveEdge(size_t parent_cluster, size_t child_cluster);
  void MergeTwoClusters(size_t parent_cluster, size_t &child_cluster);

  // Check if there's a second path between two clusters. The max path length is upper_bound
  bool HasSecondPath(size_t src, size_t dst, size_t upper_bound);
  Status MarkClustersWithConsistantId();
  // Mark all clusters
  void MarkClusters();
  Status SplitNodeInputs(const NodePtr &node, const NodePtr &corresponding_node, const ClusterPtr &child_cluster);

  /// Split all sub graph and add placeholder, end according to marks
  /// traverse marked clusters and split them into sub-graphs
  Status SplitSubGraphs(const ComputeGraphPtr &compute_graph);
  graphStatus UpdateEndOpDesc(const NodePtr &src_node, int32_t output_index, const OpDescPtr &end_op_desc) const;
  graphStatus UpdatePldOpDesc(const NodePtr &dst_node, int32_t input_index, const OpDescPtr &pld_op_desc) const;

  // Clear partition data
  void ClearAllPartitionData();
  Status SetMergedGraphId(const ComputeGraphPtr &output_merged_compute_graph,
                          const GraphPartitionInfo &graph_info) const;

  const NodeEngineMap &GetNodeEngineMap() const;
  Status UpdateCorrespondNodeInPartitions(const ComputeGraphPtr &compute_graph, GraphPartitionInfo &graph_info) const;

  Status SetMemberForSubGraphInfo(ge::SubGraphInfoPtr &sgi, const ComputeGraphPtr &sub_graph,
                                  const std::string &engine_name);
  EnginePlacer engine_placer_;
  std::unordered_map<ComputeGraphPtr, GraphPartitionInfo> graph_2_graph_partition_info_;
  Graph2SubGraphInfoList graph_2_subgraph_list_;
  Graph2InputNodesSubGraphInfo graph_2_input_subgraph_;
  GraphPartitionInfo graph_info_;
  uint32_t partition_times_;  // times of call partition
  std::map<Mode, std::string> mode_2_str_ = {{ Mode::kAtomicEnginePartitioning, "AtomicEnginePartitioning" },
                                             { Mode::kCompositeEnginePartitioning, "CompositeEnginePartitioning" },
                                             { Mode::kSecondPartitioning, "SecondPartitioning" },
                                             { Mode::kMerging, "Merging" }};
  
  // for support overflow detection
  int64_t global_workspace_type_ = -1;
  int64_t global_workspace_size_ = -1;
  std::string topo_sorting_mode_;
  Mode current_mode_;
  friend class GraphManager;
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_GRAPH_PARTITION_H_
