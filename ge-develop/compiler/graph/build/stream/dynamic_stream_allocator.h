/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_STREAM_DYNAMIC_STREAM_ALLOCATOR_H_
#define GE_GRAPH_BUILD_STREAM_DYNAMIC_STREAM_ALLOCATOR_H_

#include "ge_common/ge_api_types.h"
#include "graph/any_value.h"
#include "graph/manager/graph_manager_utils.h"
#include "stream_utils.h"

namespace ge {
class DynamicStreamAllocator {
 public:
  Status AssignStreamsForDynamicShapeGraph(const ComputeGraphPtr &root_graph,
                                           const Graph2SubGraphInfoList &subgraph_map);

  static Status AssignAttachedResource(const ComputeGraphPtr &compute_graph, int64_t &stream_num, int64_t &event_num,
                                       int64_t &notify_num, std::vector<uint32_t> &notify_types);

  int64_t GetStreamNum() const {
    return stream_num_;
  }
  int64_t GetEventNum() const {
    return event_num_;
  }

 private:
  Status GetAcParallelEnableConfig();
  Status AssignStreams(const ComputeGraphPtr &root_graph, const Graph2SubGraphInfoList &subgraph_map);
  Status AssignStreamsForGraph(const ComputeGraphPtr &graph, const Graph2SubGraphInfoList &subgraph_map,
                               const std::map<std::string, EngineConfPtr> &engine_confs);

  void AssignStreamForSubgraph(const SubgraphPtr &subgraph);
  Status AssignEnginesOwningStream(const std::vector<SubgraphPtr> &subgraphs);

  Status AssignAicpuCanParallel(const std::vector<SubgraphPtr> &subgraphs,
                                const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map,
                                const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map);
  bool HasOnlyNoTaskInput(const SubgraphPtr &subgraph,
                          const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map) const;
  bool CanReusePredSubgraph(const SubgraphPtr &subgraph,
                            const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map,
                            const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const;
  bool CanAicpuReusePredSubgraph(const SubgraphPtr &subgraph, const SubgraphPtr &pred_subgraph,
                                 const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const;

  Status AssignIndependentAicpuNode(const std::vector<SubgraphPtr> &subgraphs);

  Status AssignWithReuse(const std::vector<SubgraphPtr> &subgraphs,
                         const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map,
                         const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const;
  SubgraphPtr GetPredReusableSubgraph(const SubgraphPtr &subgraph,
                                      const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map) const;
  SubgraphPtr GetSuccReusableSubgraph(const SubgraphPtr &subgraph,
                                      const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const;
  bool CouldReuse(const SubgraphPtr &subgraph, const SubgraphPtr &peer_subgraph) const;

  Status AssignRemainSubgraphNeedAssignStream(const std::vector<SubgraphPtr> &subgraphs);

  Status SetSubgraphStreamToNodes(const ComputeGraphPtr &graph, const std::vector<SubgraphPtr> &subgraphs) const;

  Status ReassignStreamByStreamLabel(const ComputeGraphPtr &graph);

  Status GetNodeNumOfPerStream(const ComputeGraphPtr &graph, std::vector<int64_t> &stream_node_num) const;
  Status RetainEngineOwingNodes(const ComputeGraphPtr &root_graph);

  Status RefreshContinuousStreams(const ComputeGraphPtr &root_graph);
  Status RefreshStreamsForGraph(const ComputeGraphPtr &graph, const std::map<int64_t, int64_t> &old_to_new_streams);
  bool IsForcedAssignMainStream(const NodePtr &node) const;

  Status InsertEvents(const ComputeGraphPtr &root_graph);
  Status ConvertEdgesToEvents(const ComputeGraphPtr &graph);
  Status InsertEventBetweenTwoNodes(const NodePtr &cur_node, const NodePtr &next_node);

  Status OptimizeEvents();
  Status SetEventAttrToNodes(const ComputeGraphPtr &graph);

  bool ac_parallel_enable_{false};
  int64_t next_stream_{0};
  int64_t stream_num_{0};
  int64_t event_num_{0U};
  // send events corresponding to the node
  std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> node_to_send_events_;
  // send notify corresponding to the node
  std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> node_to_recv_events_;
  // engine_id and stream_id
  std::map<std::string, int64_t> engine_streams_;
  // stream_label and stream_id
  std::map<std::string, int64_t> stream_label_ids_;
  // nodes in each stream
  std::map<int64_t, std::vector<NodePtr>> stream_nodes_;
};

using DynamicStreamAllocatorPtr = std::shared_ptr<DynamicStreamAllocator>;
}  // namespace ge
#endif  // GE_GRAPH_BUILD_STREAM_DYNAMIC_STREAM_ALLOCATOR_H_