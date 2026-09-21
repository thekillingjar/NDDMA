/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_STREAM_STREAM_ALLOCATOR_H_
#define GE_GRAPH_BUILD_STREAM_STREAM_ALLOCATOR_H_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {
enum class EventType : std::uint32_t
{
  kEvent,
  kNotify,
};

struct StreamSplitNodeInfo {
  const NodePtr &cur_node;
  bool is_stream_first_node;
  bool split_for_attached_stream;
  size_t assigned_task_num;
  int64_t stream_id;
};

struct StreamSplitSyncInfo {
  const NodePtr &pre_node;
  const NodePtr &not_cur;
  const NodePtr &cur_node;
  bool not_use_cur;
  bool split_for_attached_stream;
  int64_t pre_stream_id;
  int64_t next_stream_id;
};
using TaskNumInfos = std::map<int64_t, size_t>;
using Nodes2SyncInfos = std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey>;
using Node2AttachedStreamId2EventId = std::map<NodePtr, std::map<int64_t, std::vector<uint32_t>>, NodeCompareKey>;
using GetStreamIdFunc = std::function<int64_t(const OpDescPtr &op_desc)>;
struct StreamSplitHelper {
  void Init(int64_t stream_num) {
    const size_t tmp_num = static_cast<size_t>(stream_num);
    stream_task_num_vec.resize(tmp_num);
    added_stream_num_vec.resize(tmp_num);
    latest_stream_id_vec.resize(tmp_num);
    pre_node_vec.resize(tmp_num);

    last_stream_id = stream_num - 1;
    for (int64_t i = 0; i <= last_stream_id; i++) {
      stream_task_num_vec[i] = 0;
      added_stream_num_vec[i] = 0;
      latest_stream_id_vec[i] = i;
      pre_node_vec[i] = nullptr;
    }
  }

  // stream_task_num_vec records the number of all nodes on each stream
  // added_stream_num_vec records the number of streams that each stream needs to increase
  // latest_stream_id_vec records the latest physical stream id for each stream
  std::vector<int64_t> stream_task_num_vec;
  std::vector<int64_t> added_stream_num_vec;
  std::vector<int64_t> latest_stream_id_vec;
  std::map<std::string, int64_t> stream_continuous_2_node_num_map;
  std::map<std::string, std::vector<NodePtr>> stream_continuous_2_nodes_map;
  // stream_2_nodes_map 记录的流拆分之前的stream和stream上所有节点的映射关系
  std::map<int64_t, std::vector<NodePtr>> stream_2_nodes_map;
  std::set<int64_t> attached_stream_;
  std::vector<NodePtr> pre_node_vec;

  uint32_t max_stream_count = 0U;
  uint32_t max_task_count = 0U;
  int64_t last_stream_id = 0;
};
class StreamAllocator {
 public:
  StreamAllocator(ComputeGraphPtr whole_graph, const Graph2SubGraphInfoList &subgraphs);
  StreamAllocator(const StreamAllocator &) = delete;
  StreamAllocator &operator=(const StreamAllocator &) = delete;
  ~StreamAllocator() = default;

  Status AssignLogicalStreams(const std::map<std::string, int32_t> &max_parallel_num, bool hcom_parallel);

  Status InsertSyncNodesByLogicStream(int64_t &stream_num, int64_t &event_num, int64_t &notify_num);

  Status SplitStreamAndRefreshTaskDef(
      std::unordered_map<int64_t , std::vector<domi::TaskDef>> &node_id_2_node_tasks, int64_t &stream_num,
      int64_t &event_num, int64_t &notify_num);

  const std::vector<int64_t> &GetHugeStreams() const { return huge_streams_; }
  const std::vector<uint32_t> &GetNotifyTypes() const {
    return notify_types_;
  }

  std::map<int64_t, int64_t> GetSplitStreamToLogicStream() {
    return split_stream_id_to_logic_stream_id_;
  }

 private:
  Status PreProcessOfInsertSyncNodes();
  Status InsertSyncNodesWithNotify();
  Status InserSyncNodesWithoutNotify();
  Status PostProcessOfSplitStreams();
  Status AssignSingleStream(const std::unordered_map<int64_t, std::vector<domi::TaskDef>> &node_id_2_node_tasks);

  Status SetActiveStreamsByLabel();
  Status SetActiveStreamsForSubgraphs();

  Status InsertSyncEvents(const EventType insert_event_type);
  Status InsertSyncEventsWithAttachedStream(const EventType insert_event_type);
  Status InsertOneEventInTwoNodes(const EventType insert_event_type, const NodePtr &cur_node, const NodePtr &next_node);
  Status InsertOneEventInTwoNodesWithAttachedStream(const EventType insert_event_type,
                                                    const NodePtr &src_node, const NodePtr &dst_node);
  Status InsertEventsForSubgraph(const EventType insert_event_type);

  void BuildEventReuseMap(const EventType event_type, const std::vector<uint32_t> &events,
                          std::map<uint32_t, uint32_t> &event_seen, uint32_t &event_id) const;

  Status OptimizeSyncEvents(const EventType event_type,
                            std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                            std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events);
  Status OptimizeByStreamActivate(const EventType event_type,
                                  std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                                  std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) const;

  // Determine if the successor node of RecvNode is directly or indirectly activated by the SendNode precursor node
  bool IsRecvNodeActivatedBySendNode(const NodePtr &send_node_ptr, const NodePtr &recv_node_ptr) const;
  bool IsActiveAfterNextIteration(const NodePtr &active_node_ptr) const;

  Status SplitStreams(std::unordered_map<int64_t, std::vector<domi::TaskDef>> &node_id_2_node_tasks,
                      std::vector<std::set<int64_t>> &split_streams);
  Status SplitStreamForOneNode(StreamSplitNodeInfo &stream_split_node_info, StreamSplitHelper &helper,
                               std::vector<std::set<int64_t>> &split_streams, std::vector<domi::TaskDef> &task_defs);
  Status SplitNodesToNewStream(const StreamSplitNodeInfo &stream_split_node_info,
                               std::vector<std::set<int64_t>> &split_streams, StreamSplitHelper &helper);
  bool NeedSpiltNewStream(int64_t stream_node_num, int64_t max_node_num_one_stream, const OpDescPtr &op_desc,
                          bool is_stream_first_node) const;

  Status SetLogicStreamIdAttr();

  Status UpdateStreamSwitchByLogicStream();
  Status UpdateActiveStreams(const std::vector<std::set<int64_t>> &split_streams);
  void UpdateLabelStreams(const std::vector<std::set<int64_t>> &split_streams);
  Status UpdateActiveStreamsForSwitchNode(const NodePtr &switch_node);
  Status InsertActiveNodesAfterSwitch(const NodePtr &switch_node, std::vector<NodePtr> &active_nodes);
  Status UpdateActiveStreamsForActiveNode(const std::vector<std::set<int64_t>> &split_streams, const NodePtr &node);
  Status UpdateActiveStreamsForSubgraphs();
  bool IsActivated(int64_t stream_id) const;
  Status SetActiveStreamsForLoop(bool is_before_split_stream = true,
                                 const std::vector<std::set<int64_t>> &split_streams = {});
  Status CheckStreamActived() const;

  Status ReuseEvent(bool send_to,
    const std::unordered_map<std::string, ge::NodePtr> &name_to_node_map,
    const std::unordered_map<ge::NodePtr, std::vector<std::pair<std::string, uint32_t>>> &node_to_event_id);
  Status RefreshEventsAndNotifiesWithReuse();
  // is_reuse_event = true means reuse for event, false means reuse for notify
  Status ReuseEventForMultiDims(const EventType event_type,
                                std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                                std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events);
  Status BuildEventReuseMapOfOneDim(const ComputeGraphPtr &subgraph, uint32_t depth, uint32_t &cur_event_id,
                                    std::map<uint32_t, uint32_t> &event_seen) const;

  Status BuildEventReuseMapOutOfDims(const EventType event_type, uint32_t max_event_id,
                                     const map<NodePtr, vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                                     const map<NodePtr, vector<uint32_t>, NodeCompareKey> &node_to_recv_events,
                                     std::map<uint32_t, uint32_t> &event_seen);

  std::vector<uint32_t> GetSyncIdWithinSameGraph(
      const ComputeGraphPtr &graph, const std::vector<uint32_t> &sync_ids,
      const std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &peer_sync_info) const;

  Status GenerateSyncEventNodes(bool change_topo = true);
  Status InsertSyncSendEventNode(const NodePtr &node, const std::vector<uint32_t> &event_id_list, int64_t stream_id,
                                 int32_t &total_num, std::unordered_map<std::string, uint32_t> &sync_event_name);
  Status InsertSyncRecvEventNode(const NodePtr &node, const std::vector<uint32_t> &event_id_list, int64_t stream_id,
                                 int32_t &total_num, std::unordered_map<std::string, uint32_t> &sync_event_name);
  Status InsertSyncSendNotifyNode(const NodePtr &node, int32_t &total_num,
                                  std::unordered_map<std::string, uint32_t> &sync_notify_name);
  Status InsertSyncRecvNotifyNode(const NodePtr &node, int32_t &total_num,
                                  std::unordered_map<std::string, uint32_t> &sync_notify_name);

  Status RefreshContinuousNotifies();
  Status BuildNotifyReuseMapOfOneDim(const ComputeGraphPtr &subgraph, uint32_t depth, uint32_t &cur_notify_id,
                                     std::map<uint32_t, uint32_t> &notify_seen) const;

  void DumpEvents(const EventType event_type,
                  std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                  std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) const;

  Status GetMaxStreamAndTask(bool huge_stream, uint32_t &max_stream_count, uint32_t &max_task_count) const;
  void AddTaskNum(const NodePtr &node, int64_t &task_num, size_t task_size, bool is_attached_stream) const;

  Status AddEventPair(const NodePtr &send_node, const NodePtr &recv_node, Nodes2SyncInfos &nodes_2_send_sync_infos,
                      Nodes2SyncInfos &nodes_2_recv_sync_infos);
  Status AddEventPairBetweenAttachedAndMain(const NodePtr &send_node, const NodePtr &recv_node, int64_t pre_stream_id,
                                            Node2AttachedStreamId2EventId &nodes_2_send_event,
                                            Nodes2SyncInfos &nodes_2_recv_event);
  Status AddAttachedStreamEventPair(const NodePtr &send_node, const NodePtr &recv_node, int64_t pre_stream_id,
                                    int64_t next_stream_id, Node2AttachedStreamId2EventId &nodes_2_send_event,
                                    Node2AttachedStreamId2EventId &nodes_2_recv_event);
  Status AddEventIdWhenStreamSplit(const StreamSplitSyncInfo &stream_split_sync_info);

  Status AddActiveNodes(const NodePtr &switch_node, const std::vector<std::string> &ori_active_label_list,
                        std::vector<std::string> &active_label_list, std::vector<NodePtr> &added_active_nodes);
  Status SetActiveStreamList(const NodePtr &active_node, const std::string &active_label);
  Status SetActiveNodeStreamLabel(const ge::NodePtr &node, const std::string &label,
                                  std::set<std::string> &new_active_stream_labels) const;
  Status AssignAttachedNotifyResource();
  Status AssignAttachedEventResource();
  Status CoverAllStreamByNetoutput();
  Status SetNewTopoId();
  void ClearNodes2SyncEvents();
  Status CollectTaskSize(std::unordered_map<int64_t, std::vector<domi::TaskDef>> &node_id_2_node_tasks,
                         uint32_t per_stream_max_task_size);
  Status RefreshStreamActiveNodeTaskSize(const std::vector<NodePtr> &stream_active_nodes,
                                         const std::map<int64_t, size_t> &logical_stream_id_to_real_stream_num);

  ComputeGraphPtr whole_graph_;
  const Graph2SubGraphInfoList &subgraphs_;

  int64_t stream_num_{0}; // main_stream_num + attach_stream_num
  int64_t main_stream_num_{0};
  uint32_t notify_num_{0};
  std::vector<uint32_t> notify_types_;
  uint32_t event_num_{0};
  int64_t new_topo_id_{0};
  bool enable_single_stream_{false};
  // this event_type_ describe the event between logical stream, not include the event that insert in SplitStream
  // Process.
  EventType event_type_{EventType::kEvent};
  std::vector<int64_t> huge_streams_;

  // <stream label, std::set<stream id>>
  std::map<std::string, std::set<int64_t>> labeled_streams_;
  std::map<NodePtr, std::set<std::string>> switch_to_has_set_labels_;
  std::map<NodePtr, std::set<std::string>> switch_to_new_active_stream_labels_;
  std::map<std::string, std::set<NodePtr>> specific_activated_labels_;
  std::set<int64_t> specific_activated_streams_;
  std::map<int64_t, std::set<NodePtr>> specific_activated_streams_nodes_map_;

  std::map<NodePtr, int64_t> node_split_stream_map_;
  std::map<int64_t, int64_t> split_stream_id_to_logic_stream_id_;
  std::map<ComputeGraphPtr, NodePtr> subgraph_first_active_node_map_;
  StreamSplitHelper helper_;

  // 记录node id和task num的映射，1个node可以有多个stream id
  std::map<int64_t, TaskNumInfos> node_id_to_task_num_infos_;

  // send events corresponding to the node
  Nodes2SyncInfos node_to_send_events_;

  // send notify corresponding to the node
  Nodes2SyncInfos node_to_send_notifies_;

  // recv events corresponding to the node
  Nodes2SyncInfos node_to_recv_events_;

  // recv notify corresponding to the node
  Nodes2SyncInfos node_to_recv_notifies_;

  // send events and recv events for node with one attached stream, logical stream use
  Nodes2SyncInfos attached_node_to_send_events_;
  Nodes2SyncInfos attached_node_to_recv_events_;

  // send events and recv events for node with attached stream, split stream use
  // node may have multi attached stream(SuperKernel)
  Node2AttachedStreamId2EventId  attached_node_to_stream_id_to_send_event_id_;
  Node2AttachedStreamId2EventId  attached_node_to_stream_id_to_recv_event_id_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_STREAM_STREAM_ALLOCATOR_H_
