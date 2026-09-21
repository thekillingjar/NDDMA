/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dynamic_stream_allocator.h"

#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"
#include "framework/common/types.h"
#include "graph/build/stream/assign_attached_notify_pass.h"
#include "graph/build/stream/assign_attached_stream_pass.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/constant_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "base/err_msg.h"

namespace ge {
namespace {
const std::set<std::string> kAicpuEngineIds = {"DNN_VM_AICPU", "DNN_VM_AICPU_FFTS_PLUS", "DNN_VM_AICPU_ASCEND",
                                               "DNN_VM_AICPU_ASCEND_FFTS_PLUS"};
const std::set<std::string> kIndependentAicpu = {DROPOUTGENMASK, DROPOUTDOMASKV3, GETNEXT, DYNAMICGETNEXT,
                                                 DYNAMICGETNEXTV2};
const std::set<std::string> kEnginesOwningStream = {"AIcoreEngine", "VectorEngine", "DNN_HCCL", "DSAEngine"};
const std::set<std::string> kEnginesNoTask = {"DNN_VM_GE_LOCAL"};
// FILECONSTANT needs to be handled in loading stage and does not support multi stream
const std::set<std::string> kNodesForcedInMainStream = {NETOUTPUT, FILECONSTANT};

bool TopoOrderCompare(const NodePtr &n0, const NodePtr &n1) {
  if ((n0->GetOpDesc() == nullptr) || (n1->GetOpDesc() == nullptr)) {
    return false;
  }
  return (n0->GetOpDesc()->GetId() < n1->GetOpDesc()->GetId());
}
}  // namespace

Status DynamicStreamAllocator::AssignStreamsForDynamicShapeGraph(const ComputeGraphPtr &root_graph,
                                                                 const Graph2SubGraphInfoList &subgraph_map) {
  GE_CHECK_NOTNULL(root_graph);

  GE_ASSERT_SUCCESS(GetAcParallelEnableConfig());

  GE_ASSERT_SUCCESS(AssignStreams(root_graph, subgraph_map));

  GE_ASSERT_SUCCESS(InsertEvents(root_graph));

  GELOGI("Graph: %s, stream num: %lld, event num: %lld.", root_graph->GetName().c_str(), stream_num_, event_num_);

  return SUCCESS;
}

Status DynamicStreamAllocator::AssignAttachedResource(const ComputeGraphPtr &compute_graph, int64_t &stream_num,
                                                      int64_t &event_num, int64_t &notify_num,
                                                      std::vector<uint32_t> &notify_types) {
  (void)event_num;
  std::vector<ComputeGraphPtr> dyn_graphs{compute_graph};
  for (const auto &sub_graph : compute_graph->GetAllSubgraphs()) {
    if (sub_graph->GetGraphUnknownFlag()) {
      dyn_graphs.push_back(sub_graph);
    }
  }
  AssignAttachedStreamPass attach_stream_pass;
  AssignAttachedNotifyPass attached_notify_pass;
  notify_types.resize(notify_num, RT_NOTIFY_DEFAULT);
  uint32_t cur_notify_num = static_cast<uint32_t>(notify_num);
  for (const auto &dyn_graph : dyn_graphs) {
    GE_ASSERT_SUCCESS(attach_stream_pass.Run(dyn_graph, stream_num));
    GE_ASSERT_SUCCESS(attached_notify_pass.Run(dyn_graph, cur_notify_num, notify_types));
  }
  notify_num = cur_notify_num;
  return SUCCESS;
}

Status DynamicStreamAllocator::GetAcParallelEnableConfig() {
  std::string ac_parallel_enable;
  (void)GetContext().GetOption(AC_PARALLEL_ENABLE, ac_parallel_enable);

  const std::set<std::string> options = {"", "0", "1"};
  if (options.count(ac_parallel_enable) <= 0U) {
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
                              std::vector<const char_t *>({"ac_parallel_enable", ac_parallel_enable.c_str(),
                                                           "The value can only be empty, 0 and 1."}));
    GELOGE(PARAM_INVALID, "[Check][Param] ac_parallel_enable: %s is invalid, only can be empty, 0 and 1.",
           ac_parallel_enable.c_str());
    return PARAM_INVALID;
  }

  GELOGI("ac_parallel_enable: %s.", ac_parallel_enable.c_str());
  ac_parallel_enable_ = (ac_parallel_enable == "1");
  return SUCCESS;
}

Status DynamicStreamAllocator::AssignStreams(const ComputeGraphPtr &root_graph,
                                             const Graph2SubGraphInfoList &subgraph_map) {
  const auto engine_confs = StreamUtils::GetEngineConfs();

  GE_ASSERT_SUCCESS(AssignStreamsForGraph(root_graph, subgraph_map, engine_confs),
                    "Assign stream for graph: %s failed.", root_graph->GetName().c_str());
  for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
    if (subgraph->GetGraphUnknownFlag()) {
      GE_ASSERT_SUCCESS(AssignStreamsForGraph(subgraph, subgraph_map, engine_confs),
                        "Assign stream for graph: %s failed.", subgraph->GetName().c_str());
    }
  }

  GE_ASSERT_SUCCESS(RefreshContinuousStreams(root_graph));

  GE_ASSERT_SUCCESS(StreamUtils::RunCustomStreamPass(root_graph, stream_num_));
  return SUCCESS;
}

Status DynamicStreamAllocator::AssignStreamsForGraph(const ComputeGraphPtr &graph,
                                                     const Graph2SubGraphInfoList &subgraph_map,
                                                     const std::map<std::string, EngineConfPtr> &engine_confs) {
  GE_CHECK_NOTNULL(graph);

  std::map<std::string, int32_t> max_parallel_num;
  std::vector<SubgraphPtr> subgraphs;
  GE_ASSERT_SUCCESS(StreamUtils::ConvertSubgraphs(graph, subgraph_map, engine_confs, max_parallel_num, subgraphs));

  GE_ASSERT_SUCCESS(AssignEnginesOwningStream(subgraphs));

  const auto end_subgraph_map = StreamUtils::InitEndSubgraphMap(subgraphs);
  const auto pld_subgraph_map = StreamUtils::InitPldSubgraphMap(subgraphs);
  if (ac_parallel_enable_) {
    GE_ASSERT_SUCCESS(AssignAicpuCanParallel(subgraphs, end_subgraph_map, pld_subgraph_map));
  } else {
    GE_ASSERT_SUCCESS(AssignIndependentAicpuNode(subgraphs));
  }

  GE_ASSERT_SUCCESS(AssignWithReuse(subgraphs, end_subgraph_map, pld_subgraph_map));

  GE_ASSERT_SUCCESS(AssignRemainSubgraphNeedAssignStream(subgraphs));

  GE_ASSERT_SUCCESS(SetSubgraphStreamToNodes(graph, subgraphs));

  GE_ASSERT_SUCCESS(ReassignStreamByStreamLabel(graph));

  return SUCCESS;
}

void DynamicStreamAllocator::AssignStreamForSubgraph(const SubgraphPtr &subgraph) {
  const auto iter = engine_streams_.find(subgraph->engine_conf.id);
  if (iter == engine_streams_.end()) {
    subgraph->stream_id = next_stream_;
    engine_streams_.emplace(subgraph->engine_conf.id, next_stream_);
    ++next_stream_;
  } else {
    subgraph->stream_id = iter->second;
  }
  GELOGI("Assign stream_id: %lld for engine: %s, subgraph: %s.", subgraph->stream_id, subgraph->engine_conf.id.c_str(),
         subgraph->name.c_str());
}

Status DynamicStreamAllocator::AssignEnginesOwningStream(const std::vector<SubgraphPtr> &subgraphs) {
  for (const auto &subgraph : subgraphs) {
    if (kEnginesOwningStream.count(subgraph->engine_conf.id) == 0U) {
      continue;
    }

    AssignStreamForSubgraph(subgraph);
  }

  return SUCCESS;
}

Status DynamicStreamAllocator::AssignAicpuCanParallel(
    const std::vector<SubgraphPtr> &subgraphs, const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map,
    const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) {
  for (const auto &subgraph : subgraphs) {
    if (kAicpuEngineIds.count(subgraph->engine_conf.id) == 0U) {
      continue;
    }

    const bool has_no_input =
        std::all_of(subgraph->subgraph_info.GetPld2EndMap().begin(), subgraph->subgraph_info.GetPld2EndMap().end(),
                    [](const std::pair<NodePtr, NodePtr> &pair) { return (pair.second == nullptr); });
    const bool has_only_no_task_input = HasOnlyNoTaskInput(subgraph, end_subgraph_map);
    const bool can_reuse_pred_subgraph = CanReusePredSubgraph(subgraph, end_subgraph_map, pld_subgraph_map);
    GELOGI("Subgraph: %s, has_no_input: %d, has_only_no_task_input: %d, can_reuse_pred_subgraph: %d.",
           subgraph->name.c_str(), static_cast<int32_t>(has_no_input), static_cast<int32_t>(has_only_no_task_input),
           static_cast<int32_t>(can_reuse_pred_subgraph));
    if (has_no_input || has_only_no_task_input || (!can_reuse_pred_subgraph)) {
      AssignStreamForSubgraph(subgraph);
    }
  }

  return SUCCESS;
}

bool DynamicStreamAllocator::HasOnlyNoTaskInput(
    const SubgraphPtr &subgraph, const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map) const {
  for (const auto &pld_2_end : subgraph->subgraph_info.GetPld2EndMap()) {
    const auto iter = end_subgraph_map.find(pld_2_end.second);
    if ((iter == end_subgraph_map.end()) || (iter->second == nullptr)) {
      return false;
    }
    const auto &pred_subgraph = iter->second;
    if (kEnginesNoTask.count(pred_subgraph->engine_conf.id) == 0U) {
      return false;
    }
    if (std::any_of(pred_subgraph->subgraph_info.GetPld2EndMap().begin(),
                    pred_subgraph->subgraph_info.GetPld2EndMap().end(),
                    [](const std::pair<NodePtr, NodePtr> &pair) { return (pair.second != nullptr); })) {
      return false;
    }
  }
  return true;
}

bool DynamicStreamAllocator::CanReusePredSubgraph(
    const SubgraphPtr &subgraph, const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map,
    const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const {
  for (const auto &pld_2_end : subgraph->subgraph_info.GetPld2EndMap()) {
    const auto iter = end_subgraph_map.find(pld_2_end.second);
    if ((iter == end_subgraph_map.end()) || (iter->second == nullptr)) {
      continue;
    }
    const auto &pred_subgraph = iter->second;
    if (CanAicpuReusePredSubgraph(subgraph, pred_subgraph, pld_subgraph_map)) {
      return true;
    }
  }
  return false;
}

bool DynamicStreamAllocator::CanAicpuReusePredSubgraph(
    const SubgraphPtr &subgraph, const SubgraphPtr &pred_subgraph,
    const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const {
  if ((kEnginesOwningStream.count(pred_subgraph->engine_conf.id) == 0U) ||
      StreamUtils::IsEngineIndependent(*pred_subgraph)) {
    return false;
  }

  const auto is_higher_priority_brother = [&subgraph, &pred_subgraph,
                                           &pld_subgraph_map](const std::pair<NodePtr, NodePtr> &pair) {
    const auto iter = pld_subgraph_map.find(pair.second);
    if (iter == pld_subgraph_map.end()) {
      return false;
    }
    const auto &pred_subgraph_succ = iter->second;
    return ((pred_subgraph_succ != subgraph) && (pred_subgraph_succ->engine_conf.id == pred_subgraph->engine_conf.id));
  };
  if (std::any_of(pred_subgraph->subgraph_info.GetEnd2PldMap().begin(),
                  pred_subgraph->subgraph_info.GetEnd2PldMap().end(), is_higher_priority_brother)) {
    return false;
  }
  return true;
}

Status DynamicStreamAllocator::AssignIndependentAicpuNode(const std::vector<SubgraphPtr> &subgraphs) {
  for (const auto &subgraph : subgraphs) {
    if (kAicpuEngineIds.count(subgraph->engine_conf.id) == 0U) {
      continue;
    }
    const auto &graph = subgraph->subgraph_info.GetSubGraph();
    GE_CHECK_NOTNULL(graph);
    std::set<NodePtr> aicpu_nodes;
    for (const auto &node : graph->GetDirectNode()) {
      if (kIndependentAicpu.count(node->GetType()) != 0U) {
        aicpu_nodes.emplace(node);
      }
    }
    if (aicpu_nodes.empty()) {
      continue;
    }
    AssignStreamForSubgraph(subgraph);
  }
  return SUCCESS;
}

Status DynamicStreamAllocator::AssignWithReuse(const std::vector<SubgraphPtr> &subgraphs,
                                               const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map,
                                               const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const {
  for (const auto &subgraph : subgraphs) {
    GE_CHECK_NOTNULL(subgraph->subgraph_info.GetSubGraph());
    if (StreamUtils::HasAssignedStream(*subgraph)) {
      continue;
    }

    SubgraphPtr reusable_subgraph = GetPredReusableSubgraph(subgraph, end_subgraph_map);
    if (reusable_subgraph == nullptr) {
      reusable_subgraph = GetSuccReusableSubgraph(subgraph, pld_subgraph_map);
    }
    if (reusable_subgraph == nullptr) {
      GELOGI("Can not find reusable subgraph of subgraph: %s.", subgraph->name.c_str());
      continue;
    }

    if (reusable_subgraph->reused_subgraph != nullptr) {
      reusable_subgraph = reusable_subgraph->reused_subgraph;
    }

    if (!StreamUtils::HasAssignedStream(*reusable_subgraph)) {
      GELOGI("Reusable subgraph: %s of %s has not assigned stream.", reusable_subgraph->name.c_str(),
             subgraph->name.c_str());
      continue;
    }

    subgraph->stream_id = reusable_subgraph->stream_id;
    subgraph->reused_subgraph = reusable_subgraph;
    GELOGI("Subgraph: %s of engine: %s reuses stream id: %lld of subgraph: %s of engine: %s.", subgraph->name.c_str(),
           subgraph->engine_conf.id.c_str(), subgraph->stream_id, reusable_subgraph->name.c_str(),
           reusable_subgraph->engine_conf.id.c_str());
  }

  return SUCCESS;
}

SubgraphPtr DynamicStreamAllocator::GetPredReusableSubgraph(
    const SubgraphPtr &subgraph, const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map) const {
  std::set<SubgraphPtr> reusable_subgraphs;
  const SubGraphInfo &subgraph_info = subgraph->subgraph_info;
  for (const auto &pld_2_end : subgraph_info.GetPld2EndMap()) {
    const auto iter = end_subgraph_map.find(pld_2_end.second);
    if ((iter != end_subgraph_map.end()) && (iter->second != nullptr) &&
        (reusable_subgraphs.count(iter->second) == 0U)) {
      if (CouldReuse(subgraph, iter->second)) {
        reusable_subgraphs.emplace(iter->second);
      }
    }
  }

  return StreamUtils::GetTopPrioritySubgraph(reusable_subgraphs);
}

SubgraphPtr DynamicStreamAllocator::GetSuccReusableSubgraph(
    const SubgraphPtr &subgraph, const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const {
  std::set<SubgraphPtr> reusable_subgraphs;
  const SubGraphInfo &subgraph_info = subgraph->subgraph_info;
  for (const auto &end_2_pld : subgraph_info.GetEnd2PldMap()) {
    const auto iter = pld_subgraph_map.find(end_2_pld.second);
    if ((iter != pld_subgraph_map.end()) && (iter->second != nullptr) &&
        (reusable_subgraphs.count(iter->second) == 0U)) {
      if (CouldReuse(subgraph, iter->second)) {
        reusable_subgraphs.emplace(iter->second);
      }
    }
  }

  return StreamUtils::GetTopPrioritySubgraph(reusable_subgraphs);
}

bool DynamicStreamAllocator::CouldReuse(const SubgraphPtr &subgraph, const SubgraphPtr &peer_subgraph) const {
  if (kAicpuEngineIds.count(subgraph->engine_conf.id) > 0U) {
    // aicpu can not attach to hccl
    if (StreamUtils::IsEngineIndependent(*peer_subgraph)) {
      return false;
    }
  }

  if ((subgraph->engine_conf.id == peer_subgraph->engine_conf.id) || StreamUtils::IsEngineAttach(*subgraph)) {
    return true;
  }

  if ((peer_subgraph->reused_subgraph != nullptr) &&
      (peer_subgraph->reused_subgraph->engine_conf.id == subgraph->engine_conf.id)) {
    return true;
  }
  return false;
}

// if graph only has hccl and aicpu(not in white list), aicpu need to assign streams here
Status DynamicStreamAllocator::AssignRemainSubgraphNeedAssignStream(const std::vector<SubgraphPtr> &subgraphs) {
  for (const auto &subgraph : subgraphs) {
    GE_CHECK_NOTNULL(subgraph->subgraph_info.GetSubGraph());
    if (StreamUtils::HasAssignedStream(*subgraph)) {
      continue;
    }

    if (StreamUtils::IsEngineSkip(*subgraph)) {
      continue;
    }
    AssignStreamForSubgraph(subgraph);
  }

  return SUCCESS;
}

Status DynamicStreamAllocator::SetSubgraphStreamToNodes(const ComputeGraphPtr &graph,
                                                        const std::vector<SubgraphPtr> &subgraphs) const {
  // Check if all subgraphs have been assigned a stream.
  for (const auto &subgraph : subgraphs) {
    const std::string &engine_name = subgraph->engine_conf.id;
    GELOGI("[Assign][StreamId] %lld for Subgraph %s (engine: %s).", subgraph->stream_id, subgraph->name.c_str(),
           engine_name.c_str());
  }

  // Init the stream id of node.
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    node->GetOpDesc()->SetStreamId(kInvalidStream);
  }

  // Set the stream id of the subgraph to the node.
  for (const auto &subgraph : subgraphs) {
    if (!StreamUtils::HasAssignedStream(*subgraph)) {
      continue;
    }
    const int64_t stream_id = subgraph->stream_id;
    const std::string &engine_name = subgraph->engine_conf.id;
    const auto &compute_graph = subgraph->subgraph_info.GetSubGraph();
    GE_CHECK_NOTNULL(compute_graph);
    for (const NodePtr &node : compute_graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node->GetOpDesc());
      node->GetOpDesc()->SetStreamId(stream_id);
      GELOGD("[Assign][StreamId]id:%lld for Node %s of type %s in subgraph %s (engine: %s).", stream_id,
             node->GetName().c_str(), node->GetType().c_str(), subgraph->name.c_str(), engine_name.c_str());
    }
  }

  return SUCCESS;
}

// different engines' nodes with same stream label can be assigned to same stream
Status DynamicStreamAllocator::ReassignStreamByStreamLabel(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string stream_label;
    if ((!AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, stream_label)) || (stream_label.empty())) {
      continue;
    }
    const auto iter = stream_label_ids_.find(stream_label);
    if (iter == stream_label_ids_.end()) {
      op_desc->SetStreamId(next_stream_);
      stream_label_ids_.emplace(stream_label, next_stream_);
      ++next_stream_;
    } else {
      op_desc->SetStreamId(iter->second);
    }
    GELOGI("Node: %s, stream_label: %s, reassign stream id: %lld.", node->GetName().c_str(), stream_label.c_str(),
           op_desc->GetStreamId());
  }

  return SUCCESS;
}

Status DynamicStreamAllocator::GetNodeNumOfPerStream(const ComputeGraphPtr &graph,
                                                     std::vector<int64_t> &stream_node_num) const {
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    // skip_assign_stream engines' nodes
    if (op_desc->GetStreamId() == kInvalidStream) {
      continue;
    }
    GE_ASSERT_TRUE(op_desc->GetStreamId() < next_stream_, "Node: %s, stream_id: %lld, should less than %lld.",
                   node->GetName().c_str(), op_desc->GetStreamId(), next_stream_);
    stream_node_num[static_cast<size_t>(op_desc->GetStreamId())]++;
  }

  return SUCCESS;
}

// To delete engines whose nodes were all reassigned to other streams because of stream_label.
// Otherwise, final stream_id list may not be continuous.
Status DynamicStreamAllocator::RetainEngineOwingNodes(const ComputeGraphPtr &root_graph) {
  std::vector<int64_t> stream_node_num(next_stream_, kMainStream);
  GE_ASSERT_SUCCESS(GetNodeNumOfPerStream(root_graph, stream_node_num));

  for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
    if (subgraph->GetGraphUnknownFlag()) {
      GE_ASSERT_SUCCESS(GetNodeNumOfPerStream(subgraph, stream_node_num));
    }
  }

  std::map<std::string, int64_t> final_engine_streams;
  for (const auto &engine_stream : engine_streams_) {
    const int64_t stream_id = engine_stream.second;
    GE_ASSERT_TRUE(stream_id >= kMainStream, "Engine: %s, stream_id: %lld, should greater than 0.",
                   engine_stream.first.c_str(), stream_id);
    GE_ASSERT_TRUE(stream_id < next_stream_, "Engine: %s, stream_id: %lld, should less than %lld.",
                   engine_stream.first.c_str(), stream_id, next_stream_);
    if (stream_node_num[stream_id] > 0) {
      final_engine_streams.emplace(engine_stream);
    }
  }
  engine_streams_ = std::move(final_engine_streams);

  return SUCCESS;
}

Status DynamicStreamAllocator::RefreshContinuousStreams(const ComputeGraphPtr &root_graph) {
  GE_ASSERT_SUCCESS(RetainEngineOwingNodes(root_graph));

  const auto &engine_priority = StreamUtils::GetEnginePriority();
  auto compare = [&engine_priority](const std::string &engine1, const std::string &engine2) {
    if (engine_priority.at(engine1) < engine_priority.at(engine2)) {
      return true;
    }
    return engine1 < engine2;
  };
  std::map<std::string, int64_t, decltype(compare)> engine_sort(compare);
  for (const auto &engine_stream : engine_streams_) {
    GE_ASSERT_TRUE(!engine_stream.first.empty(), "Engine name is empty, please check!");
    GE_ASSERT_TRUE(engine_priority.find(engine_stream.first) != engine_priority.end(),
                   "Engine name %s is invaild, please check!", engine_stream.first.c_str());
    engine_sort.emplace(engine_stream.first, engine_stream.second);
  }

  stream_num_ = 0;
  std::map<int64_t, int64_t> old_to_new_streams;
  for (const auto &engine_stream : engine_sort) {
    old_to_new_streams[engine_stream.second] = stream_num_;
    GELOGI("Refresh stream of engine: %s from %lld to %lld.", engine_stream.first.c_str(), engine_stream.second,
           stream_num_);
    ++stream_num_;
  }
  for (const auto &stream_label_id : stream_label_ids_) {
    old_to_new_streams[stream_label_id.second] = stream_num_;
    GELOGI("Refresh stream of label: %s from %lld to %lld.", stream_label_id.first.c_str(), stream_label_id.second,
           stream_num_);
    ++stream_num_;
  }

  if (stream_num_ == 0) {
    GELOGI("None of nodes need to assign stream, stream num is 0, it will cause problem, so change it to 1");
    stream_num_ = 1;
  }

  GE_ASSERT_SUCCESS(RefreshStreamsForGraph(root_graph, old_to_new_streams));
  for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
    if (subgraph->GetGraphUnknownFlag()) {
      GE_ASSERT_SUCCESS(RefreshStreamsForGraph(subgraph, old_to_new_streams));
    }
  }

  return SUCCESS;
}

bool DynamicStreamAllocator::IsForcedAssignMainStream(const NodePtr &node) const {
  const auto &node_type = node->GetType();
  if (OpTypeUtils::IsDataNode(node_type) || (kNodesForcedInMainStream.count(node_type) > 0U) ||
      OpTypeUtils::IsVariableNode(node_type) || OpTypeUtils::IsConstPlaceHolderNode(node_type) ||
      ConstantUtils::IsConstant(node)) {
    GELOGD("Node: %s, type: %s, is forced to assign main stream.", node->GetNamePtr(), node_type.c_str());
    return true;
  }
  return false;
}

Status DynamicStreamAllocator::RefreshStreamsForGraph(const ComputeGraphPtr &graph,
                                                      const std::map<int64_t, int64_t> &old_to_new_streams) {
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    // currently, control flow ops must be in main stream
    // subgraph's intpus(Data) and outputs(Netoutput) must return to main stream
    // put nodes without stream to main stream
    // input nodes without data inputs also need to be in main stream since they would be in Init Graph and
    // execute in load stage.
    if ((op_desc->GetStreamId() == kInvalidStream) || (!op_desc->GetSubgraphInstanceNames().empty()) ||
        old_to_new_streams.empty() || IsForcedAssignMainStream(node)) {
      op_desc->SetStreamId(kMainStream);
    } else {
      const auto iter = old_to_new_streams.find(op_desc->GetStreamId());
      GE_ASSERT_TRUE(iter != old_to_new_streams.end(), "Can not find stream: %lld of %s.", op_desc->GetStreamId(),
                     op_desc->GetName().c_str());
      op_desc->SetStreamId(iter->second);
    }

    stream_nodes_[op_desc->GetStreamId()].emplace_back(node);
    GELOGD("Refresh stream of graph: %s, stream_id: %lld, type: %s, name: %s.", graph->GetName().c_str(),
           op_desc->GetStreamId(), node->GetType().c_str(), node->GetName().c_str());
  }

  return SUCCESS;
}

Status DynamicStreamAllocator::InsertEvents(const ComputeGraphPtr &root_graph) {
  GE_ASSERT_SUCCESS(ConvertEdgesToEvents(root_graph));
  for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
    if (subgraph->GetGraphUnknownFlag()) {
      GE_ASSERT_SUCCESS(ConvertEdgesToEvents(subgraph));
    }
  }

  GE_ASSERT_SUCCESS(OptimizeEvents());

  uint32_t tmp_event = 0U;
  GE_ASSERT_SUCCESS(StreamUtils::RefreshContinuousEvents(tmp_event, node_to_send_events_, node_to_recv_events_));
  event_num_ = static_cast<int64_t>(tmp_event);

  GE_ASSERT_SUCCESS(SetEventAttrToNodes(root_graph));
  for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
    if (subgraph->GetGraphUnknownFlag()) {
      GE_ASSERT_SUCCESS(SetEventAttrToNodes(subgraph));
    }
  }

  return SUCCESS;
}

Status DynamicStreamAllocator::ConvertEdgesToEvents(const ComputeGraphPtr &graph) {
  for (const auto &cur_node : graph->GetDirectNode()) {
    // Take the adjacent points, then judge whether need to insert the event
    for (const OutDataAnchorPtr &anchor : cur_node->GetAllOutDataAnchors()) {
      for (const InDataAnchorPtr &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
        NodePtr next_node = peer_in_anchor->GetOwnerNode();
        GE_ASSERT_SUCCESS(InsertEventBetweenTwoNodes(cur_node, next_node), "Insert event from %s to %s failed.",
                          cur_node->GetName().c_str(), next_node->GetName().c_str());
      }
    }

    /// If the two nodes of the control side belong to two streams,
    /// you also need to add the send/recv event.
    if (cur_node->GetOutControlAnchor() != nullptr) {
      for (const auto peer_in_anchor : cur_node->GetOutControlAnchor()->GetPeerAnchorsPtr()) {
        NodePtr next_node = peer_in_anchor->GetOwnerNode();
        GE_ASSERT_SUCCESS(InsertEventBetweenTwoNodes(cur_node, next_node), "Insert event from %s to %s failed.",
                          cur_node->GetName().c_str(), next_node->GetName().c_str());
      }
    }
  }

  return SUCCESS;
}

Status DynamicStreamAllocator::InsertEventBetweenTwoNodes(const NodePtr &cur_node, const NodePtr &next_node) {
  const auto &cur_desc = cur_node->GetOpDesc();
  GE_CHECK_NOTNULL(cur_desc);
  const auto &next_desc = next_node->GetOpDesc();
  GE_CHECK_NOTNULL(next_desc);

  const int64_t cur_stream_id = cur_desc->GetStreamId();
  GE_ASSERT_TRUE(cur_stream_id != kInvalidStream, "Node: %s, stream id is invalid", cur_node->GetName().c_str());
  const int64_t next_stream_id = next_desc->GetStreamId();
  GE_ASSERT_TRUE(next_stream_id != kInvalidStream, "Node: %s, stream id is invalid", next_node->GetName().c_str());

  // No need to insert events between nodes in the same stream.
  if (cur_stream_id == next_stream_id) {
    return SUCCESS;
  }

  // Add send and receive events.
  StreamUtils::AddSendEventId(cur_node, static_cast<uint32_t>(event_num_), node_to_send_events_);
  StreamUtils::AddRecvEventId(next_node, static_cast<uint32_t>(event_num_), node_to_recv_events_);
  ++event_num_;
  GELOGI("Insert event %lld between node %s(stream %lld) and %s(stream %lld)", event_num_, cur_node->GetName().c_str(),
         cur_stream_id, next_node->GetName().c_str(), next_stream_id);

  return SUCCESS;
}

Status DynamicStreamAllocator::OptimizeEvents() {
  for (auto &stream_nodes : stream_nodes_) {
    std::sort(stream_nodes.second.begin(), stream_nodes.second.end(), TopoOrderCompare);
  }

  GE_ASSERT_SUCCESS(StreamUtils::OptimizeBySendEvents(stream_nodes_, node_to_send_events_, node_to_recv_events_));

  GE_ASSERT_SUCCESS(StreamUtils::OptimizeByRecvEvents(stream_nodes_, node_to_send_events_, node_to_recv_events_));

  return SUCCESS;
}

Status DynamicStreamAllocator::SetEventAttrToNodes(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    const auto send_event_ids = StreamUtils::GetSyncIdList(node, node_to_send_events_);
    if (!send_event_ids.empty()) {
      GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, ATTR_NAME_SEND_EVENT_IDS, send_event_ids),
                     "Set attr _send_event_ides to %s failed, send_event_ids count: %zu.", op_desc->GetName().c_str(),
                     send_event_ids.size());
      GELOGD("Node: %s, send_event_ids size: %zu.", op_desc->GetName().c_str(), send_event_ids.size());
    }

    const auto recv_event_ids = StreamUtils::GetSyncIdList(node, node_to_recv_events_);
    if (!recv_event_ids.empty()) {
      GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, ATTR_NAME_RECV_EVENT_IDS, recv_event_ids),
                     "Set attr _recv_event_ides to %s failed, recv_event_ids count: %zu.", op_desc->GetName().c_str(),
                     recv_event_ids.size());
      GELOGD("Node: %s, recv_event_ids size: %zu.", op_desc->GetName().c_str(), recv_event_ids.size());
    }
  }

  return SUCCESS;
}
}  // namespace ge
