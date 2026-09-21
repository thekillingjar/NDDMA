/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/engine_partitioner.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <stack>
#include "analyzer/analyzer.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/op/ge_op_utils.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/build/stream/stream_utils.h"
#include "common/checker.h"
#include "graph/ge_context.h"
#include "api/gelib/gelib.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/ge_common/ge_types.h"

namespace ge {
namespace {
const char_t *const kEngineDefaultData = "ENGINE_DEFAULT_DATA";
const char_t *const kEndType = "End";
const char_t *const kPlaceHolderType = "PlaceHolder";
const char_t *const kPeerIndex = "peerIndex";
const char_t *const kParentOpType = "parentOpType";
const char_t *const kParentNode = "parentNode";
const char_t *const kPeerNodeName = "_peerNodeName";
const char_t *const kParentNodeName = "_parentNodeName";
const char_t *const kParentId = "parentId";
const char_t *const kAnchorIndex = "anchorIndex";
const char_t *const kTaskL2FusionInfo = "_task_L2FusionInfo";
const char_t *const kDataAnchorIndexForLxfusion = "_data_anchor_index_for_lxfusion";
const char_t *const kEnableCvParallel = "_enable_cv_parallel";
const char_t *const kVectorEngineName = "VectorEngine";
const std::string kStableRdfsSort = "3";
const int32_t kOneGraph = 1;  // only one graph
const int32_t kRankOne = 1;   // order of graph list is 0,1,2,3..., 1 means second order
const int32_t kRankZero = 0;  // order of graph list is 0,1,2,3..., 0 means first order
const int64_t kOverflowDefaultValue = -1;
struct DeviceIndex {
  std::string engine_type;
  std::vector<int32_t> indices;
  std::string DebugString() const {return engine_type + ToString(indices);};
  bool operator==(const DeviceIndex &rhs) const{
    return engine_type == rhs.engine_type && indices == rhs.indices;
  };
  bool operator!=(const DeviceIndex &rhs) const{
    return !(*this == rhs);
  };
  bool operator<(const DeviceIndex &rhs) const{
    if (engine_type < rhs.engine_type) {
      return true;
    }
    if (rhs.engine_type < engine_type) {
      return false;
    }
    return indices < rhs.indices;
  };
};

std::string GenClusterEngineName(const NodePtr &node,
    EnginePartitioner::Mode mode, const NodeEngineMap &engine_map) {
  auto engine_name = engine_map.at(node);
  // 流分配时，自定义算子需要跟aicore算子一条流
  if ((mode == EnginePartitioner::Mode::kSecondPartitioning) && (engine_name == kEngineNameCustom)) {
    // 临时改动，后续需要从用户的注册引擎信息里面获取此处的自定义算子挂靠的引擎名字
    engine_name = kEngineNameAiCore;
  }
  return engine_name;
}

bool IsLinkedInGraph(const NodePtr &src_node, const NodePtr &dst_node) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    return false;
  }
  const auto src_node_id = src_node->GetOpDesc()->GetId();
  const auto dst_node_id = dst_node->GetOpDesc()->GetId();
  if (src_node_id > dst_node_id) {
    return false;
  }

  std::stack<NodePtr> node_stack;
  std::unordered_set<NodePtr> visited;  // 记录已访问节点，避免循环

  node_stack.push(src_node);
  visited.insert(src_node);

  while (!node_stack.empty()) {
    NodePtr current_node = node_stack.top();
    node_stack.pop();

    for (const auto &node : current_node->GetOutAllNodes()) {
      if (node == dst_node) {
        return true;
      }
      if (node->GetOpDesc()->GetId() > dst_node_id) {
        continue;
      }
      if (visited.find(node) == visited.end()) {
        visited.insert(node);
        node_stack.push(node);
      }
    }
  }

  return false;
}

bool IsAivNode(const NodePtr &node) {
  std::string core_type;
  const auto op_desc = node->GetOpDesc();
  bool is_aiv = false;
  if (ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type)) {
    if ((core_type == kTaskTypeAiv) || (core_type == kTaskTypeMixAiv)) {
      is_aiv = true;
    } else {
      if ((core_type == "MIX")) {
        (void)ge::AttrUtils::GetBool(op_desc, "_mix_is_aiv", is_aiv);
      }
    }
  }
  return is_aiv;
}

bool IsAicNode(const NodePtr &node) {
  std::string core_type;
  const auto op_desc = node->GetOpDesc();
  if (ge::AttrUtils::GetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type)) {
    return !IsAivNode(node);
  }
  return false;
}

// 根据topo序找到aiv前后相邻的aic节点，如果aiv和其中至少1个aic在图上没有通路则可以并发
void MarkCvParallelAivNodes(const ComputeGraphPtr &graph) {
  if (!StreamUtils::EnableCvParallel()) {
    return;
  }
  std::map<NodePtr, std::pair<NodePtr, NodePtr>> aiv_to_adjacent_aic_nodes;
  NodePtr pre_aic = nullptr;
  std::vector<decltype(aiv_to_adjacent_aic_nodes.begin())> aiv_iters;
  for (auto node : graph->GetDirectNode()) {
    if (IsAivNode(node)) {
      auto insert_ret = aiv_to_adjacent_aic_nodes.insert({node, {pre_aic, nullptr}});
      aiv_iters.emplace_back(insert_ret.first);
    }
    if (IsAicNode(node)) {
      for (auto iter : aiv_iters) {
        if ((iter != aiv_to_adjacent_aic_nodes.end()) && (iter->second.second == nullptr)) {
          iter->second.second = node;
        }
      }
      aiv_iters.clear();
      pre_aic = node;
    }
  }
  for (auto aiv_iter : aiv_to_adjacent_aic_nodes) {
    auto aiv_node = aiv_iter.first;
    auto pre_aic_node = aiv_iter.second.first;
    auto after_aic_node = aiv_iter.second.second;
    bool is_pre_aic_link_to_aiv = IsLinkedInGraph(pre_aic_node, aiv_node);
    if (is_pre_aic_link_to_aiv && (after_aic_node == nullptr)) {
      continue;
    }
    if (!is_pre_aic_link_to_aiv || !IsLinkedInGraph(aiv_node, after_aic_node)) {
      AttrUtils::SetBool(aiv_node->GetOpDesc(), kEnableCvParallel, true);
      GELOGD("node %s set cv parallel", aiv_node->GetNamePtr());
    }
    GELOGD("pre aic %s, current aiv %s, after aic %s", (pre_aic_node == nullptr) ? nullptr : pre_aic_node->GetNamePtr(),
           aiv_node->GetNamePtr(), (after_aic_node == nullptr) ? nullptr : after_aic_node->GetNamePtr());
  }
}
}  // namespace
Status ge::EnginePartitioner::CheckValidIfEnd2PldEmpty(const GraphPartitionInfo &graph_info,
                                                       ge::ComputeGraphPtr &output_merged_compute_graph) const {
  // only one condition:no data node, one engine, there is only one graph + input graph
  if (graph_info.partitions_.size() == kOneGraph) {
    const auto &partition = (*graph_info.partitions_.begin());
    if (partition.first == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "partition.first is nullptr, check invalid, engine name is %s",
                         partition.second.c_str());
      GELOGE(GE_GRAPH_EMPTY_PARTITION, "[Check][Param] partition.first is null, engine name is %s",
             partition.second.c_str());
      return FAILED;
    }
    output_merged_compute_graph = partition.first;
  } else {  // if placeholder to end map is empty, it should be an exception condition
    REPORT_INNER_ERR_MSG("E19999", "partitions size:%zu is not 1, check invalid.", graph_info.partitions_.size());
    GELOGE(GE_GRAPH_EMPTY_PARTITION, "[Check][Param] placeholder to end map is empty, partitions size:%zu is not 1.",
           graph_info.partitions_.size());
    return FAILED;
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::MergeAllSubGraph(ge::ComputeGraphPtr &output_merged_compute_graph,
                                               const std::vector<SubGraphInfoPtr> &sub_graph_list,
                                               const GraphPartitionInfo &graph_info) const {
  for (size_t rank = 0UL; rank < graph_info.rank_2_partitions_.size(); rank++) {
    std::string temp_stream;
    // sub_graph_list index is one ahead of rank_2_partitions_list index
    if (rank > 0UL) {
      temp_stream = sub_graph_list[rank - 1UL]->GetStreamLabel();
    }
    for (const auto &node : graph_info.rank_2_partitions_[rank]->GetDirectNode()) {
      if (node == nullptr) {
        continue;
      }
      if ((node->GetType() == kEndType) || (node->GetType() == kPlaceHolderType)) {
        continue;
      }
      if ((!temp_stream.empty()) && (!AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL))) {
        (void)AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, temp_stream);
      }
      GE_ASSERT_GRAPH_SUCCESS(node->SetOwnerComputeGraph(output_merged_compute_graph),
                              "[Set][OwnerComputeGraph] failed, node %s", node->GetName().c_str());
      (void)output_merged_compute_graph->AddNode(node);
    }
  }
  // get session graph id from subgraph
  GE_ASSERT_SUCCESS(SetMergedGraphId(output_merged_compute_graph, graph_info),
                    "[Call][SetMergedGraphId] failed, graph:%s", output_merged_compute_graph->GetName().c_str());
  return SUCCESS;
}

Status ge::EnginePartitioner::SetMergedGraphId(const ge::ComputeGraphPtr &output_merged_compute_graph,
                                               const GraphPartitionInfo &graph_info) const {
  std::string session_graph_id;
  // get session graph id from subgraph
  if (graph_info.rank_2_partitions_.empty() ||
      !AttrUtils::GetStr(*(graph_info.rank_2_partitions_[0U]), ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Get graph session_graph_id attr failed.");
  }
  // set session graph id into merged subgraph
  if (!session_graph_id.empty()) {
    GELOGI("Set session graph id %s in merged compute graph", session_graph_id.c_str());
    // private function, promise output_merged_compute_graph not null
    GE_ASSERT_TRUE(AttrUtils::SetStr(*output_merged_compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                   "SetStr ATTR_NAME_SESSION_GRAPH_ID[%s] failed of graph:%s.", session_graph_id.c_str(),
                   output_merged_compute_graph->GetName().c_str());
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::RemoveNodeAndEdgeBetweenEndPld(ge::ComputeGraphPtr &output_merged_compute_graph,
                                                             const std::vector<SubGraphInfoPtr> &sub_graph_list,
                                                             const GraphPartitionInfo &graph_info) {
  GE_ASSERT_NOTNULL(output_merged_compute_graph, "[Check][Input] failed, output_merged_compute_graph is null.");
  GE_ASSERT_SUCCESS(MergeAllSubGraph(output_merged_compute_graph, sub_graph_list, graph_info),
                    "[Merge][AllSubGraph] failed.");
  for (const auto &it : graph_info.index_2_end_) {
    const auto &end = it.second;
    const auto &pld = graph_info.end_2_pld_.at(it.second);
    if ((end != nullptr) &&
        (pld != nullptr) &&
        (end->GetInDataAnchor(0) != nullptr) &&
        (pld->GetOutDataAnchor(0) != nullptr)) {
      AnchorPtr end_in_anchor = (end->GetInDataAnchor(0)->GetFirstPeerAnchor() == nullptr)
                                  ? Anchor::DynamicAnchorCast<Anchor>(end->GetInControlAnchor())
                                  : Anchor::DynamicAnchorCast<Anchor>(end->GetInDataAnchor(0));
      AnchorPtr pld_out_anchor = (pld->GetOutDataAnchor(0)->GetFirstPeerAnchor() == nullptr)
                                   ? Anchor::DynamicAnchorCast<Anchor>(pld->GetOutControlAnchor())
                                   : Anchor::DynamicAnchorCast<Anchor>(pld->GetOutDataAnchor(0));
      GE_CHECK_NOTNULL(end_in_anchor);
      auto src_anchor = end_in_anchor->GetFirstPeerAnchor();  // src_anchor should be only 1
      GE_CHECK_NOTNULL(src_anchor);
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(src_anchor, end_in_anchor),
                              "[Remove][Edge] between %s and %s failed. node_name:%s, graph_name:%s",
                              src_anchor->GetOwnerNode()->GetName().c_str(),
                              end_in_anchor->GetOwnerNode()->GetName().c_str(), end->GetName().c_str(),
                              end->GetOwnerComputeGraph()->GetName().c_str());
      GE_CHECK_NOTNULL(pld_out_anchor);
      for (const auto &peer_in_anchor : pld_out_anchor->GetPeerAnchors()) {
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(pld_out_anchor, peer_in_anchor),
                                "[Remove][Edge] between %s and %s failed. node_name:%s, graph_name:%s",
                                pld_out_anchor->GetOwnerNode()->GetName().c_str(),
                                peer_in_anchor->GetOwnerNode()->GetName().c_str(), pld->GetName().c_str(),
                                pld->GetOwnerComputeGraph()->GetName().c_str());
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(src_anchor, peer_in_anchor),
                                "[Add][Edge] from %s to %s failed.",
                                src_anchor->GetOwnerNode()->GetName().c_str(),
                                peer_in_anchor->GetOwnerNode()->GetName().c_str());
      }
      NodeUtils::UnlinkAll(*pld);
      NodeUtils::UnlinkAll(*end);
    } else {
      GELOGW("End or pld is nullptr or in data anchor of end is nullptr or out data anchor of pld is nullptr");
    }
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::MergeOverflowAttr(const ge::ComputeGraphPtr &sub_graph,
                                                ge::ComputeGraphPtr &root_graph) const {
  GE_CHECK_NOTNULL(sub_graph);
  GE_CHECK_NOTNULL(root_graph);
  if (AttrUtils::HasAttr(sub_graph, GLOBALWORKSPACE_TYPE) &&
      !AttrUtils::HasAttr(root_graph, GLOBALWORKSPACE_TYPE)) {
    (void)AttrUtils::SetInt(root_graph, GLOBALWORKSPACE_TYPE, global_workspace_type_);
    (void)AttrUtils::SetInt(root_graph, "globalworkspace_size", global_workspace_size_);
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::MergeAfterSubGraphOptimization(ge::ComputeGraphPtr &output_merged_compute_graph,
                                                             const ge::ComputeGraphPtr &original_compute_graph,
                                                             EnginePartitioner::Mode mode) {
  current_mode_ = mode;
  // Assign engine for each node in the graph
  DNNEngineManager::GetInstance().InitPerformanceStatistic();
  (void)mode;
  Status real_ret = SUCCESS;
  auto ret = MergeSubGraph(output_merged_compute_graph, original_compute_graph);
  if (ret != SUCCESS) {
    // even though failed, ensure all op do finish check support
    real_ret = FAILED;
    GELOGE(ret, "[Merge][SubGraph] Failed, ret:%d", ret);
  }
  GE_CHECK_NOTNULL(original_compute_graph);
  output_merged_compute_graph->SetName(original_compute_graph->GetName());
  // merge sub graph
  for (const auto &sub_graph : original_compute_graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(sub_graph);
    bool no_need_merge = false;
    (void)ge::AttrUtils::GetBool(sub_graph, ATTR_NAME_NO_NEED_MERGE, no_need_merge);
    if (no_need_merge) {
      GELOGI("sub graph %s no need merge, skip it", sub_graph->GetName().c_str());
      continue;
    }
    ComputeGraphPtr merged_sub_graph = nullptr;
    ret = MergeSubGraph(merged_sub_graph, sub_graph);
    if (ret != SUCCESS) {
      real_ret = FAILED;
      GELOGE(ret, "[Merge][SubGraph] Failed, ret:%d", ret);
      continue;
    }
    // this means subgraph added in optimize subgraph and without partitions, so just add to root graph
    if (merged_sub_graph == sub_graph) {
      GELOGI("Just add subgraph %s (parent node is %s) to root graph %s.", sub_graph->GetName().c_str(),
             sub_graph->GetParentNode()->GetName().c_str(), output_merged_compute_graph->GetName().c_str());
      sub_graph->SetParentGraph(sub_graph->GetParentNode()->GetOwnerComputeGraph());
      GE_ASSERT_GRAPH_SUCCESS(output_merged_compute_graph->AddSubgraph(sub_graph->GetName(), merged_sub_graph),
                              "[Call][AddSubgraph] failed, subgraph:%s, merged subgraph:%s",
                              sub_graph->GetName().c_str(), merged_sub_graph->GetName().c_str());
      continue;
    }
    // add sub graph
    merged_sub_graph->SetName(sub_graph->GetName());
    merged_sub_graph->SetInputSize(sub_graph->GetInputSize());
    merged_sub_graph->SetOutputSize(sub_graph->GetOutputSize());
    const auto &parent_node = sub_graph->GetParentNode();
    GE_ASSERT_NOTNULL(parent_node, "[Check][Param] Parent node is null, graph name is %s",
                      sub_graph->GetName().c_str());
    const auto &original_graph = parent_node->GetOwnerComputeGraph();
    GE_ASSERT_TRUE(graph_2_graph_partition_info_.find(original_graph) != graph_2_graph_partition_info_.end(),
                   "[Check][Param] Find graph info failed, graph name is %s",
                   original_graph->GetName().c_str());
    auto &graph_info = graph_2_graph_partition_info_[original_graph];
    GE_ASSERT_TRUE(graph_info.corresponding_node_in_partitions_.count(parent_node->GetName()) != 0U,
                   "[Check][Param] Find corresponding node failed, parent node name is %s",
                   parent_node->GetName().c_str());
    const auto &corresponding_node = graph_info.corresponding_node_in_partitions_[parent_node->GetName()];
    GE_ASSERT_NOTNULL(corresponding_node,
                      "[Check][Param] Get null node in corresponding_node_in_partitions_, parent node name is %s",
                      parent_node->GetName().c_str());
    merged_sub_graph->SetParentNode(corresponding_node);
    merged_sub_graph->SetParentGraph(corresponding_node->GetOwnerComputeGraph());

    // merge overflow detection attr to root_graph
    if (MergeOverflowAttr(merged_sub_graph, output_merged_compute_graph) != SUCCESS) {
      return FAILED;
    }
    GE_ASSERT_GRAPH_SUCCESS(output_merged_compute_graph->AddSubgraph(sub_graph->GetName(), merged_sub_graph),
                            "[Call][AddSubgraph] failed, subgraph:%s, merged subgraph:%s.",
                            sub_graph->GetName().c_str(), merged_sub_graph->GetName().c_str());
  }
  DNNEngineManager::GetInstance().LogCheckSupportCost();
  ClearAllPartitionData();
  if (real_ret != SUCCESS) {
    auto root_graph = ge::GraphUtils::FindRootGraph(original_compute_graph);
    GE_CHECK_NOTNULL(root_graph);
    (void)Analyzer::GetInstance()->SaveAnalyzerDataToFile(root_graph->GetSessionID(), root_graph->GetGraphID());
  }
  return real_ret;
}

Status ge::EnginePartitioner::FindOverflowAttr(const ge::ComputeGraphPtr &sub_graph,
                                               ge::ComputeGraphPtr &original_graph) {
  GE_CHECK_NOTNULL(sub_graph);
  GE_CHECK_NOTNULL(original_graph);
  if (AttrUtils::HasAttr(original_graph, GLOBALWORKSPACE_TYPE)) {
    return SUCCESS;
  }
  for (const auto &node : sub_graph->GetDirectNode()) {
    (void)AttrUtils::GetInt(node->GetOpDesc(), GLOBALWORKSPACE_TYPE, global_workspace_type_);
    (void)AttrUtils::GetInt(node->GetOpDesc(), "globalworkspace_size", global_workspace_size_);

    if ((global_workspace_type_ == kOverflowDefaultValue) || (global_workspace_size_ == kOverflowDefaultValue)) {
      continue;
    }

    (void)AttrUtils::SetInt(original_graph, GLOBALWORKSPACE_TYPE, global_workspace_type_);
    (void)AttrUtils::SetInt(original_graph, "globalworkspace_size", global_workspace_size_);
    break;
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::MergeSubGraph(ge::ComputeGraphPtr &output_merged_compute_graph,
                                            const ge::ComputeGraphPtr &original_compute_graph) {
  GE_ASSERT_NOTNULL(original_compute_graph, "[Check][Param] original_compute_graph is nullptr.");
  if ((graph_2_graph_partition_info_.find(original_compute_graph) == graph_2_graph_partition_info_.end()) ||
      (graph_2_subgraph_list_.find(original_compute_graph) == graph_2_subgraph_list_.end())) {
    GELOGW("[GraphPartition]: compute_graph has not found, just return original.");
    output_merged_compute_graph = original_compute_graph;
    return SUCCESS;
  }
  GraphPartitionInfo &graph_info = graph_2_graph_partition_info_[original_compute_graph];
  const auto &sub_graph_list = graph_2_subgraph_list_[original_compute_graph];

  GE_ASSERT_TRUE(graph_info.mode_ == Mode::kMerging, "[Check][Param] Cannot call merging in partition mode, as mode != %d",
                 Mode::kMerging);
  GELOGD("Graph merge starts.");
  ComputeGraphPtr new_sub_graph = MakeShared<ComputeGraph>(original_compute_graph->GetName());
  GE_CHECK_NOTNULL(new_sub_graph);
  // check input param
  for (const auto &it : sub_graph_list) {
    GE_ASSERT_NOTNULL(it, "[Check][Param] merging sub-graphs failed, sub-graph is nullptr");
    // Get overflow detection attr
    if (FindOverflowAttr(it->GetSubGraph(), new_sub_graph) != SUCCESS) {
      return FAILED;
    }
  }
  bool is_map_empty = graph_info.end_2_pld_.empty() || graph_info.pld_2_end_.empty();
  if (is_map_empty) {
    if (CheckValidIfEnd2PldEmpty(graph_info, output_merged_compute_graph) != SUCCESS) {
      return FAILED;
    }
  }
  output_merged_compute_graph = new_sub_graph;
  GE_TRACE_START(MergeSubGraphRemoveNode);
  GE_ASSERT_GRAPH_SUCCESS(RemoveNodeAndEdgeBetweenEndPld(output_merged_compute_graph, sub_graph_list, graph_info),
                          "[Call][RemoveNodeAndEdgeBetweenEndPld] failed, graph:%s",
                          output_merged_compute_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(MergeSubGraphRemoveNode, "EnginePartitioner::MergeGraphRemoveNodeAndEdge");
  // flush all nodes' engine of merged graph
  GE_TRACE_START(MergeSubGraphEnginePlacerRun);
  engine_placer_.SetComputeGraph(output_merged_compute_graph);
  GE_CHK_STATUS_RET(engine_placer_.Run(), "[Call][Run] engine_placer run failed, graph:%s",
                    output_merged_compute_graph->GetName().c_str());
  GE_CHK_STATUS_RET(InheritOriginalAttr(original_compute_graph, output_merged_compute_graph),
                    "[Inherit][OriginalAttr] failed, graph:%s", output_merged_compute_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(MergeSubGraphEnginePlacerRun, "EnginePartitioner::MergeGraphEnginePlacerRun");
  return UpdateCorrespondNodeInPartitions(output_merged_compute_graph, graph_info);
}

Status EnginePartitioner::InheritOriginalAttr(const ComputeGraphPtr &original_compute_graph,
                                              ComputeGraphPtr &output_merged_compute_graph) const {
  if (original_compute_graph->GetGraphUnknownFlag()) {
    output_merged_compute_graph->SetGraphUnknownFlag(true);
    for (const auto &node : output_merged_compute_graph->GetDirectNode()) {
      ge::AttrUtils::SetBool(node->GetOpDesc(), "OwnerGraphIsUnknown", true);
      GELOGD("Set OwnerGraphIsUnknow attr to node[%s], graph [%s]",
             node->GetName().c_str(), output_merged_compute_graph->GetName().c_str());
    }
  }
  const std::map<string, GeAttrValue> &original_attrs = AttrUtils::GetAllAttrs(original_compute_graph);
  for (auto const &attr_iter : original_attrs) {
    if (output_merged_compute_graph->TrySetAttr(attr_iter.first, attr_iter.second) != GRAPH_SUCCESS) {
      GELOGW("Set inherit original attr[%s] failed, Please Check.", attr_iter.first.c_str());
    }
  }
  auto *device_mapping = original_compute_graph->GetExtAttr<std::map<DeviceIndex, std::vector<int32_t>>>(
      ge::ATTR_NAME_DEVICE_INDEX_TO_LOGIC_DEVICE_ID);
  if (device_mapping != nullptr) {
    GE_ASSERT_TRUE(
        output_merged_compute_graph->SetExtAttr(ge::ATTR_NAME_DEVICE_INDEX_TO_LOGIC_DEVICE_ID, *device_mapping));
  }

  // AttrStore里面属性组没有被拷贝，并且没有提供CopyAllAttrStore方法，暂时先手动拷贝必须的
  auto origin_shape_env_attr = original_compute_graph->GetAttrsGroup<ShapeEnvAttr>();
  if (origin_shape_env_attr != nullptr) {
    auto shape_env_attr = output_merged_compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
    GE_ASSERT_NOTNULL(shape_env_attr);
    *shape_env_attr = *origin_shape_env_attr;
  }
  return SUCCESS;
}

graphStatus ge::EnginePartitioner::UpdatePldOpDesc(const NodePtr &dst_node, int32_t input_index,
                                                   const OpDescPtr &pld_op_desc) const {
  GE_ASSERT_NOTNULL(dst_node, "[Check][Param] parameter dst_node is null.");
  GE_ASSERT_NOTNULL(pld_op_desc, "[Check][Param] parameter pld_op_desc is null.");
  GE_ASSERT_NOTNULL(dst_node->GetOpDesc(), "[Check][Param] parameter dst_node opdesc is null.");
  const auto &input_desc = dst_node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(input_index));
  GE_ASSERT_GRAPH_SUCCESS(pld_op_desc->AddOutputDesc(input_desc), "[Add][OutputDesc] to op:%s failed",
                          pld_op_desc->GetName().c_str());
  const auto &pld_op = pld_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(pld_op, "[Check][Param] output(0) of op:%s is nullptr.", pld_op_desc->GetName().c_str());
  ge::TensorUtils::SetRealDimCnt(*(pld_op_desc->MutableOutputDesc(0).get()),
                                 static_cast<uint32_t>(input_desc.GetShape().GetDims().size()));
  return GRAPH_SUCCESS;
}

graphStatus ge::EnginePartitioner::UpdateEndOpDesc(const NodePtr &src_node, int32_t output_index,
                                                   const OpDescPtr &end_op_desc) const {
  GE_ASSERT_NOTNULL(src_node, "[Check][Param] src_node is null.");
  GE_ASSERT_NOTNULL(src_node->GetOpDesc(), "[Check][Param] src_op_desc is null.");
  GE_ASSERT_NOTNULL(end_op_desc, "[Check][Param] end_op_desc is null.");
  const auto &output_desc = src_node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(output_index));
  GE_ASSERT_GRAPH_SUCCESS(end_op_desc->AddInputDesc(output_desc), "[Add][InputDesc] to op:%s failed",
                          end_op_desc->GetName().c_str());
  const auto &end_op_input_tensor0 = end_op_desc->MutableInputDesc(0);
  GE_ASSERT_NOTNULL(end_op_input_tensor0, "[Check][Param] input(0) of op:%s is nullptr.",
                    end_op_desc->GetName().c_str());
  ge::TensorUtils::SetRealDimCnt(*(end_op_desc->MutableInputDesc(0).get()),
                                 static_cast<uint32_t>(output_desc.GetShape().GetDims().size()));
  return GRAPH_SUCCESS;
}

graphStatus ge::EnginePartitioner::MakeEndOpNode(const AnchorPtr &out_anchor, const ge::ComputeGraphPtr &end_graph,
                                                 NodePtr &new_end_node) {
  std::string end_name = kEndType + std::to_string(graph_info_.num_of_pld_end_);
  auto end_op_desc = MakeShared<OpDesc>(end_graph->GetName() + "_" + end_name, END);
  GE_CHECK_NOTNULL(end_op_desc);
  // replace input_desc of end with owner node's desc
  int32_t output_index = ge::AnchorUtils::GetIdx(out_anchor);
  bool is_need_update_desc = (output_index >= 0) && ((graph_info_.mode_ == Mode::kAtomicEnginePartitioning) ||
                                                     (graph_info_.mode_ == Mode::kCompositeEnginePartitioning));
  if (is_need_update_desc) {
    GE_ASSERT_GRAPH_SUCCESS(UpdateEndOpDesc(out_anchor->GetOwnerNode(), output_index, end_op_desc),
                            "[Update][EndOpDesc] failed, input index:%d, end_op_desc:%s", output_index,
                            end_op_desc->GetName().c_str());
  } else {
    GeTensorDesc input_desc;
    GE_ASSERT_GRAPH_SUCCESS(end_op_desc->AddInputDesc(input_desc), "[Add][InputDesc] to op:%s failed, input index %d",
                            end_op_desc->GetName().c_str(), output_index);
  }
  new_end_node = end_graph->AddNode(end_op_desc);
  return GRAPH_SUCCESS;
}

graphStatus ge::EnginePartitioner::MakePldOpNode(const AnchorPtr &peer_in_anchor, const NodePtr &src_node,
                                                 const ge::ComputeGraphPtr &pld_graph, NodePtr &new_pld_node) {
  /// For fe, op id has been set in AddNode,
  /// we can take op id of srcNode as the mark of parentId now
  const auto &src_node_op_desc = src_node->GetOpDesc();
  GE_CHECK_NOTNULL(src_node_op_desc);
  const std::string pld_name = kPlaceHolderType + std::to_string(graph_info_.num_of_pld_end_);
  auto pld_op_desc = MakeShared<OpDesc>(pld_graph->GetName() + "_" + pld_name, PLACEHOLDER);
  GE_CHECK_NOTNULL(pld_op_desc);
  // replace output_desc of pld with input node's output desc
  int32_t input_index = ge::AnchorUtils::GetIdx(peer_in_anchor);
  bool is_need_update_desc = (input_index >= 0) && ((graph_info_.mode_ == Mode::kAtomicEnginePartitioning) ||
                                                    (graph_info_.mode_ == Mode::kCompositeEnginePartitioning));
  if (is_need_update_desc) {
    GE_ASSERT_GRAPH_SUCCESS(UpdatePldOpDesc(peer_in_anchor->GetOwnerNode(), input_index, pld_op_desc),
                            "[Update][PldOpDesc] failed, output index:%d, pld_op_desc:%s", input_index,
                            pld_op_desc->GetName().c_str());
  } else {
    GeTensorDesc output_desc;
    GE_ASSERT_GRAPH_SUCCESS(pld_op_desc->AddOutputDesc(output_desc),
                            "[Add][OutputDesc] to op:%s failed, input index %d", pld_op_desc->GetName().c_str(),
                            input_index);
  }
  new_pld_node = pld_graph->AddNode(pld_op_desc);
  return GRAPH_SUCCESS;
}

graphStatus ge::EnginePartitioner::SetPldOpAttr(const NodePtr &src_node, const NodePtr &new_end_node,
                                                const ge::ComputeGraphPtr &end_graph, const AnchorPtr &out_anchor,
                                                const OpDescPtr &pld_op_desc) const {
  int64_t node_id = src_node->GetOpDesc()->GetId();
  auto src_node_op_desc = src_node->GetOpDesc();
  GE_ASSERT_TRUE(AttrUtils::SetInt(pld_op_desc, kPeerIndex, graph_info_.num_of_pld_end_),
                 "SetInt peerIndex failed of op:%s.", pld_op_desc->GetName().c_str());
  GE_ASSERT_TRUE(AttrUtils::SetStr(pld_op_desc, kParentOpType, src_node->GetType()),
                 "SetStr parentOpType failed of op:%s.", pld_op_desc->GetName().c_str());
  GE_ASSERT_TRUE(AttrUtils::SetStr(pld_op_desc, kParentNodeName, src_node->GetName()),
                 "SetStr parentOpName failed of op:%s.", pld_op_desc->GetName().c_str());
  GE_ASSERT_TRUE(pld_op_desc->SetExtAttr(kParentNode, src_node), "SetPldExtAttr parentNode failed of op:%s.",
                 pld_op_desc->GetName().c_str());
  GE_ASSERT_TRUE(
      AttrUtils::SetStr(pld_op_desc, ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, src_node_op_desc->GetOpEngineName()),
      "SetStr frontNodeEngineName failed of op:%s.", pld_op_desc->GetName().c_str());
  std::string l2_info_attr;
  if (AttrUtils::GetStr(src_node_op_desc, kTaskL2FusionInfo, l2_info_attr)) {
    GE_ASSERT_TRUE(AttrUtils::SetStr(pld_op_desc, kTaskL2FusionInfo, l2_info_attr),
                   "SetStr l2_info_attr failed of op:%s.", src_node_op_desc->GetName().c_str());
  }
  int64_t anchor_index_for_lxfusion;
  if (AttrUtils::GetInt(src_node_op_desc, kDataAnchorIndexForLxfusion, anchor_index_for_lxfusion)) {
    GE_ASSERT_TRUE(AttrUtils::SetInt(pld_op_desc, kDataAnchorIndexForLxfusion, anchor_index_for_lxfusion),
                   "SetInt anchor_index_for_lxfusion failed");
  }
  GE_ASSERT_TRUE(AttrUtils::SetStr(pld_op_desc, kParentId, end_graph->GetName() + ":" + std::to_string(node_id)),
                 "SetStr parentId failed of op:%s.", pld_op_desc->GetName().c_str());
  GE_ASSERT_TRUE(AttrUtils::SetInt(pld_op_desc, kAnchorIndex, AnchorUtils::GetIdx(out_anchor)),
                 "SetInt anchorIndex failed of op:%s.", pld_op_desc->GetName().c_str());
  GE_ASSERT_TRUE(AttrUtils::SetStr(pld_op_desc, kPeerNodeName, new_end_node->GetName()),
                 "SetStr _peerNodeName failed of op:%s.", pld_op_desc->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus ge::EnginePartitioner::SetEndOpAttr(const NodePtr &dst_node, const OpDescPtr &end_op_desc) const {
  GE_ASSERT_TRUE(AttrUtils::SetInt(end_op_desc, kPeerIndex, graph_info_.num_of_pld_end_),
                 "SetInt peerIndex failed of op:%s.", end_op_desc->GetName().c_str());
  GE_ASSERT_TRUE(AttrUtils::SetStr(end_op_desc, kParentOpType, dst_node->GetType()),
                 "SetStr parentOpType failed of op:%s", end_op_desc->GetName().c_str());
  GE_ASSERT_TRUE(end_op_desc->SetExtAttr(kParentNode, dst_node), "SetEndExtAttr parentNode failed of op:%s",
                 dst_node->GetName().c_str());
  OpDescPtr dst_node_op_desc = dst_node->GetOpDesc();
  GE_CHECK_NOTNULL(dst_node_op_desc);
  GE_ASSERT_TRUE(
      AttrUtils::SetStr(end_op_desc, ATTR_NAME_END_REAR_NODE_ENGINE_NAME, dst_node_op_desc->GetOpEngineName()),
      "SetStr rearNodeEngineName failed of op:%s", end_op_desc->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus ge::EnginePartitioner::AddPlaceHolderEndInSrcDstGraph(const AnchorPtr &out_anchor,
                                                                  const AnchorPtr &peer_in_anchor,
                                                                  const ge::ComputeGraphPtr &pld_graph,
                                                                  const ge::ComputeGraphPtr &end_graph) {
  const auto &src_node = out_anchor->GetOwnerNode();
  const auto &dst_node = peer_in_anchor->GetOwnerNode();
  // link input -> end
  NodePtr new_end_node = nullptr;
  GE_ASSERT_GRAPH_SUCCESS(MakeEndOpNode(out_anchor, end_graph, new_end_node),
                          "[Make][EndOpNode] failed, pld_graph[%s], end_graph[%s], src_node[%s], dst_node[%s]",
                          pld_graph->GetName().c_str(), end_graph->GetName().c_str(), src_node->GetName().c_str(),
                          dst_node->GetName().c_str());
  GE_ASSERT_NOTNULL(new_end_node, "[Add][Node] in graph:%s failed.", end_graph->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(SetEndOpAttr(dst_node, new_end_node->GetOpDesc()), "[Set][EndOpAttr] failed, op name[%s].",
                          new_end_node->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(new_end_node->SetOwnerComputeGraph(end_graph),
                          "[Set][OwnerComputeAttrUtilsGraph] %s for node:%s failed", end_graph->GetName().c_str(),
                          new_end_node->GetName().c_str());
  AnchorPtr end_dst_anchor = GetEndInAnchor(out_anchor, new_end_node);
  GE_ASSERT_NOTNULL(end_dst_anchor);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(out_anchor, end_dst_anchor), "[Add][Edge] from %s to %s failed",
                          out_anchor->GetOwnerNode()->GetName().c_str(),
                          end_dst_anchor->GetOwnerNode()->GetName().c_str());
  NodePtr new_pld_node = nullptr;
  GE_ASSERT_GRAPH_SUCCESS(MakePldOpNode(peer_in_anchor, src_node, pld_graph, new_pld_node),
                          "[Make][PldOpNode] failed, pld_graph[%s], end_graph[%s], src_node[%s], dst_node[%s]",
                          pld_graph->GetName().c_str(), end_graph->GetName().c_str(), src_node->GetName().c_str(),
                          dst_node->GetName().c_str());
  GE_ASSERT_NOTNULL(new_pld_node, "[Add][Node] in graph:%s failed.", pld_graph->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(SetPldOpAttr(src_node, new_end_node, end_graph, out_anchor, new_pld_node->GetOpDesc()),
                          "[Set][PldOpAttr] failed, op name[%s].", new_pld_node->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(new_pld_node->SetOwnerComputeGraph(pld_graph),
                          "[Set][OwnerComputeGraph] for node:%s failed, graph:%s", new_pld_node->GetName().c_str(),
                          pld_graph->GetName().c_str());
  AnchorPtr pld_src_anchor = GetPldOutAnchor(new_pld_node, peer_in_anchor);
  // link placeHolder -> computeNode
  GE_CHECK_NOTNULL(pld_src_anchor);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(pld_src_anchor, peer_in_anchor), "[Add][Edge] from %s to %s failed",
                          pld_src_anchor->GetOwnerNode()->GetName().c_str(),
                          peer_in_anchor->GetOwnerNode()->GetName().c_str());
  // do not care over flow
  graph_info_.num_of_pld_end_++;
  graph_info_.index_2_end_[graph_info_.num_of_pld_end_] = new_end_node;
  graph_info_.pld_2_end_[new_pld_node] = new_end_node;
  graph_info_.end_2_pld_[new_end_node] = new_pld_node;
  return SUCCESS;
}

Status ge::EnginePartitioner::LinkInput2EndRemoveOrginalLink(const ge::NodePtr &input_node,
                                                             const ge::ComputeGraphPtr &src_graph,
                                                             const ge::ComputeGraphPtr &dst_graph) {
  if ((input_node == nullptr) || (src_graph == nullptr) || (dst_graph == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param input_node or src_graph or dst_graph is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] parameter input_node or src_graph or dst_graph is nullptr.");
    return FAILED;
  }
  // get the original anchors and remove the original link
  for (const auto &out_data_anchor : input_node->GetAllOutAnchors()) {
    for (auto &peer_in_anchor : out_data_anchor->GetPeerAnchors()) {
      if (peer_in_anchor->GetOwnerNode()->GetType() != kEndType) {
        if (GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "RemoveEdge between %s and %s failed.",
                            out_data_anchor->GetOwnerNode()->GetName().c_str(),
                            peer_in_anchor->GetOwnerNode()->GetName().c_str());
          GELOGE(FAILED, "[Remove][Edge] between %s and %s failed.",
                 out_data_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetOwnerNode()->GetName().c_str());
          return FAILED;
        }
        // link input -> end
        auto ret = AddPlaceHolderEndInSrcDstGraph(out_data_anchor, peer_in_anchor, src_graph, dst_graph);
        if (ret != SUCCESS) {
          GELOGE(GE_GRAPH_ADD_PLC_END_FAILED, "[Call][AddPlaceHolderEndInSrcDstGraph] failed, ret:%d.", ret);
          return ret;
        }
      } else {
        auto end_node = peer_in_anchor->GetOwnerNode();
        if (GraphUtils::RemoveJustNode(src_graph, end_node) != GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "RemoveJustNode %s from graph:%s failed.",
                            end_node->GetName().c_str(), src_graph->GetName().c_str());
          GELOGE(FAILED, "[Remove][JustNode] %s from graph:%s failed.",
                 end_node->GetName().c_str(), src_graph->GetName().c_str());
          return FAILED;
        }
        if (end_node->SetOwnerComputeGraph(dst_graph) != GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "SetOwnerComputeGraph for node:%s failed, graph:%s.",
                            end_node->GetName().c_str(), dst_graph->GetName().c_str());
          GELOGE(FAILED, "[Set][OwnerComputeGraph] to node:%s failed, graph:%s.",
                 end_node->GetName().c_str(), dst_graph->GetName().c_str());
          return FAILED;
        }
        if (dst_graph->AddNode(end_node) == nullptr) {
          REPORT_INNER_ERR_MSG("E19999", "AddNode %s in graph:%s failed.",
                            end_node->GetName().c_str(), dst_graph->GetName().c_str());
          GELOGE(FAILED, "[Add][Node] %s in graph:%s failed.",
                 end_node->GetName().c_str(), dst_graph->GetName().c_str());
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::PutInputNodesInSubGraph(const ge::ComputeGraphPtr &src_graph,
                                                      const ge::ComputeGraphPtr &dst_graph) {
  GE_CHECK_NOTNULL(src_graph);
  GE_CHECK_NOTNULL(dst_graph);
  for (const auto &input_node : src_graph->GetDirectNode()) {
    if (IsDataLike(input_node)) {
      if (input_node->SetOwnerComputeGraph(dst_graph) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "SetOwnerComputeGraph for node:%s failed, graph:%s.",
                          input_node->GetName().c_str(), dst_graph->GetName().c_str());
        GELOGE(FAILED, "[Set][OwnerComputeGraph] for node:%s failed, graph:%s.",
               input_node->GetName().c_str(), dst_graph->GetName().c_str());
        return FAILED;
      }
      // remove input node from src_graph
      if (GraphUtils::RemoveJustNode(src_graph, input_node) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "RemoveJustNode %s from graph:%s failed.",
                          input_node->GetName().c_str(), src_graph->GetName().c_str());
        GELOGE(FAILED, "[Remove][JustNode] %s from graph:%s failed.",
               input_node->GetName().c_str(), src_graph->GetName().c_str());
        return FAILED;
      }
      // add input node to dst_graph
      if (dst_graph->AddNode(input_node) == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "AddNode %s in graph:%s failed.",
                          input_node->GetName().c_str(), src_graph->GetName().c_str());
        GELOGE(FAILED, "[Add][Node] %s in graph:%s failed.",
               input_node->GetName().c_str(), src_graph->GetName().c_str());
        return FAILED;
      }
      if (LinkInput2EndRemoveOrginalLink(input_node, src_graph, dst_graph) != ge::SUCCESS) {
        GELOGE(FAILED, "[Call][LinkInput2EndRemoveOrginalLink] failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

void ge::EnginePartitioner::AddNewGraphToPartition(const ge::ComputeGraphPtr &input_graph,
                                                   const std::string &engine_name) {
  if (input_graph == nullptr) {
    GELOGW("[EnginePartitioner]: input_graph is null, engine name is %s", engine_name.c_str());
    return;
  }
  graph_info_.partitions_[input_graph] = engine_name;
}

bool ge::EnginePartitioner::IsDataLike(ge::NodePtr node) const {
  const auto &node_type = node->GetType();
  return (node_type == CONSTANT) || OpTypeUtils::IsDataNode(node_type) || (node_type == CONSTANTOP) ||
         OpTypeUtils::IsVarLikeNode(node_type);
}

bool ge::EnginePartitioner::HasNoInput(const ge::NodePtr &node) const {
  if (node == nullptr) {
    GELOGE(FAILED, "[Check][Param] node is nullptr.");
    return true;
  }
  return node->GetInNodesSize() == 0UL;
}

Status ge::EnginePartitioner::InitializeInputClusters(const NodePtr &node, const ClusterPtr &cluster, size_t index) {
  auto node_id = node->GetOpDesc()->GetId();
  for (const auto &parent : node->GetInAllNodes()) {
    GE_CHECK_NOTNULL(parent->GetOpDesc());
    auto parent_id = parent->GetOpDesc()->GetId();
    if (parent_id < node_id) {
      const auto iter = graph_info_.node_2_cluster_.find(parent);
      GE_CHK_BOOL_RET_STATUS(iter != graph_info_.node_2_cluster_.cend(), FAILED,
                             "[Check][Param] node[%s]id[%ld]'s parent_node[%s]id[%ld] should make cluster in advance",
                             node->GetOpDesc()->GetName().c_str(), node_id, parent->GetOpDesc()->GetName().c_str(),
                             parent_id);
      cluster->in_clu_.insert(iter->second->index_);
      iter->second->out_clu_.insert(index);
    }
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::Initialize(const ge::ComputeGraphPtr &compute_graph, Mode mode) {
  GELOGI("Initialize starts, Engine partition mode: %d.", static_cast<int32_t>(graph_info_.mode_));
  const auto &node_engine_map = GetNodeEngineMap();
  size_t temp_index = 0;
  std::map<NodePtr, OpInfo> nodes_to_op_infos;
  MarkCvParallelAivNodes(compute_graph);
  for (const auto &node : compute_graph->GetDirectNode()) {
    std::string temp_stream;
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    // node opdesc has been checked before
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, temp_stream);
    std::string temp_user_stream;
    (void)AttrUtils::GetStr(op_desc, public_attr::USER_STREAM_LABEL, temp_user_stream);
    ClusterPtr new_cluster;
    // data like node without input should be handle specific
    if (HasNoInput(node) && IsDataLike(node)) {
      // data类节点不应该打上用户流标签，后续逻辑流分配会报错
      if (!temp_user_stream.empty()) {
        (void)AttrUtils::SetStr(op_desc, public_attr::USER_STREAM_LABEL, "");
        temp_user_stream.clear();
      }

      ClusterPtr cluster = MakeShared<Cluster>(temp_index, kEngineDefaultData, temp_stream, temp_user_stream);
      new_cluster = cluster;
    } else {
      if (node_engine_map.count(node) == 0) {
        bool is_check_support_success = false;
        std::set<std::string> exclude_engines;
        DNNEngineManager::GetExcludeEngines(exclude_engines);
        OpInfo op_info;
        GE_CHK_STATUS_RET(engine_placer_.SelectEngine(node, exclude_engines, is_check_support_success, op_info),
                          "[Check][Param] node[%s] does not owner engine!", node->GetName().c_str());
        nodes_to_op_infos.emplace(node, op_info);
      }
      GE_ASSERT_TRUE(node_engine_map.count(node) > 0, "Failed to find node:%s(%s) in node engine map, mode:%d",
                     node->GetName().c_str(), node->GetType().c_str(), static_cast<int32_t>(graph_info_.mode_));
      std::string engine_name = GenClusterEngineName(node, mode, node_engine_map);
      ClusterPtr cluster = MakeShared<Cluster>(temp_index, engine_name, temp_stream, temp_user_stream);
      new_cluster = cluster;
    }
    GE_ASSERT_NOTNULL(new_cluster, "[Allocate][Cluster] failed, index:%zu", temp_index);
    new_cluster->nodes_.push_back(node);
    if (AttrUtils::HasAttr(op_desc, kEnableCvParallel)) {
      new_cluster->engine_name_ = kVectorEngineName;
    }

    if (!HasNoInput(node)) {
      GE_CHK_STATUS_RET(InitializeInputClusters(node, new_cluster, temp_index),
                        "Failed to init input clusters of cluster:%zu", temp_index);
    }
    graph_info_.node_2_cluster_[node] = new_cluster;
    graph_info_.clusters_[temp_index] = new_cluster;
    GELOGD("Node name is %s, engine is %s, cluster index is %zu, stream label is %s", node->GetName().c_str(),
           new_cluster->engine_name_.c_str(), new_cluster->index_, new_cluster->stream_label_.c_str());
    temp_index++;
  }
  DNNEngineManager::UpdateOpDescsWithOpInfos(nodes_to_op_infos);
  GELOGD("Initialize ends.");
  return SUCCESS;
}

Status ge::EnginePartitioner::AddPartitionsToGraphNode(std::vector<ge::SubGraphInfoPtr> &output_subgraphs,
                                                       ge::ComputeGraphPtr compute_graph) {
  const std::string &input_subgraph_name = "inputNodesSubGraph";
  std::string session_graph_id;
  if (!AttrUtils::GetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Get graph session_graph_id attr failed.");
    return INTERNAL_ERROR;
  }
  // the output_subgraphs have topological order
  for (const auto &sub_graph : graph_info_.rank_2_partitions_) {
    if (graph_info_.partitions_.find(sub_graph) == graph_info_.partitions_.end()) {
      REPORT_INNER_ERR_MSG("E19999", "partition is null, subgraph:%s", sub_graph->GetName().c_str());
      GELOGE(GE_GRAPH_EMPTY_PARTITION, "[Check][Param] partition is null, subgraph:%s", sub_graph->GetName().c_str());
      return FAILED;
    }
    auto &engine_name = graph_info_.partitions_.at(sub_graph);
    (void)AttrUtils::SetStr(sub_graph, ATTR_NAME_PARENT_GRAPH_NAME, compute_graph->GetName());
    (void)sub_graph->SetExtAttr("part_src_graph", compute_graph);
    GELOGD("set attr success. subgraph(%s) with parent graph(%s)", sub_graph->GetName().c_str(),
           compute_graph->GetName().c_str());
    GE_DUMP(sub_graph, sub_graph->GetName()  + "_" + mode_2_str_[graph_info_.mode_]);
    if (!session_graph_id.empty()) {
      GE_ASSERT_TRUE(AttrUtils::SetStr(sub_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                     "SetStr ATTR_NAME_SESSION_GRAPH_ID[%s] failed of subgraph:%s", session_graph_id.c_str(),
                     sub_graph->GetName().c_str());
    }
    // flush parent node of subgraph
    if (compute_graph->GetParentNode() == nullptr) {
      GE_ASSERT_TRUE(AttrUtils::SetBool(sub_graph, ATTR_NAME_IS_ROOT_GRAPH, true),
                     "Set attr ATTR_NAME_IS_ROOT_GRAPH[%s] failed of subgraph:%s", session_graph_id.c_str(),
                     sub_graph->GetName().c_str());
    } else {
      sub_graph->SetParentNode(compute_graph->GetParentNode());
    }

    sub_graph->SetGraphUnknownFlag(compute_graph->GetGraphUnknownFlag());
    auto sgi = MakeShared<SubGraphInfo>();
    if (SetMemberForSubGraphInfo(sgi, sub_graph, engine_name) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "set members for subgraph:%s info failed", sub_graph->GetName().c_str());
      GELOGE(FAILED, "set members for subgraph:%s info failed", sub_graph->GetName().c_str());
      return FAILED;
    }

    AddEndPldInformationToSubGraphInfo(sgi);
    GELOGD("[EnginePartitioner]: subGraph engine name is %s, graph name is %s, stream label[%s], user stream label[%s]",
           engine_name.c_str(), sub_graph->GetName().c_str(),
           sgi->GetStreamLabel().empty() ? "null" : sgi->GetStreamLabel().c_str(),
           sgi->GetUserStreamLabel().empty() ? "null" : sgi->GetUserStreamLabel().c_str());
    if (engine_name != input_subgraph_name) {  // do not add Data subGraph into SubGraphInfo
      output_subgraphs.push_back(sgi);
    } else {
      graph_2_input_subgraph_[compute_graph] = sgi;
    }
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::SetMemberForSubGraphInfo(ge::SubGraphInfoPtr &sgi, const ComputeGraphPtr &sub_graph,
                                                       const std::string &engine_name) {
  // The caller gurantee the sub_graph parm is not null
  if (sgi == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "allocate memory for SubGraphInfo failed.");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Allocate][Memory] for SubGraphInfo failed.");
    return FAILED;
  }
  // set engine name
  sgi->SetEngineName(engine_name);
  // set stream label
  std::string sub_graph_stream;
  GE_ASSERT_TRUE(sub_graph->GetDirectNodesSize() != 0, "[Check][Param]graph:%s has no node",
                 sub_graph->GetName().c_str());
  if (AttrUtils::GetStr(sub_graph->GetDirectNodePtr().at(0)->GetOpDesc(), ATTR_NAME_STREAM_LABEL, sub_graph_stream)) {
    sgi->SetStreamLabel(sub_graph_stream);
  }
  std::string sub_graph_user_stream;
  if (AttrUtils::GetStr(sub_graph->GetDirectNodePtr().at(0)->GetOpDesc(), public_attr::USER_STREAM_LABEL,
                        sub_graph_user_stream)) {
    sgi->SetUserStreamLabel(sub_graph_user_stream);
  }
  /// for now inputFlag is the same before and after partition. It should
  /// be changed according to the real partition
  std::vector<bool> sub_graph_input(graph_info_.input_size_, true);
  std::vector<bool> sub_graph_output(graph_info_.output_size_, true);
  sgi->SetSubGraph(sub_graph);
  sgi->SetOutputFlag(sub_graph_output);
  sgi->SetInputFlag(sub_graph_input);
  sgi->SetOutputContext(graph_info_.output_name_);
  return SUCCESS;
}

// check if two clusters can merge
bool ge::EnginePartitioner::IsMergeable(size_t parent_cluster, size_t child_cluster, size_t upper_bound) {
  if ((graph_info_.clusters_[parent_cluster] == nullptr) || (graph_info_.clusters_[parent_cluster]->nodes_.empty()) ||
      (graph_info_.clusters_[child_cluster] == nullptr) || (graph_info_.clusters_[child_cluster]->nodes_.empty())) {
    return false;
  }
  if ((current_mode_ == Mode::kSecondPartitioning) &&
      ((!graph_info_.clusters_[parent_cluster]->user_stream_label_.empty()) &&
      (graph_info_.clusters_[parent_cluster]->user_stream_label_ ==
      graph_info_.clusters_[child_cluster]->user_stream_label_))) {
    GELOGD(
        "Parent cluster[%zu] engine[%s] stream label[%s] user stream label[%s], child cluster[%zu] engine[%s] stream "
        "label[%s] user stream label[%s] should merge",
        parent_cluster, graph_info_.clusters_[parent_cluster]->engine_name_.c_str(),
        graph_info_.clusters_[parent_cluster]->stream_label_.c_str(),
        graph_info_.clusters_[parent_cluster]->user_stream_label_.c_str(), child_cluster,
        graph_info_.clusters_[child_cluster]->engine_name_.c_str(),
        graph_info_.clusters_[child_cluster]->stream_label_.c_str(),
        graph_info_.clusters_[child_cluster]->user_stream_label_.c_str());
    return true;
  }
  // Check if parent_cluster,child_cluster has same engine or stream label
  if ((graph_info_.clusters_[parent_cluster]->engine_name_ != graph_info_.clusters_[child_cluster]->engine_name_) ||
      (graph_info_.clusters_[parent_cluster]->stream_label_ != graph_info_.clusters_[child_cluster]->stream_label_)) {
    GELOGD(
        "Parent cluster[%zu] engine[%s] stream label[%s] user stream label[%s], child cluster[%zu] engine[%s] stream "
        "label[%s] user stream label[%s] can not merge",
        parent_cluster, graph_info_.clusters_[parent_cluster]->engine_name_.c_str(),
        graph_info_.clusters_[parent_cluster]->stream_label_.c_str(),
        graph_info_.clusters_[parent_cluster]->user_stream_label_.c_str(), child_cluster,
        graph_info_.clusters_[child_cluster]->engine_name_.c_str(),
        graph_info_.clusters_[child_cluster]->stream_label_.c_str(),
        graph_info_.clusters_[child_cluster]->user_stream_label_.c_str());
    return false;
  }
  // Check if parent_cluster,child_cluster is reachable
  RemoveEdge(parent_cluster, child_cluster);
  // Check if there is a path between parent and child, if return true, can not merge
  if (HasSecondPath(parent_cluster, child_cluster, upper_bound)) {
    GELOGD("Find second path from %zu to %zu, upper bound is %zu", parent_cluster, child_cluster, upper_bound);
    InsertEdge(parent_cluster, child_cluster);
    return false;
  }
  InsertEdge(parent_cluster, child_cluster);
  return true;
}

void ge::EnginePartitioner::MergeTwoClusters(size_t parent_cluster, size_t &child_cluster) {
  // check which index is bigger
  size_t big_cluster, small_cluster;
  size_t child_cluster_original = child_cluster;
  if (parent_cluster > child_cluster) {
    small_cluster = child_cluster;
    big_cluster = parent_cluster;
  } else {
    big_cluster = child_cluster;
    small_cluster = parent_cluster;
    // flush child_cluster, because it has been modified
    child_cluster = small_cluster;
  }

  // update node_2_cluster_ map
  for (const auto &node : graph_info_.clusters_[big_cluster]->nodes_) {
    graph_info_.node_2_cluster_[node] = graph_info_.clusters_[small_cluster];
  }
  // merge nodes
  graph_info_.clusters_[small_cluster]->nodes_.splice(graph_info_.clusters_[small_cluster]->nodes_.cend(),
                                                      graph_info_.clusters_[big_cluster]->nodes_);
  // remove child_cluster's out parent_cluster's in between child_cluster and parent_cluster
  // this should be called before `merge all input & output to small cluster`
  RemoveEdge(parent_cluster, child_cluster_original);
  // merge all input & output to small cluster
  graph_info_.clusters_[small_cluster]->in_clu_.insert(graph_info_.clusters_[big_cluster]->in_clu_.cbegin(),
                                                       graph_info_.clusters_[big_cluster]->in_clu_.cend());
  graph_info_.clusters_[small_cluster]->out_clu_.insert(graph_info_.clusters_[big_cluster]->out_clu_.cbegin(),
                                                        graph_info_.clusters_[big_cluster]->out_clu_.cend());
  // update in/out of the cluster with bigger index
  for (auto in_clu : graph_info_.clusters_[big_cluster]->in_clu_) {
    GE_CHECK_NOTNULL_JUST_RETURN(graph_info_.clusters_[in_clu]);
    graph_info_.clusters_[in_clu]->out_clu_.insert(small_cluster);
    graph_info_.clusters_[in_clu]->out_clu_.erase(big_cluster);
  }
  for (auto out_clu : graph_info_.clusters_[big_cluster]->out_clu_) {
    GE_CHECK_NOTNULL_JUST_RETURN(graph_info_.clusters_[out_clu]);
    graph_info_.clusters_[out_clu]->in_clu_.insert(small_cluster);
    graph_info_.clusters_[out_clu]->in_clu_.erase(big_cluster);
  }
  graph_info_.clusters_[big_cluster] = graph_info_.clusters_[small_cluster];
}

void ge::EnginePartitioner::RemoveEdge(size_t parent_cluster, size_t child_cluster) {
  graph_info_.clusters_[child_cluster]->in_clu_.erase(parent_cluster);
  graph_info_.clusters_[parent_cluster]->out_clu_.erase(child_cluster);
}

void ge::EnginePartitioner::InsertEdge(size_t from, size_t to) {
  if (from == to) {
    return;
  }
  if (!graph_info_.clusters_[from]->out_clu_.insert(to).second) {
    // edge has already exists
    return;
  }
  graph_info_.clusters_[to]->in_clu_.insert(from);
}

Status ge::EnginePartitioner::MarkClustersWithConsistantId() {
  GELOGI("MarkClustersWithConsistantId starts. cluster size is %zu",
      graph_info_.clusters_.size());
  size_t cluster_size = graph_info_.clusters_.size();
  std::vector<size_t> cluster_id;
  for (size_t i = 0UL; i < cluster_size; i++) {
    auto cluster = graph_info_.clusters_[i];
    GE_ASSERT_NOTNULL(cluster);
    if (cluster->engine_name_ != kEngineDefaultData) {
      cluster_id.emplace_back(i);
    }
  }
  if (cluster_id.empty()) {
    return SUCCESS;
  }
  for (size_t i = cluster_id.size() - 1UL; i > 0UL; i--) {
    auto cur_cluster_id = cluster_id[i];
    auto next_cluster_id = cluster_id[i - 1UL];
    auto merged_id = cur_cluster_id;
    if (IsMergeable(next_cluster_id, merged_id, merged_id)) {
      MergeTwoClusters(next_cluster_id, merged_id);
      GELOGD("Merging cluster %zu and %zu to %zu", cur_cluster_id, next_cluster_id, merged_id);
    }
  }
  GELOGD("MarkClustersWithConsistantId ends.");
  return SUCCESS;
}

void ge::EnginePartitioner::MarkClusters() {
  GELOGI("MarkClusters starts. cluster size is %zu", graph_info_.clusters_.size());
  size_t cluster_size = graph_info_.clusters_.size();
  for (size_t child_cluster = 0; child_cluster < cluster_size; child_cluster++) {
    auto found_child_cluster = graph_info_.clusters_[child_cluster];
    if (found_child_cluster == nullptr) {
      GELOGW("can not found child_cluster is %zu", child_cluster);
      continue;
    }
    auto copy_parents_clusters = found_child_cluster->in_clu_;
    std::vector<size_t> ordered_cluster;
    for (const auto &parent_cluster : copy_parents_clusters) {
      ordered_cluster.emplace_back(parent_cluster);
    }
    // sort cluster according to it's output amount
    auto comp_func = [this](const size_t &parent_cluster1, const size_t &parent_cluster2) -> bool {
      return graph_info_.clusters_[parent_cluster1]->out_clu_.size() <
          graph_info_.clusters_[parent_cluster2]->out_clu_.size();
    };
    std::sort(ordered_cluster.begin(), ordered_cluster.end(), comp_func);
    auto child_merged = child_cluster;
    for (const auto &parent_cluster : ordered_cluster) {
      if (IsMergeable(parent_cluster, child_merged, child_cluster)) {
        MergeTwoClusters(parent_cluster, child_merged);
        GELOGD("Merging cluster %zu and %zu to %zu", parent_cluster, child_cluster, child_merged);
      }
    }
  }
  GELOGD("MarkClusters ends.");
}

Status ge::EnginePartitioner::SplitNodeInputs(const NodePtr &node,
                                              const NodePtr &corresponding_node,
                                              const ClusterPtr &child_cluster) {
  for (const auto &in_anchor : node->GetAllInAnchors()) {
    GELOGD("In anchor index is %d", AnchorUtils::GetIdx(in_anchor));
    for (const auto &peer_out_anchor : in_anchor->GetPeerAnchors()) {
      GE_CHECK_NOTNULL(peer_out_anchor->GetOwnerNode()->GetOpDesc());
      GELOGD("Peer out anchor index is %d", AnchorUtils::GetIdx(peer_out_anchor));
      // Normally, all nodes have a copy in corresponding_node_in_partitions_, so function at can not be exception
      const auto iter = graph_info_.corresponding_node_in_partitions_.find(peer_out_anchor->GetOwnerNode()->GetName());
      GE_CHK_BOOL_RET_STATUS(iter != graph_info_.corresponding_node_in_partitions_.cend(), FAILED,
                             "[Check][Param] node[%s]id[%ld]'s parent_node[%s]id[%ld]"
                             "should make corresponding in advance",
                             node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetId(),
                             peer_out_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
                             peer_out_anchor->GetOwnerNode()->GetOpDesc()->GetId());
      const auto &parent_node = iter->second;
      GE_CHECK_NOTNULL(parent_node);
      GELOGD("Parent node name is %s", parent_node->GetName().c_str());
      // add edge
      const auto &src_anchor = parent_node->GetOutAnchor(AnchorUtils::GetIdx(peer_out_anchor));
      const auto &dst_anchor = corresponding_node->GetInAnchor(AnchorUtils::GetIdx(in_anchor));
      // if child and parent's cluster is not same, add plc and end
      const auto &parent_cluster = graph_info_.node_2_cluster_[peer_out_anchor->GetOwnerNode()];
      GE_CHECK_NOTNULL(parent_cluster);
      if (parent_cluster != child_cluster) {
        GELOGD("Parent cluster is %zu, child_cluster is %zu", parent_cluster->index_, child_cluster->index_);
        GE_CHK_STATUS_RET(AddPlaceHolderEnd(peer_out_anchor, in_anchor),
                          "[AddPlaceHolderEnd] failed, out_anchor:%s index:%d, in_anchor:%s index:%d.",
                          peer_out_anchor->GetOwnerNode()->GetName().c_str(), AnchorUtils::GetIdx(peer_out_anchor),
                          in_anchor->GetOwnerNode()->GetName().c_str(), AnchorUtils::GetIdx(in_anchor));
      } else {  // parent and child in the same cluster, add edge
        GELOGD("AddEdge from parent cluster %zu to child %zu", parent_cluster->index_, child_cluster->index_);
        GE_CHK_STATUS_RET(GraphUtils::AddEdge(src_anchor, dst_anchor), "Add edge from %s to %s failed",
                          peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                          in_anchor->GetOwnerNode()->GetName().c_str());
      }
    }
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::SplitSubGraphs(const ge::ComputeGraphPtr &compute_graph) {
  GELOGD("SplitSubGraphs starts.");
  // Create graphs for all clusters
  std::unordered_set<ClusterPtr> cluster_set;
  // add pld&end
  for (const auto &node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GELOGD("Node name is %s.", node->GetName().c_str());
    const auto &child_cluster = graph_info_.node_2_cluster_[node];
    ge::ComputeGraphPtr corresponding_graph;
    // unordered_set's insert returns a pair, second of pair is bool
    if (!cluster_set.insert(child_cluster).second) {
      GELOGD("Old sub graph, child_cluster is %zu", child_cluster->index_);
      corresponding_graph = graph_info_.cluster_2_partition_.at(child_cluster);
    } else {
      std::string graph_name = "new_sub_graph" + std::to_string(graph_info_.partitions_.size());
      ComputeGraphPtr new_sub_graph = MakeShared<ge::ComputeGraph>(graph_name);
      GE_ASSERT_NOTNULL(new_sub_graph, "[Allocate][Memory] for ge::ComputeGraph failed.");
      AddNewGraphToPartition(new_sub_graph, child_cluster->engine_name_);
      corresponding_graph = new_sub_graph;
      graph_info_.cluster_2_partition_[child_cluster] = corresponding_graph;
      GELOGD("New sub graph, name is %s", graph_name.c_str());
    }
    // build node to corresponding node map
    NodePtr corresponding_node = corresponding_graph->AddNode(node->GetOpDesc());
    GE_CHECK_NOTNULL(corresponding_node);
    if (!graph_info_.corresponding_node_in_partitions_.insert({node->GetName(), corresponding_node}).second) {
      REPORT_INNER_ERR_MSG("E19999", "Node name %s already existed in graph %s", node->GetName().c_str(),
                        compute_graph->GetName().c_str());
      GELOGE(FAILED, "Node name %s already existed in graph %s", node->GetName().c_str(),
             compute_graph->GetName().c_str());
      return FAILED;
    }
    GE_CHK_STATUS_RET(corresponding_node->SetOwnerComputeGraph(corresponding_graph));
    GE_CHK_STATUS_RET(SplitNodeInputs(node, corresponding_node, child_cluster),
                      "[Split][NodeInputs] failed, node[%s], child_cluster[%zu]", node->GetName().c_str(),
                      child_cluster->index_);
  }
  GELOGD("SplitSubGraphs ends.");
  return SUCCESS;
}

/// before calling this function, the direct path between src and dst are already removed.
/// return true if a second path is found
bool ge::EnginePartitioner::HasSecondPath(size_t src, size_t dst, size_t upper_bound) {
  bool has_second = false;
  if (graph_info_.clusters_.at(src)->out_clu_.empty() || graph_info_.clusters_.at(dst)->in_clu_.empty()) {
    return has_second;
  }
  /// Avoid recursion since stack space might be limited.
  /// We instead keep a stack of nodes to visit.
  std::vector<size_t> temp_stack;
  std::vector<size_t> second_path_ids;
  std::set<size_t> visited;
  temp_stack.push_back(src);
  while (!temp_stack.empty()) {
    if (has_second) {
      break;
    }
    size_t cluster = temp_stack.back();
    second_path_ids.emplace_back(cluster);
    temp_stack.pop_back();
    ClusterPtr cur_cluster = graph_info_.clusters_[cluster];
    if (!visited.insert(cluster).second) {
      continue;
    }
    for (auto out : cur_cluster->out_clu_) {
      if (out == dst) {
        has_second = true;  // There is cycle
        second_path_ids.emplace_back(out);
        break;
      }
      if (out < upper_bound) {
        temp_stack.push_back(out);
      }
    }
  }
  if (has_second) {
    std::stringstream path;
    std::for_each(second_path_ids.begin(), second_path_ids.end(), [&path](const size_t &id) { path << id << "->"; });
    GELOGD("Second path is [%s]", path.str().c_str());
  }
  return has_second;
}

Status ge::EnginePartitioner::Partition(const ge::ComputeGraphPtr &compute_graph, Mode mode) {
  current_mode_ = mode;
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHK_STATUS_RET(compute_graph->TopologicalSorting(), "TopologicalSorting for graph:%s failed",
                    compute_graph->GetName().c_str());
  if (mode == EnginePartitioner::Mode::kCompositeEnginePartitioning) {
    GE_CHK_STATUS_RET(engine_placer_.AssignCompositeEngine(),
                      "[Partition][SubGraph] Assign composite engine for graph %s failed",
                      compute_graph->GetName().c_str());
  }
  ge::GetContext().GetOption(ge::OPTION_TOPOSORTING_MODE, topo_sorting_mode_);
  ClearAllPartitionData();
  GELOGD("%s start part with mode %d", compute_graph->GetName().c_str(), mode);
  auto real_ret = SUCCESS;
  auto ret = PartitionSubGraph(compute_graph, mode);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Partition][SubGraph] Failed, ret:%d", ret);
    real_ret = ret;
  }
  GE_CHECK_NOTNULL(compute_graph);
  // partition sub graph
  for (const auto &sub_graph : compute_graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(sub_graph);
    GELOGD("%s start part for its subgraph %s with mode %d", compute_graph->GetName().c_str(),
           sub_graph->GetName().c_str(), mode);
    bool no_need_partition_and_merge = false;
    bool no_need_partition = false;
    (void)ge::AttrUtils::GetBool(sub_graph, ATTR_NAME_NO_NEED_PARTITION, no_need_partition);
    (void)ge::AttrUtils::GetBool(sub_graph, ATTR_NAME_NO_NEED_PARTITION_AND_MERGE, no_need_partition_and_merge);
    if (no_need_partition_and_merge || no_need_partition) {
      GELOGI("sub graph %s no need partition, skip it", sub_graph->GetName().c_str());
      continue;
    }
    ret = PartitionSubGraph(sub_graph, mode);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Partition][SubGraph] Failed, ret:%d", ret);
      real_ret = ret;
    }
  }
  if (real_ret != SUCCESS) {
    auto root_graph = ge::GraphUtils::FindRootGraph(compute_graph);
    GE_CHECK_NOTNULL(root_graph);
    (void)Analyzer::GetInstance()->SaveAnalyzerDataToFile(root_graph->GetSessionID(),
                                                          root_graph->GetGraphID());
  }
  return real_ret;
}

Status ge::EnginePartitioner::PartitionSubGraph(const ge::ComputeGraphPtr &compute_graph, Mode mode) {
  GE_CHECK_NOTNULL(compute_graph);
  // clear graph_info
  GraphPartitionInfo graph_info(mode);
  graph_info_ = std::move(graph_info);
  graph_info_.output_name_ = compute_graph->GetOutput();
  graph_info_.output_size_ = compute_graph->GetOutputSize();
  graph_info_.input_size_ = compute_graph->GetInputSize();
  GELOGI("Graph Partition starts, graph nodes size is %zu", compute_graph->GetDirectNodesSize());
  GE_TRACE_START(PartitionSubGraphInitialize);
  GE_CHK_STATUS_RET(Initialize(compute_graph, mode), "[Initialize] for graph:%s failed", compute_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(PartitionSubGraphInitialize, "EnginePartitioner::PartitionInitialize");
  GE_TRACE_START(PartitionSubGraphMarkClusters);
  if (topo_sorting_mode_ == kStableRdfsSort) {
    GE_ASSERT_SUCCESS(MarkClustersWithConsistantId());
  } else {
    MarkClusters();
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(PartitionSubGraphMarkClusters, "EnginePartitioner::PartitionMarkClusters");
  GE_TRACE_START(PartitionSubGraphSplitSubGraphs);
  if (SplitSubGraphs(compute_graph) != SUCCESS) {
    GELOGE(FAILED, "[Split][SubGraphs] for graph:%s failed", compute_graph->GetName().c_str());
    return FAILED;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(PartitionSubGraphSplitSubGraphs, "EnginePartitioner::PartitionSplitSubGraphs");
  GE_TRACE_START(PartitionSubGraphSortSubGraphs);
  if (SortSubGraphs(compute_graph) != ge::SUCCESS) {
    GELOGE(GE_GRAPH_TOPO_SORT_FAILED, "[Sort][SubGraphs] for graph:%s failed.",
           compute_graph->GetName().c_str());
    return ge::FAILED;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(PartitionSubGraphSortSubGraphs, "EnginePartitioner::PartitionSortSubGraphs");
  GE_TRACE_START(PartitionSubGraphAddPartitionsToGraphNode);
  std::vector<ge::SubGraphInfoPtr> output_subgraphs;
  if (AddPartitionsToGraphNode(output_subgraphs, compute_graph) != ge::SUCCESS) {
    GELOGE(GE_GRAPH_EMPTY_PARTITION, "[Add][Partitions] To GraphNode failed, graph:%s.",
           compute_graph->GetName().c_str());
    return ge::FAILED;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(PartitionSubGraphAddPartitionsToGraphNode,
                         "EnginePartitioner::PartitionAddPartitionsToGraphNode");
  GELOGI("Graph Partition ends. Adding partitions to SubGraphInfo, got %zu sub graphs", output_subgraphs.size());
  partition_times_++;  // do not care over flow
  graph_2_graph_partition_info_[compute_graph] = std::move(graph_info_);
  graph_2_graph_partition_info_[compute_graph].mode_ = Mode::kMerging;
  graph_2_subgraph_list_[compute_graph] = std::move(output_subgraphs);
  return SUCCESS;
}

// all the inputs are the nodes and anchors in the original graph
Status ge::EnginePartitioner::AddPlaceHolderEnd(const AnchorPtr &out_anchor, const AnchorPtr &in_anchor) {
  GE_CHECK_NOTNULL(out_anchor);
  GE_CHECK_NOTNULL(in_anchor);
  // nodes in original graph
  const auto &src_node = out_anchor->GetOwnerNode();
  const auto &dst_node = in_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(src_node);
  GE_CHECK_NOTNULL(dst_node);
  // All nodes have a copy in corresponding_node_in_partitions_, so function at can not be execption
  const auto &node_in_partitions = graph_info_.corresponding_node_in_partitions_;
  const auto &src_anchor = node_in_partitions.at(src_node->GetName())->GetOutAnchor(AnchorUtils::GetIdx(out_anchor));
  const auto &dst_anchor = node_in_partitions.at(dst_node->GetName())->GetInAnchor(AnchorUtils::GetIdx(in_anchor));
  GE_CHECK_NOTNULL(src_anchor, "src_anchor(index:%d) is nullptr", AnchorUtils::GetIdx(out_anchor));
  GE_CHECK_NOTNULL(dst_anchor, "dst_anchor(index:%d) is nullptr", AnchorUtils::GetIdx(in_anchor));
  // anchors in subGraph
  const ComputeGraphPtr &src_subgraph = src_anchor->GetOwnerNode()->GetOwnerComputeGraph();
  const ComputeGraphPtr &dst_subgraph = dst_anchor->GetOwnerNode()->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(src_subgraph);
  GE_CHECK_NOTNULL(dst_subgraph);
  // add end and pld node
  auto ret = AddPlaceHolderEndInSrcDstGraph(src_anchor, dst_anchor, dst_subgraph, src_subgraph);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_ADD_PLC_END_FAILED, "[Call][AddPlaceHolderEndInSrcDstGraph] failed, ret:%d.", ret);
    return ret;
  }
  return SUCCESS;
}

Status ge::EnginePartitioner::SortSubGraphs(const ge::ComputeGraphPtr &compute_graph) {
  uint32_t rank = kRankOne;  // rank 0 for data graph
  ComputeGraphPtr new_input_nodes_sub_graph = MakeShared<ComputeGraph>("inputNodeGraph");
  GE_CHECK_NOTNULL(new_input_nodes_sub_graph);
  GE_CHECK_NOTNULL(compute_graph);
  for (const auto &node : compute_graph->GetDirectNode()) {
    // All nodes in original graph have a copy in corresponding_node_in_partitions_, so it can not be null
    auto sub_graph = graph_info_.corresponding_node_in_partitions_.at(node->GetName())->GetOwnerComputeGraph();
    if ((graph_info_.partitions_2_rank_.find(sub_graph) == graph_info_.partitions_2_rank_.end()) &&
        (graph_info_.partitions_[sub_graph] != kEngineDefaultData)) {
      graph_info_.partitions_2_rank_[sub_graph] = rank;
      graph_info_.rank_2_partitions_.push_back(sub_graph);
      rank++;
    } else if (graph_info_.partitions_[sub_graph] == kEngineDefaultData) {  // merge data graph
      if (PutInputNodesInSubGraph(sub_graph, new_input_nodes_sub_graph) != SUCCESS) {
        GELOGE(FAILED, "[Call][putInputNodesInSubGraph] failed.");
        return FAILED;
      }
      graph_info_.partitions_.erase(graph_info_.partitions_.find(sub_graph));
    }
  }
  if (!new_input_nodes_sub_graph->GetDirectNode().empty()) {
    graph_info_.rank_2_partitions_.insert(graph_info_.rank_2_partitions_.cbegin(), new_input_nodes_sub_graph);
    graph_info_.partitions_2_rank_[new_input_nodes_sub_graph] = 0;
    AddNewGraphToPartition(new_input_nodes_sub_graph, "inputNodesSubGraph");
  }
  // reinit rank
  rank = kRankZero;
  for (const auto &it : graph_info_.rank_2_partitions_) {
    // rename subGraph based on rank
    if (it != nullptr) {
      // rename subGraph based on rank
      std::string graph_name =
        "partition" + std::to_string(partition_times_) + "_rank" + std::to_string(rank) + "_" + it->GetName();
      it->SetName(graph_name);
    }
    rank++;
  }
  return SUCCESS;
}

AnchorPtr ge::EnginePartitioner::GetEndInAnchor(const AnchorPtr &src_anchor, const NodePtr &end_node) const {
  if ((src_anchor == nullptr) || (end_node == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param src_anchor or end_node is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] parameter src_anchor or end_node is nullptr.");
    return nullptr;
  }
  AnchorPtr end_in_anchor;
  if (Anchor::DynamicAnchorCast<OutDataAnchor>(src_anchor) != nullptr) {
    end_in_anchor = end_node->GetInDataAnchor(0);
  } else {
    end_in_anchor = end_node->GetInControlAnchor();
  }
  return end_in_anchor;
}

AnchorPtr ge::EnginePartitioner::GetPldOutAnchor(const NodePtr &pld_node, const AnchorPtr &dst_anchor) const {
  if ((pld_node == nullptr) || (dst_anchor == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param pld_node or dst_anchor is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] parameter pld_node or dst_anchor is nullptr.");
    return nullptr;
  }
  AnchorPtr pld_out_anchor;
  if (Anchor::DynamicAnchorCast<InDataAnchor>(dst_anchor) != nullptr) {
    pld_out_anchor = pld_node->GetOutDataAnchor(0);
  } else {
    pld_out_anchor = pld_node->GetOutControlAnchor();
  }
  return pld_out_anchor;
}

void ge::EnginePartitioner::AddEndPldInformationToSubGraphInfo(ge::SubGraphInfoPtr &subgraph_info) {
  if (subgraph_info == nullptr) {
    GELOGE(FAILED, "[Check][Param] parameter subgraph_info is nullptr.");
    return;
  }
  auto subgraph = subgraph_info->GetSubGraph();
  GE_CHECK_NOTNULL_JUST_RETURN(subgraph);
  NodetoNodeMap end_map;
  NodetoNodeMap pld_map;
  for (const auto &node : subgraph->GetDirectNode()) {
    if (strcmp(node->GetTypePtr(), kEndType) == 0) {
      end_map[node] = graph_info_.end_2_pld_.at(node);
    }
    if (strcmp(node->GetTypePtr(), kPlaceHolderType) == 0) {
      pld_map[node] = graph_info_.pld_2_end_.at(node);
    }
  }
  subgraph_info->SetEnd2PldMap(end_map);
  subgraph_info->SetPld2EndMap(pld_map);
}

const Graph2SubGraphInfoList &ge::EnginePartitioner::GetSubGraphMap() { return graph_2_subgraph_list_; }

void ge::EnginePartitioner::ClearAllPartitionData() {
  graph_2_graph_partition_info_.clear();
  graph_2_subgraph_list_.clear();
  graph_2_input_subgraph_.clear();
  GELOGD("Clear all partition data success.");
}

const NodeEngineMap &EnginePartitioner::GetNodeEngineMap() const {
  return engine_placer_.GetNodeEngineMap(graph_info_.mode_ == Mode::kCompositeEnginePartitioning);
}

Status EnginePartitioner::UpdateCorrespondNodeInPartitions(const ComputeGraphPtr &compute_graph,
                                                           GraphPartitionInfo &graph_info) const {
  GELOGI("Graph partition info: %s", graph_info.output_name_.c_str());
  for (const auto &node_after_optimize : compute_graph->GetDirectNode()) {
    auto iter = graph_info.corresponding_node_in_partitions_.find(node_after_optimize->GetName());
    if (iter != graph_info.corresponding_node_in_partitions_.end()) {
      GELOGD("Update correspond node in partitions[%s]", node_after_optimize->GetName().c_str());
      iter->second = node_after_optimize;
    }
  }
  GELOGD("Graph partition update ends.");
  return SUCCESS;
}
}  // namespace ge
