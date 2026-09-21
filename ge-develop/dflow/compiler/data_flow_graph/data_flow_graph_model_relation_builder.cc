/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/data_flow_graph/data_flow_graph_model_relation_builder.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/utils/op_type_utils.h"

namespace {
constexpr char const *kNodeFlowInfoTypeIn = "in";
constexpr char const *kNodeFlowInfoTypeOut = "out";
}  // namespace

namespace ge {
Status DataFlowGraphModelRelationBuilder::DoBuildForFlowData(
    const DataFlowGraph &data_flow_graph, const NodePtr &node,
    std::map<NodePtr, std::map<int, std::string>> &paired_inputs) {
  std::string endpoint_name;
  GE_CHK_STATUS_RET(CreateQueueForDataNode(*node, data_flow_graph.GetName(), endpoint_name),
                    "Failed to create queue for data[%s].", node->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(DoBuildForNodeOutput(data_flow_graph, node, endpoint_name, paired_inputs));
  return SUCCESS;
}

Status DataFlowGraphModelRelationBuilder::DoBuildForNodeOutput(
    const DataFlowGraph &data_flow_graph,
    const NodePtr &node,
    const std::string &endpoint_name,
    std::map<NodePtr, std::map<int, std::string>> &paired_inputs) {
  const auto &out_data_anchor = node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(out_data_anchor);
  for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto &peer_node = in_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    if (peer_node->GetType() != FLOWNODE) {
      GELOGE(FAILED, "Peer node of Data is not a FlowNode, type = %s", peer_node->GetType().c_str());
      return FAILED;
    }
    ModelRelation::ModelEndpointInfo *dst_model_endpoints = nullptr;
    uint32_t queue_index = 0U;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateModelQueueInfoForDataFlowGraph(
        data_flow_graph, {peer_node->GetName(), in_data_anchor->GetIdx(), kNodeFlowInfoTypeIn}, dst_model_endpoints,
        queue_index));
    dst_model_endpoints->input_endpoint_names[queue_index] = endpoint_name;
    (void)paired_inputs[peer_node].emplace(in_data_anchor->GetIdx(), endpoint_name);
  }
  return SUCCESS;
}

Status DataFlowGraphModelRelationBuilder::DoBuildForFlowNode(
    const DataFlowGraph &data_flow_graph, const NodePtr &node,
    std::map<NodePtr, std::map<int, std::string>> &paired_inputs) {
  // check all input are valid
  std::vector<std::string> unused;
  GE_CHK_STATUS_RET_NOLOG(GetInputQueueNames(node, paired_inputs, unused));
  // create queue for submodels outputs, and set input to peer submodel
  Status ret = SUCCESS;
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    const int32_t output_idx = out_data_anchor->GetIdx();
    const std::string queue_name = data_flow_graph.GetDataFlowScope() + node->GetName() + ":"
                                   + std::to_string(output_idx);
    bool is_dummy = out_data_anchor->GetPeerInDataAnchors().empty();
    if (is_dummy) {
      GELOGI("Output queue[%s] is dummy.", queue_name.c_str());
    }
    GE_CHK_STATUS_RET(CreateQueueDef(node->GetOpDesc()->GetOutputDesc(output_idx), queue_name, *node, is_dummy),
                      "Create queue in model relation failed.");
    ModelRelation::ModelEndpointInfo *model_queues = nullptr;
    uint32_t queue_index = 0U;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateModelQueueInfoForDataFlowGraph(
        data_flow_graph, {node->GetName(), output_idx, kNodeFlowInfoTypeOut}, model_queues, queue_index));
    model_queues->output_endpoint_names[queue_index] = queue_name;
    for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(in_data_anchor);
      const auto &dequeue_node = in_data_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(dequeue_node);
      GE_CHECK_NOTNULL(dequeue_node->GetOpDesc());
      if (dequeue_node->GetType() == FLOWNODE) {
        ModelRelation::ModelEndpointInfo *dst_model_queues = nullptr;
        GE_CHK_STATUS_RET_NOLOG(GetOrCreateModelQueueInfoForDataFlowGraph(
            data_flow_graph, {dequeue_node->GetName(), in_data_anchor->GetIdx(), kNodeFlowInfoTypeIn}, dst_model_queues,
            queue_index));
        dst_model_queues->input_endpoint_names[queue_index] = queue_name;
      }
      (void)paired_inputs[dequeue_node].emplace(in_data_anchor->GetIdx(), queue_name);
    }
  }
  return ret;
}

Status DataFlowGraphModelRelationBuilder::GetOrCreateModelQueueInfoForDataFlowGraph(
    const DataFlowGraph &data_flow_graph, const NodeFlowInfo &node_flow_info,
    ModelRelation::ModelEndpointInfo *&model_endpoint_info, uint32_t &queue_index) {
  std::map<std::string, std::vector<std::pair<ComputeGraphPtr, uint32_t>>> nodes_map_graphs;
  if (node_flow_info.type == kNodeFlowInfoTypeIn) {
    nodes_map_graphs = data_flow_graph.GetNodesInputs();
  } else {
    nodes_map_graphs = data_flow_graph.GetNodesOutputs();
  }
  const std::map<std::string, std::vector<std::pair<ComputeGraphPtr, uint32_t>>>::const_iterator &node_map_graphs_it =
      nodes_map_graphs.find(node_flow_info.name);
  GE_CHK_BOOL_RET_STATUS(node_map_graphs_it != nodes_map_graphs.cend(), FAILED,
                         "Failed to get node[%s] from data flow graph[%s].", node_flow_info.name.c_str(),
                         data_flow_graph.GetName().c_str());
  GE_CHK_BOOL_RET_STATUS(static_cast<size_t>(node_flow_info.index) < node_map_graphs_it->second.size(), FAILED,
                         "The index[%d] need less than node[%s] %s size[%zu].", node_flow_info.index,
                         node_flow_info.name.c_str(), node_flow_info.type.c_str(), node_map_graphs_it->second.size());
  const auto &graph = node_map_graphs_it->second[node_flow_info.index].first;
  GE_CHECK_NOTNULL(graph);
  const std::string model_name = graph->GetName();
  queue_index = node_map_graphs_it->second[node_flow_info.index].second;
  const auto &it = model_relation_.submodel_endpoint_infos.find(model_name);
  if (it != model_relation_.submodel_endpoint_infos.end()) {
    model_endpoint_info = &it->second;
    return SUCCESS;
  }
  auto &ret = model_relation_.submodel_endpoint_infos[model_name];
  ret.model_name = model_name;
  ret.input_endpoint_names.resize(graph->GetInputNodes().size());
  ret.output_endpoint_names.resize(graph->GetOutputNodes().size());
  model_endpoint_info = &ret;
  if (node_flow_info.type == kNodeFlowInfoTypeIn) {
    GE_CHK_BOOL_RET_STATUS(static_cast<size_t>(queue_index) < ret.input_endpoint_names.size(), FAILED,
                           "The queue index[%u] is out of range of input queue size[%zu].", queue_index,
                           ret.input_endpoint_names.size());
  }
  if (node_flow_info.type == kNodeFlowInfoTypeOut) {
    GE_CHK_BOOL_RET_STATUS(static_cast<size_t>(queue_index) < ret.output_endpoint_names.size(), FAILED,
                           "The queue index[%u] is out of range of output queue size[%zu].", queue_index,
                           ret.output_endpoint_names.size());
  }
  return SUCCESS;
}

Status DataFlowGraphModelRelationBuilder::BuildFromDataFlowGraph(const DataFlowGraph &data_flow_graph,
                                                                 std::unique_ptr<ModelRelation> &model_relation) {
  GELOGD("Begin to build model relation from data flow graph[%s].", data_flow_graph.GetName().c_str());
  const auto &root_graph = data_flow_graph.GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  model_relation_.root_model_endpoint_info.model_name = root_graph->GetName();
  std::map<NodePtr, std::map<int, std::string>> paired_inputs;
  GE_CHK_STATUS_RET(root_graph->TopologicalSorting(), "Failed to topological sort, graph[%s].",
                    root_graph->GetName().c_str());
  Status ret = SUCCESS;
  for (const auto &node : root_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    const auto &op_type = node->GetType();
    if (OpTypeUtils::IsDataNode(op_type)) {
      GE_CHK_STATUS_RET_NOLOG(DoBuildForFlowData(data_flow_graph, node, paired_inputs));
    } else if (op_type == FLOWNODE) {
      const auto temp_ret = DoBuildForFlowNode(data_flow_graph, node, paired_inputs);
      if (temp_ret != SUCCESS) {
        GELOGE(temp_ret, "Failed to build model relation for node[%s].", node->GetName().c_str());
        ret = temp_ret;
      }
    } else if (op_type == NETOUTPUT) {
      GE_CHK_STATUS_RET_NOLOG(
          GetInputQueueNames(node, paired_inputs, model_relation_.root_model_endpoint_info.output_endpoint_names));
    } else {
      GELOGW("Error node type, name = %s, type = %s", node->GetName().c_str(), op_type.c_str());
    }
  }
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to build model relation from data flow graph[%s].", data_flow_graph.GetName().c_str());
    return ret;
  }
  model_relation = MakeUnique<ModelRelation>();
  GE_CHECK_NOTNULL(model_relation);
  *model_relation = std::move(model_relation_);
  GELOGD("Build model relation from data flow graph[%s] successfully.", data_flow_graph.GetName().c_str());
  return SUCCESS;
}
}  // namespace ge
