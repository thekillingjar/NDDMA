/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/data_flow_graph/data_flow_graph.h"
#include "nlohmann/json.hpp"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "dflow/compiler/data_flow_graph/process_point_loader.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph_prune_pass.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph_utils.h"
#include "dflow/compiler/data_flow_graph/convert_batch_attr_to_udf_pass.h"
#include "graph/utils/op_type_utils.h"
#include "graph/passes/pass_manager.h"

namespace ge {
namespace {
constexpr const char *kModelPpFusionInputs = "INVOKED_MODEL_FUSION_INPUTS";
}
const std::vector<std::string> &DataFlowGraph::GetInvokeKeys(const std::string &graph_name) const {
  const std::map<std::string, std::vector<std::string>>::const_iterator it = invokes_.find(graph_name);
  if (it != invokes_.cend()) {
    return it->second;
  }
  GELOGW("The graph[%s] does not has invoke keys.", graph_name.c_str());
  static std::vector<std::string> empty_ret;
  return empty_ret;
}

const std::map<std::string, std::string> &DataFlowGraph::GetGraphBuildOptions(const std::string &graph_name) const {
  const std::map<std::string, std::map<std::string, std::string>>::const_iterator it =
      graphs_build_options_.find(graph_name);
  if (it != graphs_build_options_.cend()) {
    return it->second;
  }
  GELOGW("The graph[%s] does not has build options.", graph_name.c_str());
  static std::map<std::string, std::string> empty_ret;
  return empty_ret;
}

const std::string &DataFlowGraph::GetInvokedGraphKey(const std::string &graph_name) const {
  const std::map<std::string, std::string>::const_iterator it = invoked_keys_.find(graph_name);
  if (it != invoked_keys_.cend()) {
    return it->second;
  }
  GELOGW("The graph[%s] does not has invoked key.", graph_name.c_str());
  static std::string empty_ret;
  return empty_ret;
}

const std::string &DataFlowGraph::GetInvokedKeyOriginName(const std::string &invoke_key) const {
  const std::map<std::string, std::string>::const_iterator it = invoke_origins_.find(invoke_key);
  if (it != invoke_origins_.cend()) {
    return it->second;
  }
  GELOGW("The invoke key with scope[%s] does not has original invoke key.", invoke_key.c_str());
  static std::string empty_ret;
  return empty_ret;
}

bool DataFlowGraph::InvokedByBuiltIn(const std::string &invoke_key) const {
  const auto &it = invoked_by_built_in_.find(invoke_key);
  if (it != invoked_by_built_in_.cend()) {
    return it->second;
  }
  return false;
}

Status DataFlowGraph::AddLoadedModel(const std::string &node_name, const std::string &graph_name,
                                     const FlowModelPtr &model) {
  std::lock_guard<std::mutex> lock(loaded_models_mt_);
  const auto find_ret = loaded_models_.find(graph_name);
  if (find_ret != loaded_models_.end()) {
    GELOGE(FAILED, "model[%s] is repeated.", graph_name.c_str());
    return FAILED;
  }
  loaded_models_[graph_name] = model;
  node_loaded_models_[node_name].emplace_back(model);
  return SUCCESS;
}

bool DataFlowGraph::IsInvokedGraph(const std::string &graph_name) const {
  return invoked_keys_.count(graph_name) > 0;
}

Status DataFlowGraph::CheckAlignAttrs(bool &align_enable) const {
  int64_t cache_num = 0;
  align_enable = AttrUtils::GetInt(root_graph_, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, cache_num);
  if (align_enable) {
    align_enable = (cache_num != 0);
    // max cache num is 1024
    GE_CHK_BOOL_RET_STATUS((cache_num >= 0) && (cache_num <= 1024), PARAM_INVALID,
                           "attr[%s]=%ld is out of range [0, 1024]",
                           dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, cache_num);
  }
  int64_t timeout = 0;
  if (AttrUtils::GetInt(root_graph_, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, timeout)) {
    // -1 means no time out, max value is 600 * 1000ms
    GE_CHK_BOOL_RET_STATUS((timeout == (-1)) || ((timeout > 0) && (timeout <= 600 * 1000)), PARAM_INVALID,
                           "attr[%s]=%ld is invalid, must be -1 or in range(0, 600 * 1000]",
                           dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, timeout);
  }
  return SUCCESS;
}

Status DataFlowGraph::CheckAndFixDataFlowAttrs() const {
  bool exception_catch = false;
  (void)AttrUtils::GetBool(root_graph_, dflow::ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH, exception_catch);
  bool align_enable = false;
  GE_CHK_STATUS_RET(CheckAlignAttrs(align_enable), "Check align attr failed.");
  if (exception_catch) {
    if (!align_enable) {
      GELOGE(PARAM_INVALID, "It is not supported exception catch is enable while align is disable.");
      return PARAM_INVALID;
    }
    GE_CHK_STATUS_RET(DataFlowGraphUtils::EnsureNMappingAttr(root_graph_), "Failed to set n-mapping attr for graph[%s]",
                      root_graph_->GetName().c_str());
  }
  return SUCCESS;
}

Status DataFlowGraph::Initialize() {
  GE_CHECK_NOTNULL(root_graph_);
  graph_name_ = data_flow_scope_.empty() ? root_graph_->GetName() : data_flow_scope_ + "/" + root_graph_->GetName();
  GE_TRACE_START(Initialize);
  GE_DUMP(root_graph_, "DataFlowGraph");
  GE_CHK_STATUS_RET(CheckAndFixDataFlowAttrs(), "Failed to check dataflow attrs for graph[%s].", graph_name_.c_str());
  PassManager pass_manager;
  GE_CHK_STATUS_RET(pass_manager.AddPass("DataFlowGraphPrunePass", new (std::nothrow) DataFlowGraphPrunePass));
  GE_CHK_STATUS_RET(pass_manager.AddPass("ConvertBatchAttrToUdfPass", new (std::nothrow) ConvertBatchAttrToUdfPass));
  GE_CHK_STATUS_RET(pass_manager.Run(root_graph_), "Failed to run data flow passes for graph[%s].",
                    graph_name_.c_str());
  for (const NodePtr &node : root_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (NeedSkip(node->GetType())) {
      continue;
    }
    if (node->GetType() == FLOWNODE) {
      const auto node_name = node->GetName();
      GE_CHK_BOOL_RET_STATUS((node_subgraphs_.find(node_name) == node_subgraphs_.cend()), FAILED,
                             "The node[%s] of data flow graph[%s] is already initialized.", node_name.c_str(),
                             graph_name_.c_str());
      GE_CHK_STATUS_RET(CheckFlowNode(node), "Failed to check flow node[%s] of data flow graph[%s]", node_name.c_str(),
                        graph_name_.c_str());
      GE_CHK_STATUS_RET(InitializeFlowNode(node), "Failed to initialize node[%s] of data flow graph[%s].",
                        node_name.c_str(), graph_name_.c_str());
      continue;
    }
    GELOGE(FAILED, "The dataflow graph can only have FlowData and FlowNode, but got %s", node->GetType().c_str());
    return FAILED;
  }
  GE_CHK_STATUS_RET_NOLOG(CheckGraph());
  GE_CHK_STATUS_RET(WaitPreprocessTaskFinish(), "wait preprocess task finish failed, graph[%s]", graph_name_.c_str());

  // root graph subgraphs are unused and maybe invalid graph, need remove them, otherwise normalize will fail.
  auto subgraphs = root_graph_->GetAllSubgraphs();
  for (const auto &subgraph : subgraphs) {
    GELOGD("Remove unused dataflow subgraph[%s]", subgraph->GetName().c_str());
    GE_CHK_STATUS_RET(ProcessPointLoader::RemoveGraphFromParent(root_graph_, subgraph),
                      "Remove unused dataflow subgraph[%s] failed", subgraph->GetName().c_str());
  }

  std::string trace_log = "Initialize dataflow graph[" + graph_name_ + "] during building";
  GE_COMPILE_TRACE_TIMESTAMP_END(Initialize, trace_log.c_str());
  return SUCCESS;
}

Status DataFlowGraph::UpdateInputsFlowAttrs(const NodePtr &node) {
  const auto &node_name = node->GetName();
  const auto &op_desc = node->GetOpDesc();
  for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
    const auto &target = nodes_inputs_[node_name][i];
    if (target.first == nullptr) {
      // not map input
      continue;
    }
    for (const auto &n : target.first->GetInputNodes()) {
      if (!OpTypeUtils::IsDataNode(n->GetType())) {
        continue;
      }
      int64_t index = 0;
      GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(n->GetOpDesc(), ATTR_NAME_INDEX, index), FAILED,
                             "Failed to get attr[%s] from op[%s].", ATTR_NAME_INDEX.c_str(),
                             n->GetOpDesc()->GetName().c_str());
      if (index != static_cast<int64_t>(target.second)) {
        continue;
      }
      auto data_out_anchor = n->GetOutDataAnchor(0);
      GE_CHECK_NOTNULL(data_out_anchor);
      for (const auto &in_data_anchor : data_out_anchor->GetPeerInDataAnchors()) {
        GE_CHECK_NOTNULL(in_data_anchor);
        const auto &dst_node = in_data_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(dst_node);
        const auto &dst_op_desc = dst_node->GetOpDesc();
        GE_CHECK_NOTNULL(dst_op_desc);
        auto dst_input_desc = dst_op_desc->MutableInputDesc(in_data_anchor->GetIdx());
        for (const auto &iter : op_desc->GetInputDescPtr(i)->GetAllAttrs()) {
          GE_CHK_STATUS_RET(dst_input_desc->SetAttr(iter.first, iter.second),
                            "Failed to set attr[%s] to op[%s] input[%zu]", iter.first.c_str(),
                            dst_op_desc->GetName().c_str(), i);
        }
      }
    }
  }
  return SUCCESS;
}

void DataFlowGraph::GetInOrOutIndex(const std::vector<std::pair<ComputeGraphPtr, uint32_t>> &vec, size_t &index) const {
  index = 0U;
  for (const auto &it : vec) {
    if (it.first == nullptr && it.second == 0) {
      return;
    }
    ++index;
  }
}

Status DataFlowGraph::MapNodeInputs(const NodePtr &node, const dataflow::ProcessPoint &process_point) {
  const std::string &node_name = node->GetName();
  const std::string &process_point_name = process_point.name();
  GELOGD("Begin to map node[%s] inputs from process point[%s].", node_name.c_str(), process_point_name.c_str());
  const auto &graph = subgraphs_[process_point_name];
  GE_CHK_BOOL_RET_STATUS((process_point.in_edges_size() == 0) ||
                             (static_cast<size_t>(process_point.in_edges_size()) == graph->GetInputNodes().size()),
                         FAILED,
                         "The process point[%s] in edges size[%d] not equal to process point graph inputs num[%zu].",
                         process_point_name.c_str(), process_point.in_edges_size(), graph->GetInputNodes().size());
  if (process_point.in_edges_size() == 0) {
    // auto map
    size_t input_index = 0U;
    GetInOrOutIndex(nodes_inputs_[node_name], input_index);
    GE_CHK_BOOL_RET_STATUS((nodes_inputs_[node_name].size() - input_index) >= graph->GetInputNodes().size(), FAILED,
                           "The node[%s] not mapped input num[%zu] should >= process point[%s] graph inputs num[%zu].",
                           node_name.c_str(), (nodes_inputs_[node_name].size() - input_index),
                           process_point_name.c_str(), graph->GetInputNodes().size());
    for (size_t i = 0U; i < graph->GetInputNodes().size(); ++i) {
      GE_CHK_BOOL_RET_STATUS(
          nodes_inputs_[node_name][input_index].first == nullptr, FAILED,
          "Failed to map process point[%s] input[%zu] to node[%s] input[%zu], which already mapped to "
          "process point[%s] input[%u].",
          process_point_name.c_str(), i, node_name.c_str(), input_index,
          nodes_inputs_[node_name][input_index].first->GetName().c_str(), nodes_inputs_[node_name][input_index].second);
      nodes_inputs_[node_name][input_index] = {graph, static_cast<uint32_t>(i)};
      ++input_index;
    }
  } else {
    for (int32_t i = 0; i < process_point.in_edges_size(); ++i) {
      const auto &map_node_name = process_point.in_edges(i).node_name();
      const auto map_node_index = process_point.in_edges(i).index();
      GE_CHK_BOOL_RET_STATUS(nodes_inputs_.find(map_node_name) != nodes_inputs_.cend(), FAILED,
                             "Can't find node[%s] of process point[%s] in edges[%d].", map_node_name.c_str(),
                             process_point_name.c_str(), i);
      GE_CHK_BOOL_RET_STATUS(map_node_index < nodes_inputs_[map_node_name].size(), FAILED,
                             "The process point[%s] in edges[%d] index[%u] is out of rang node[%s] inputs num[%zu].",
                             process_point_name.c_str(), i, map_node_index, map_node_name.c_str(),
                             nodes_inputs_[map_node_name].size());
      GE_CHK_BOOL_RET_STATUS(nodes_inputs_[map_node_name][map_node_index].first == nullptr, FAILED,
                             "Failed to map process point[%s] input[%d] to node[%s] input[%u], which already mapped to "
                             "process point[%s] input[%u].",
                             process_point_name.c_str(), i, map_node_name.c_str(), map_node_index,
                             nodes_inputs_[map_node_name][map_node_index].first->GetName().c_str(),
                             nodes_inputs_[map_node_name][map_node_index].second);
      nodes_inputs_[map_node_name][map_node_index] = {graph, static_cast<uint32_t>(i)};
    }
  }
  GELOGD("Map node[%s] inputs from process point[%s] successfully, map inputs num[%zu].", node_name.c_str(),
         process_point_name.c_str(), graph->GetInputNodes().size());
  return SUCCESS;
}

Status DataFlowGraph::MapNodeOutputs(const NodePtr &node, const dataflow::ProcessPoint &process_point) {
  const std::string &node_name = node->GetName();
  const std::string &process_point_name = process_point.name();
  GELOGD("Begin to map node[%s] outputs from process point[%s].", node_name.c_str(), process_point_name.c_str());
  const auto &graph = subgraphs_[process_point_name];
  GE_CHK_BOOL_RET_STATUS((process_point.out_edges_size() == 0) ||
                             (static_cast<size_t>(process_point.out_edges_size()) == graph->GetOutputNodes().size()),
                         FAILED,
                         "The process point[%s] out edges size[%d] not equal to process point graph outputs num[%zu].",
                         process_point_name.c_str(), process_point.out_edges_size(), graph->GetOutputNodes().size());
  if (process_point.out_edges_size() == 0) {
    // auto map
    size_t output_index = 0U;
    GetInOrOutIndex(nodes_outputs_[node_name], output_index);
    GE_CHK_BOOL_RET_STATUS(
        (nodes_outputs_[node_name].size() - output_index) >= graph->GetOutputNodes().size(), FAILED,
        "The node[%s] not mapped output num[%zu] should >= process point[%s] graph outputs num[%zu].",
        node_name.c_str(), (nodes_outputs_[node_name].size() - output_index), process_point_name.c_str(),
        graph->GetOutputNodes().size());
    for (size_t i = 0; i < graph->GetOutputNodes().size(); ++i) {
      GE_CHK_BOOL_RET_STATUS(
          nodes_outputs_[node_name][output_index].first == nullptr, FAILED,
          "Failed to map process point[%s] output[%zu] to node[%s] output[%zu], which already mapped to "
          "process point[%s] output[%u].",
          process_point_name.c_str(), i, node_name.c_str(), output_index,
          nodes_outputs_[node_name][output_index].first->GetName().c_str(),
          nodes_outputs_[node_name][output_index].second);
      nodes_outputs_[node_name][output_index] = {graph, static_cast<uint32_t>(i)};
      ++output_index;
    }
  } else {
    for (int32_t i = 0; i < process_point.out_edges_size(); ++i) {
      const auto &map_node_name = process_point.out_edges(i).node_name();
      const auto map_node_index = process_point.out_edges(i).index();
      GE_CHK_BOOL_RET_STATUS((nodes_outputs_.find(map_node_name) != nodes_outputs_.cend()), FAILED,
                             "Can't find node[%s] of process point[%s] out edges[%d].", map_node_name.c_str(),
                             process_point_name.c_str(), i);
      GE_CHK_BOOL_RET_STATUS((map_node_index < nodes_outputs_[map_node_name].size()), FAILED,
                             "The process point[%s] out edges[%d] index[%u] is out of rang node[%s] outputs num[%zu].",
                             process_point_name.c_str(), i, map_node_index, map_node_name.c_str(),
                             nodes_outputs_[map_node_name].size());
      GE_CHK_BOOL_RET_STATUS(
          nodes_outputs_[map_node_name][map_node_index].first == nullptr, FAILED,
          "Failed to map process point[%s] output[%d] to node[%s] output[%u], which already mapped to "
          "process point[%s] output[%u].",
          process_point_name.c_str(), i, map_node_name.c_str(), map_node_index,
          nodes_outputs_[map_node_name][map_node_index].first->GetName().c_str(),
          nodes_outputs_[map_node_name][map_node_index].second);
      nodes_outputs_[map_node_name][map_node_index] = {graph, i};
    }
  }
  GELOGD("Map node[%s] outputs from process point[%s] successfully, map outputs num[%zu].", node_name.c_str(),
         process_point_name.c_str(), graph->GetOutputNodes().size());
  return SUCCESS;
}

Status DataFlowGraph::MapNodeInputsAndOutputs(const NodePtr &node, const dataflow::ProcessPoint &process_point) {
  const std::string &node_name = node->GetName();
  const std::string &process_point_name = process_point.name();
  GELOGD("Begin to map node[%s] inputs and outputs from process point[%s].", node_name.c_str(),
         process_point_name.c_str());
  GE_CHK_STATUS_RET_NOLOG(MapNodeInputs(node, process_point));
  GE_CHK_STATUS_RET_NOLOG(MapNodeOutputs(node, process_point));
  GELOGD("Map node[%s] inputs and outputs from process point[%s] successfully.", node_name.c_str(),
         process_point_name.c_str());
  return SUCCESS;
}

Status DataFlowGraph::CheckGraph() const {
  for (const auto &it : nodes_inputs_) {
    for (size_t i = 0; i < it.second.size(); ++i) {
      if (it.second[i].first == nullptr) {
        GELOGE(FAILED, "The node[%s] input[%zu] not mapped.", it.first.c_str(), i);
        return FAILED;
      }
    }
  }
  for (const auto &it : nodes_outputs_) {
    for (size_t i = 0; i < it.second.size(); ++i) {
      if (it.second[i].first == nullptr) {
        GELOGE(FAILED, "The node[%s] output[%zu] not mapped.", it.first.c_str(), i);
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status DataFlowGraph::CheckFlowNode(const NodePtr &node) const {
  const auto in_all_nodes = node->GetInAllNodes();
  if (in_all_nodes.size() <= 1) {
    return SUCCESS;
  }
  size_t with_balance_attr_num = 0;
  for (const auto &in_node : in_all_nodes) {
    const auto op_desc = in_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const bool has_balance_scatter = AttrUtils::HasAttr(op_desc, dflow::ATTR_NAME_BALANCE_SCATTER);
    const bool has_balance_gather = AttrUtils::HasAttr(op_desc, dflow::ATTR_NAME_BALANCE_GATHER);
    if (has_balance_scatter || has_balance_gather) {
      GELOGD("node[%s] input node[%s] is balance node", node->GetNamePtr(), in_node->GetNamePtr());
      ++with_balance_attr_num;
    }
  }
  if ((with_balance_attr_num != 0) && (with_balance_attr_num != in_all_nodes.size())) {
    GELOGE(PARAM_INVALID, "node[%s] has %zu input nodes, but only %zu nodes have balance attr.", node->GetNamePtr(),
           in_all_nodes.size(), with_balance_attr_num);
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status DataFlowGraph::InitializeFlowNode(const NodePtr &node) {
  std::string node_name = node->GetName();
  std::vector<std::string> pps;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetListStr(node->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pps),
                         FAILED, "Failed to get process points of node[%s]", node_name.c_str());
  nodes_inputs_[node_name].resize(node->GetOpDesc()->GetInputsSize(), {nullptr, 0});
  nodes_outputs_[node_name].resize(node->GetOpDesc()->GetOutputsSize(), {nullptr, 0});
  GELOGD("The node[%s] inputs num is: [%zu], outputs num is: [%zu].", node_name.c_str(),
         nodes_inputs_[node_name].size(), nodes_outputs_[node_name].size());
  for (const auto &pp : pps) {
    dataflow::ProcessPoint process_point;
    GE_CHK_BOOL_RET_STATUS(process_point.ParseFromString(pp), FAILED, "Failed to parse process point[%s] of node[%s].",
                           pp.c_str(), node_name.c_str());
    GELOGD("The process point[%s] debug string: { %s }", process_point.name().c_str(),
           process_point.DebugString().c_str());
    GE_CHK_STATUS_RET(ProcessPointLoader::LoadProcessPoint(process_point, *this, node),
                      "Failed to load process point[%s] of node[%s].", process_point.name().c_str(),
                      node->GetName().c_str());
    GE_CHK_STATUS_RET(MapNodeInputsAndOutputs(node, process_point),
                      "Failed map node[%s] inputs and outputs from process point[%s].", node_name.c_str(),
                      process_point.name().c_str());
  }
  GE_CHK_STATUS_RET(UpdateInputsFlowAttrs(node), "Failed to update attrs for node[%s].", node_name.c_str());
  return SUCCESS;
}

Status DataFlowGraph::CommitPreprocessTask(const std::string &name, std::function<Status()> &task) {
  std::future<Status> f = thread_pool_.commit(task);
  GE_CHK_BOOL_RET_STATUS(f.valid(), FAILED, "Failed to commit process task, name[%s].", name.c_str());
  preprocess_tasks_[name] = std::move(f);
  return SUCCESS;
}

Status DataFlowGraph::GetInvokedModelFusionAttrs(const std::vector<std::string> &invoke_keys,
                                                 std::string &invoked_model_attrs) const {
  std::map<std::string, std::string> invoked_and_attr;
  for (const auto &invoke_key : invoke_keys) {
    const auto iter = invoked_graphs_.find(invoke_key);
    if (iter == invoked_graphs_.cend()) {
      continue;
    }
    const auto &invoked_graph = iter->second;
    const auto model_iter = loaded_models_.find(invoked_graph);
    if (model_iter == loaded_models_.cend()) {
      continue;
    }
    const auto &flow_mdoel = model_iter->second;
    GE_CHECK_NOTNULL(flow_mdoel);
    const auto root_graph = flow_mdoel->GetRootGraph();
    GE_CHECK_NOTNULL(root_graph);
    std::string fusion_inputs;
    (void)AttrUtils::GetStr(root_graph, kModelPpFusionInputs, fusion_inputs);
    if (!fusion_inputs.empty()) {
      invoked_and_attr[invoke_key] = fusion_inputs;
      GELOGD("Find fusion attr[%s] for invokde key[%s]", fusion_inputs.c_str(), invoke_key.c_str());
    }
  }
  if (invoked_and_attr.empty()) {
    return SUCCESS;
  }
  try {
    const nlohmann::json js = invoked_and_attr;
    invoked_model_attrs = js.dump();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to dump invoke attrs, err = %s", e.what());
    return FAILED;
  }
  return SUCCESS;
}

Status DataFlowGraph::WaitPreprocessTaskFinish() {
  GELOGI("wait dataflow preprocess task finish begin, task_size=%zu.", preprocess_tasks_.size());
  for (auto &preprocess_task : preprocess_tasks_) {
    GELOGD("wait dataflow preprocess task[%s] begin.", preprocess_task.first.c_str());
    auto ret = preprocess_task.second.get();
    GE_CHK_STATUS_RET(ret, "preprocess task failed, name[%s].", preprocess_task.first.c_str());
    GELOGD("wait dataflow preprocess task[%s] end.", preprocess_task.first.c_str());
  }
  preprocess_tasks_.clear();
  GELOGI("wait dataflow preprocess task finish end.");
  return SUCCESS;
}

bool DataFlowGraph::NeedSkip(const string &op_type) {
  return OpTypeUtils::IsDataNode(op_type) || (op_type == NETOUTPUT);
}
}  // namespace ge
