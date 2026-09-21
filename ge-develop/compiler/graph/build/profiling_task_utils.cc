/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_task_utils.h"
#include <stack>
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "task_generator_utils.h"

using domi::LogTimeStampDef;
using domi::ModelTaskDef;
using domi::TaskDef;
namespace ge {
namespace {
const char *const kIteratorV2 = "IteratorV2";
const int64_t kProfilingArStep = 2;
const int64_t kProfilingFpStartLogid = 2;
const int64_t kProfilingBpEndLogid = 3;
const int64_t kProfilingArStartLogid = 10000;
const int64_t kProfilingArEndLogid = 10001;
const int64_t kProfilingIterEndLogid = 4;
const int64_t kProfilingGetNextStartLogid = 20000;
const int64_t kProfilingGetNextEndLogid = 20001;

bool IsSubGraphOfDynamicGraph(const ComputeGraphPtr &graph) {
  const auto &parent_graph_ptr = graph->GetParentGraph();
  if (parent_graph_ptr == nullptr) {
    return false;
  }
  const auto &root_graph_ptr = GraphUtils::FindRootGraph(parent_graph_ptr);  // root_graph_ptr must not be null
  return root_graph_ptr->GetGraphUnknownFlag();
}

bool IsFpNodeTypes(const std::string &type) {
  return OpTypeUtils::IsDataNode(type) || (type == ge::GETNEXT) || (type == kIteratorV2);
}

bool IsGetNextOpType(const OpDescPtr op_desc) {
  std::string origin_op_type;
  (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, origin_op_type);
  auto is_get_next = [](const std::string op_type) -> bool {
    return (op_type == GETNEXT || op_type == DYNAMICGETNEXT || op_type == DYNAMICGETNEXTV2 ||
            op_type == kIteratorV2);
  };
  return is_get_next(op_desc->GetType()) || is_get_next(origin_op_type);
}
}  // namespace

Status ProfilingTaskUtils::FindGetNextNodes(std::vector<int64_t> &get_next_nodes) const {
  for (const auto &node : nodes_) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(node->GetOpDesc());
    const auto current_idx = op_desc->GetId();
    std::string op_kernel_lib_name = op_desc->GetOpKernelLibName();
    if (IsGetNextOpType(op_desc)) {
      if (op_kernel_lib_name.empty()) {
        continue;
      }
      get_next_nodes.emplace_back(current_idx);
      GELOGI("GetNext name %s, idx %lld", op_desc->GetName().c_str(), current_idx);
    }
  }

  return SUCCESS;
}

Status ProfilingTaskUtils::FindFpOpCrossSubgraph(const NodePtr &node, int32_t index,
                                                 OpDescPtr &fp_op_desc) const {
  std::stack<NodePtr> node_stack;
  node_stack.push(node);
  while (!node_stack.empty()) {
    const auto cur_node = node_stack.top();
    node_stack.pop();
    GE_CHECK_NOTNULL(cur_node);
    const auto &data_node_vec = NodeUtils::GetSubgraphDataNodesByIndex(*cur_node, index);
    for (const auto& data_node : data_node_vec) {
      GE_CHECK_NOTNULL(data_node);
      const auto &out_anchor = data_node->GetOutDataAnchor(0);
      GE_CHECK_NOTNULL(out_anchor);
      for (const auto& peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        const auto &in_node_desc = peer_in_anchor->GetOwnerNodeBarePtr()->GetOpDesc();
        GE_CHECK_NOTNULL(in_node_desc);
        if (NoNeedGenTask(in_node_desc)) {
          continue;
        }
        if ((fp_op_desc == nullptr) || (in_node_desc->GetId() < fp_op_desc->GetId())) {
          fp_op_desc = in_node_desc;
        }
        if (!in_node_desc->GetSubgraphInstanceNames().empty()) {
          node_stack.push(peer_in_anchor->GetOwnerNode());
        }
      }
    }
  }
  return SUCCESS;
}

Status ProfilingTaskUtils::AutoFindFpOpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point) const {
  GELOGI("Start AutoFindFpOpIndex");
  OpDescPtr fp_op_desc = nullptr;
  for (const auto &node : nodes_) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    auto type = op_desc->GetType();
    std::string original_type;
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type);
    if (IsFpNodeTypes(type) || IsFpNodeTypes(original_type)) {
      const auto &out_anchor = node->GetOutDataAnchor(0);
      GE_CHECK_NOTNULL(out_anchor);
      for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        const auto &in_node = peer_in_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(in_node);
        const auto &in_node_desc = in_node->GetOpDesc();
        GE_CHECK_NOTNULL(in_node_desc);
        if (NoNeedGenTask(in_node_desc)) {
          continue;
        }
        (void)FindFpOpCrossSubgraph(in_node, peer_in_anchor->GetIdx(), fp_op_desc);
        if (!in_node_desc->GetSubgraphInstanceNames().empty()) {
          continue;
        }
        if ((fp_op_desc == nullptr) || (in_node_desc->GetId() < fp_op_desc->GetId())) {
          fp_op_desc = in_node_desc;
        }
      }
      break;
    }
  }

  if (fp_op_desc == nullptr) {
    GELOGW("not find fp_op_desc.");
    return SUCCESS;
  }
  GEEVENT("Auto find graph[%s]'s fp node[%s], type[%s], index[%ld], stream id[%ld]", graph->GetName().c_str(),
          fp_op_desc->GetName().c_str(), fp_op_desc->GetType().c_str(), fp_op_desc->GetId(), fp_op_desc->GetStreamId());
  profiling_point.fp_index = fp_op_desc->GetId();
  return SUCCESS;
}

Status ProfilingTaskUtils::AutoFindBpOpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                                             std::vector<int64_t> &all_reduce_nodes) const {
  GELOGI("Start AutoFindBpOpIndex");
  Node *bp_node = nullptr;
  for (const auto &node : nodes_) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto current_idx = op_desc->GetId();
    if (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE) {
      bp_node = node;
      all_reduce_nodes.emplace_back(current_idx);
      GELOGI("Allreduce name %s, idx %ld", op_desc->GetName().c_str(), current_idx);
    }
    if ((op_desc->GetType() == NETOUTPUT) && (node->GetOwnerComputeGraph()->GetParentNode() == nullptr)) {
      if (bp_node == nullptr) {
        bp_node = node;
      }
    }
    if (graph->GetNeedIteration()) {
      if (op_desc->GetName() == NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %ld, from Node_Output_IteratorCtrl_StreamSwitch_StreamActive",
               op_desc->GetName().c_str(), current_idx);
      }
      if (op_desc->GetName() == NODE_NAME_FLOWCTRL_LOOP_ASSIGN) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %ld, from FlowCtrl_LoopCond_ASSIGN", op_desc->GetName().c_str(), current_idx);
      }
    } else {
      if ((op_desc->GetType() == NETOUTPUT) && (node->GetOwnerComputeGraph()->GetParentNode() == nullptr)) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %ld, from NETOUTPUT", op_desc->GetName().c_str(), current_idx);
      }
    }
  }

  if (bp_node == nullptr) {
    GELOGW("not find bp_node.");
    return SUCCESS;
  }

  return FindLastBpFromBpNode(graph, bp_node, profiling_point.bp_index);
}

bool ProfilingTaskUtils::IsGlobalStep(const NodePtr &node) {
  if (node->GetName() == NODE_NAME_GLOBAL_STEP) {
    GELOGD("The node name is ge_global_step.");
    return true;
  }
  const auto &op_desc = node->GetOpDesc();  // op_desc must not be null
  for (const auto &subgraph_name : op_desc->GetSubgraphInstanceNames()) {
    const auto &graph = node->GetOwnerComputeGraph();  // graph must not be null
    const auto &subgraph = graph->GetSubgraph(subgraph_name);
    if (subgraph == nullptr) {
      continue;
    }
    if (subgraph->FindNode(NODE_NAME_GLOBAL_STEP) != nullptr) {
      GELOGD("The Node named global_step is found in graph.");
      return true;
    }
  }
  GELOGD("The Node named global_step is not found in graph.");
  return false;
}

Status ProfilingTaskUtils::FindLastBpFromBpNode(const ComputeGraphPtr &graph, const Node *target_node,
                                                int64_t &bp_index) {
  bp_index = 0;
  const auto &target_desc = target_node->GetOpDesc();
  GE_CHECK_NOTNULL(target_desc);
  OpDescPtr bp_op_desc = nullptr;
  for (const auto &in_node : target_node->GetInAllNodes()) {
    GE_CHECK_NOTNULL(in_node);
    const auto &in_node_desc = in_node->GetOpDesc();
    GE_CHECK_NOTNULL(in_node_desc);
    if (IsGlobalStep(in_node)) {
      GELOGD("The Node name is %s", in_node_desc->GetName().c_str());
      continue;
    }
    if (NoNeedGenTask(in_node_desc)) {
      continue;
    }
    (void)FindLastBpCrossSubgraph(in_node, bp_op_desc);
    if (!in_node_desc->GetSubgraphInstanceNames().empty()) {
      continue;
    }
    if (((bp_op_desc == nullptr) || (in_node_desc->GetId() > bp_op_desc->GetId())) &&
        (in_node_desc->GetStreamId() == target_desc->GetStreamId())) {
      bp_op_desc = in_node_desc;
    }
  }

  if (bp_op_desc == nullptr) {
    GELOGI("Did not find bp node.");
    return SUCCESS;
  }
  bp_index = bp_op_desc->GetId();
  GEEVENT("Auto find graph[%s]'s bp node[%s], type[%s], index[%ld], stream id[%ld]", graph->GetName().c_str(),
          bp_op_desc->GetName().c_str(), bp_op_desc->GetType().c_str(), bp_op_desc->GetId(), bp_op_desc->GetStreamId());
  return SUCCESS;
}

Status ProfilingTaskUtils::FindLastBpCrossSubgraph(const NodePtr &in_node, OpDescPtr &bp_op_desc) {
  std::stack<NodePtr> node_stack;
  node_stack.push(in_node);
  while (!node_stack.empty()) {
    const auto cur_node = node_stack.top();
    node_stack.pop();
    GE_CHECK_NOTNULL(cur_node);
    const auto &output_node_vec = ge::NodeUtils::GetSubgraphOutputNodes(*cur_node);
    if (!output_node_vec.empty() && (output_node_vec.at(0U) != nullptr)) {
      const auto &output_node = output_node_vec.at(0U);
      for (const auto &in_node_in_subgraph : output_node->GetInAllNodes()) {
        GE_CHECK_NOTNULL(in_node_in_subgraph);
        const auto &in_node_subgraph_desc = in_node_in_subgraph->GetOpDesc();
        GE_CHECK_NOTNULL(in_node_subgraph_desc);
        if (IsGlobalStep(in_node_in_subgraph) || NoNeedGenTask(in_node_subgraph_desc)) {
          continue;
        }
        if ((bp_op_desc == nullptr) || (in_node_subgraph_desc->GetId() > bp_op_desc->GetId())) {
          bp_op_desc = in_node_subgraph_desc;
        }
        if (!in_node_subgraph_desc->GetSubgraphInstanceNames().empty()) {
          node_stack.push(in_node_in_subgraph);
        }
      }
    }
  }
  return SUCCESS;
}

Status ProfilingTaskUtils::FindFpOfEnv(const ComputeGraphPtr &graph, const std::string &fp_point_str,
                                       ProfilingPoint &profiling_point) const {
  GELOGI("Start FindFpOfEnv");
  (void)graph;
  for (const auto &node : nodes_) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (IsProfPoint(op_desc, fp_point_str)) {
      profiling_point.fp_index = op_desc->GetId();
      GELOGI("First fp name from env is %s, idx %ld", op_desc->GetName().c_str(), profiling_point.fp_index);
      break;
    }
  }
  return SUCCESS;
}

Status ProfilingTaskUtils::FindBpOfEnv(const ComputeGraphPtr &graph, const std::string &bp_point_str,
                                       ProfilingPoint &profiling_point, std::vector<int64_t> &all_reduce_nodes) const {
  GELOGI("Start FindBpOfEnv");
  int64_t last_bp = 0;
  for (const auto &node : nodes_) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto current_idx = op_desc->GetId();
    std::string op_kernel_lib_name = op_desc->GetOpKernelLibName();
    if (op_kernel_lib_name.empty()) {
      continue;
    }

    if (graph->GetNeedIteration()) {
      if (op_desc->GetName() == NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %ld, from Node_Output_IteratorCtrl_StreamSwitch_StreamActive",
               op_desc->GetName().c_str(), current_idx);
      }
      if (op_desc->GetName() == NODE_NAME_FLOWCTRL_LOOP_ASSIGN) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %ld, from FlowCtrl_LoopCond_ASSIGN", op_desc->GetName().c_str(), current_idx);
      }
    } else {
      if (op_desc->GetName() == NODE_NAME_NET_OUTPUT) {
        profiling_point.end_index.insert(current_idx);
        GELOGI("Iter end name %s, idx %ld, from NETOUTPUT", op_desc->GetName().c_str(), current_idx);
      }
    }

    if (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE) {
      all_reduce_nodes.emplace_back(current_idx);
      GELOGI("Allreduce name %s, idx %ld", op_desc->GetName().c_str(), current_idx);
    }
    if (IsProfPoint(op_desc, bp_point_str)) {
      last_bp = current_idx;
      GELOGI("Last bp name from env is %s, idx %ld", op_desc->GetName().c_str(), last_bp);
    }
  }

  profiling_point.bp_index = last_bp;
  return SUCCESS;
}

void ProfilingTaskUtils::GetFpBpIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point,
                                      std::vector<int64_t> &all_reduce_nodes, std::string &fp_point_str,
                                      std::string &bp_point_str) const {
  if (fp_point_str.empty()) {
    (void)AutoFindFpOpIndex(graph, profiling_point);
  } else {
    GEEVENT("Find fp node set by user, graph[%s], node[%s].", graph->GetName().c_str(), fp_point_str.c_str());
    (void)FindFpOfEnv(graph, fp_point_str, profiling_point);
  }

  if (bp_point_str.empty()) {
    (void)AutoFindBpOpIndex(graph, profiling_point, all_reduce_nodes);
  } else {
    GEEVENT("Find bp node set by user, graph[%s], node[%s].", graph->GetName().c_str(), bp_point_str.c_str());
    (void)FindBpOfEnv(graph, bp_point_str, profiling_point, profiling_point.all_reduce_node_index);
  }
}

Status ProfilingTaskUtils::FindProfilingTaskIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point) const {
  GE_CHECK_NOTNULL(graph);
  const bool is_profiling =
      ProfilingProperties::Instance().ProfilingInited() || ProfilingProperties::Instance().ProfilingTrainingTraceOn();
  if (!is_profiling) {
    GELOGI("Profiling of graph:%s is not open.", graph->GetName().c_str());
    return SUCCESS;
  }

  // subgraph  of dynamic graph no need to find index, has been found in parent graph
  if (IsSubGraphOfDynamicGraph(graph)) {
    GELOGI("Graph[%s] is subgraph of dynamic graph, no need to find index.", graph->GetName().c_str());
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(FindGetNextNodes(profiling_point.get_next_node_index));
  GELOGI("Start get FP/BP index.");
  std::string fp_point_str;
  std::string bp_point_str;
  ProfilingProperties::Instance().GetFpBpPoint(fp_point_str, bp_point_str);
  GELOGI("User set show: {fp_point_str:%s, bp_point_str:%s}.", fp_point_str.c_str(), bp_point_str.c_str());
  GetFpBpIndex(graph, profiling_point, profiling_point.all_reduce_node_index, fp_point_str, bp_point_str);
  const bool train_graph = graph->GetNeedIteration();
  if (profiling_point.fp_index == 0 && train_graph) {
    GELOGW("First forward op name can't be found in graph for training trace.");
  }
  if (profiling_point.bp_index == 0 && train_graph) {
    GELOGW("Last backward op name can't be found in graph for training trace.");
  }
  for (const auto end_idx : profiling_point.end_index) {
    GEEVENT("Find end index: %ld, graph: %s.", end_idx, graph->GetName().c_str());
  }
  return SUCCESS;
}

void ProfilingTaskUtils::AssembleTaskForProfilerTrace(const int64_t stream_id, const int64_t log_id,
                                                      bool is_iteration_end,
                                                      std::vector<domi::TaskDef> &task_def_list) {
  TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  task_def.set_stream_id(static_cast<uint32_t>(stream_id));
  LogTimeStampDef *log_def = task_def.mutable_log_timestamp();
  if (log_def != nullptr) {
    log_def->set_logid(static_cast<uint64_t>(log_id));
    log_def->set_notify(is_iteration_end);
  }
  task_def_list.emplace_back(task_def);
  GELOGI("Assemble task for training trace, stream id: %lld, logid: %lld", stream_id, log_id);
}

Status ProfilingTaskUtils::CalculateLogId(const std::vector<int64_t> &trace_nodes,
                                          const int64_t node_index,
                                          const uint64_t start_id, int64_t &log_id) {
  for (size_t i = 0U; i < trace_nodes.size(); i++) {
    if (trace_nodes[i] == node_index) {
      log_id = static_cast<int64_t>(i) * kProfilingArStep + start_id;
      break;
    }
  }

  return SUCCESS;
}

Status ProfilingTaskUtils::InsertProfilingArTaskBefore(const OpDescPtr &op_desc,
                                                       const std::vector<int64_t> &all_reduce_nodes, int64_t node_index,
                                                       std::vector<domi::TaskDef> &task_def_list,
                                                       bool is_insert_bp_profiling_task) {
  bool is_insert_all_reduce_task = false;
  int64_t ar_log_id = 0xFFFF;
  if (is_insert_bp_profiling_task) {
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, ar_log_id);
    is_insert_all_reduce_task = true;
  }
  if (!is_insert_all_reduce_task) {
    for (size_t i = 0; i < all_reduce_nodes.size(); i++) {
      if (all_reduce_nodes[i] == node_index) {
        ar_log_id = static_cast<int64_t>(i) * kProfilingArStep + kProfilingArStartLogid;
        is_insert_all_reduce_task = true;
        break;
      }
    }
  }

  if (is_insert_all_reduce_task) {
    GELOGI("The start allreduce operator is %s, idx %ld, log_id %ld", op_desc->GetName().c_str(), node_index,
           ar_log_id);
    AssembleTaskForProfilerTrace(op_desc->GetStreamId(), ar_log_id, false, task_def_list);
  }

  return SUCCESS;
}

Status ProfilingTaskUtils::InsertProfilingTaskBefore(const OpDescPtr &op_desc, const ProfilingPoint &profiling_point,
                                                     std::vector<domi::TaskDef> &task_def_list) {
  GE_CHECK_NOTNULL(op_desc);
  const bool is_profiling =
      ProfilingProperties::Instance().ProfilingOn() || ProfilingProperties::Instance().ProfilingTrainingTraceOn();
  bool is_insert_fp_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_FP_PROFILILNG_TASK, is_insert_fp_profiling_task);
  bool is_insert_bp_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, is_insert_bp_profiling_task);
  const bool no_insert_profiling_task = ((profiling_point.bp_index == 0) || (profiling_point.end_index.empty())) &&
                                        (!(is_insert_fp_profiling_task || is_insert_bp_profiling_task));
  if (!is_profiling || no_insert_profiling_task) {
    return SUCCESS;
  }
  const int64_t node_index = op_desc->GetId();
  GELOGD(
      "Node index %ld insert fp profiling task: %d, insert bp profiling task: %d, fp index: %ld, bp index: %ld, end "
      "index size: %zu",
      node_index, is_insert_fp_profiling_task, is_insert_bp_profiling_task, profiling_point.fp_index,
      profiling_point.bp_index, profiling_point.end_index.size());
  if ((profiling_point.fp_index == node_index) || is_insert_fp_profiling_task) {
    GELOGI("The first FP operator is %s, idx %ld", op_desc->GetName().c_str(), node_index);
    AssembleTaskForProfilerTrace(op_desc->GetStreamId(), kProfilingFpStartLogid, false, task_def_list);
  }

  const bool is_all_reduce = (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE);
  if (is_all_reduce) {
    (void)InsertProfilingArTaskBefore(op_desc, profiling_point.all_reduce_node_index, node_index, task_def_list,
                                      is_insert_bp_profiling_task);
  }

  if (IsGetNextOpType(op_desc)) {
    int64_t log_id = -1;
    GE_CHK_STATUS_RET(CalculateLogId(profiling_point.get_next_node_index, node_index,
                                     kProfilingGetNextStartLogid, log_id));
    AssembleTaskForProfilerTrace(op_desc->GetStreamId(), log_id, false, task_def_list);
  }

  return SUCCESS;
}

Status ProfilingTaskUtils::InsertProfilingArTaskAfter(const OpDescPtr &op_desc,
                                                      const std::vector<int64_t> &all_reduce_nodes, int64_t node_index,
                                                      std::vector<domi::TaskDef> &task_def_list,
                                                      bool is_insert_bp_profiling_task) {
  bool is_insert_all_reduce_task = false;
  int64_t ar_log_id = 0xFFFF;
  if (is_insert_bp_profiling_task) {
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, ar_log_id);
    ar_log_id += 1;
    is_insert_all_reduce_task = true;
  }
  if (!is_insert_all_reduce_task) {
    for (size_t i = 0U; i < all_reduce_nodes.size(); i++) {
      if (all_reduce_nodes[i] == node_index) {
        ar_log_id = static_cast<int64_t>(i) * kProfilingArStep + kProfilingArEndLogid;
        is_insert_all_reduce_task = true;
        break;
      }
    }
  }

  if (is_insert_all_reduce_task) {
    GELOGI("The start allreduce operator is %s, idx %ld, log_id %ld", op_desc->GetName().c_str(), node_index,
           ar_log_id);
    TaskDef ar_task_def;
    ar_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
    ar_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *ar_log_def = ar_task_def.mutable_log_timestamp();
    if (ar_log_def != nullptr) {
      ar_log_def->set_logid(ar_log_id);
      ar_log_def->set_notify(false);
    }
    task_def_list.push_back(ar_task_def);
  }

  return SUCCESS;
}

Status ProfilingTaskUtils::InsertProfilingTaskAfter(const OpDescPtr &op_desc, const ProfilingPoint &profiling_point,
                                                    std::vector<domi::TaskDef> &task_def_list) {
  GE_CHECK_NOTNULL(op_desc);
  const int64_t node_index = op_desc->GetId();
  const bool is_profiling =
      ProfilingProperties::Instance().ProfilingOn() || ProfilingProperties::Instance().ProfilingTrainingTraceOn();
  bool is_insert_bp_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, is_insert_bp_profiling_task);
  bool is_insert_end_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_END_PROFILILNG_TASK, is_insert_end_profiling_task);
  const bool no_insert_profiling_task =
      profiling_point.end_index.empty() && (!(is_insert_bp_profiling_task || is_insert_end_profiling_task));
  if (!is_profiling || no_insert_profiling_task) {
    return SUCCESS;
  }
  GELOGD(
      "Node index %ld insert bp profiling task: %d, insert end profiling task: %d, fp index: %ld, bp index: %ld, end "
      "index size: %zu",
      node_index, is_insert_bp_profiling_task, is_insert_end_profiling_task, profiling_point.fp_index,
      profiling_point.bp_index, profiling_point.end_index.size());

  const bool is_all_reduce = (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE);
  if ((profiling_point.bp_index == node_index) || (!is_all_reduce && is_insert_bp_profiling_task)) {
    GELOGI("The last BP operator is %s, idx %ld", op_desc->GetName().c_str(), node_index);
    AssembleTaskForProfilerTrace(op_desc->GetStreamId(), kProfilingBpEndLogid, false, task_def_list);
  }

  if (profiling_point.end_index.find(node_index) != profiling_point.end_index.end() || is_insert_end_profiling_task) {
    GELOGI("The iteration end operator is %s, idx %ld", op_desc->GetName().c_str(), node_index);
    AssembleTaskForProfilerTrace(op_desc->GetStreamId(), kProfilingIterEndLogid, true, task_def_list);
  }

  if (is_all_reduce) {
    (void)InsertProfilingArTaskAfter(op_desc, profiling_point.all_reduce_node_index, node_index, task_def_list,
                                     is_insert_bp_profiling_task);
  }

  if (IsGetNextOpType(op_desc)) {
    int64_t log_id = -1;
    GE_CHK_STATUS_RET(CalculateLogId(profiling_point.get_next_node_index,
        node_index, kProfilingGetNextEndLogid, log_id));
    AssembleTaskForProfilerTrace(op_desc->GetStreamId(), log_id, false, task_def_list);
  }

  return SUCCESS;
}

bool ProfilingTaskUtils::IsProfPoint(const OpDescPtr &op, const string &name) {
  if (op->GetName() == name) {  // op must not be null
    return true;
  }
  std::vector<std::string> original_op_names;
  bool ret = AttrUtils::GetListStr(op, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);
  if (!ret) {
    return false;
  }
  for (auto &origin_name : original_op_names) {
    if (origin_name == name) {
      return true;
    }
  }
  return false;
}
}  // namespace ge
