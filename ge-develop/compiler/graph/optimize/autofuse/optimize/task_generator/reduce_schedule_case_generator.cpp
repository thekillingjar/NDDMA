/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <queue>
#include "graph/utils/graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"
#include "schedule_utils.h"
#include "ascir_utils.h"
#include "node_utils.h"
#include "reduce_schedule_case_generator.h"
#include "register/op_def_factory.h"
#include "base/err_msg.h"
#include "graph/symbolizer/symbolic.h"

namespace optimize {
namespace {
size_t TWO = 2;
size_t kMaxFullLoadAxisSize = 3UL;
size_t NODE_COUNT_AFTER_REDUCE = 4UL;
std::string GetNewNodeName(const ge::AscNodePtr &src_node, const ge::AscNodePtr &dst_node,
                           const std::string &type, int32_t idx) {
  return src_node->GetName() + "_to_" + dst_node->GetName() + "_" + type + "_" + to_string(idx);
}

Status DoCopyAscNodeTensorAttr(const ge::AscNodePtr &src_node, ge::AscNodePtr &dst_node) {
  auto op_desc = dst_node->GetOpDesc();
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<ge::AscNodeAttr>();
  auto src_asc_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<ge::AscNodeAttr>();
  if (src_asc_node_attr != nullptr && dst_asc_node_attr != nullptr) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
    if (src_asc_node_attr->ir_attr) {
      dst_asc_node_attr->ir_attr = src_asc_node_attr->ir_attr->Clone();
    }
  }
  for (size_t i = 0U; i < src_node->outputs().size(); i++) {
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(i));
    auto tensor_attr_group = op_desc->MutableOutputDesc(i)->GetOrCreateAttrsGroup<ge::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    *tensor_attr_group = src_node->outputs[i].attr;
  }
  return ge::SUCCESS;
}

Status DoCopyWorkspaceTensorAttr(const ge::AscNodePtr &src_node, ge::AscNodePtr &workspace_node) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(workspace_node);
  GE_ASSERT_TRUE(!src_node->outputs().empty());
  GE_ASSERT_TRUE(!workspace_node->outputs().empty());
  workspace_node->outputs[0].attr.dtype = src_node->outputs[0].attr.dtype;
  return ge::SUCCESS;
}

const std::unordered_map<std::string, std::function<ReduceType(const char*)>> reducers = {
  {"Max",  [](const char* n) { return ReduceType{std::in_place_type_t<ge::ascir_op::Max>{}, n}; }},
  {"Sum",  [](const char* n) { return ReduceType{std::in_place_type_t<ge::ascir_op::Sum>{}, n}; }},
  {"Mean", [](const char* n) { return ReduceType{std::in_place_type_t<ge::ascir_op::Sum>{}, n}; }},  // Mean 二阶段使用 Sum
  {"Min",  [](const char* n) { return ReduceType{std::in_place_type_t<ge::ascir_op::Min>{}, n}; }},
  {"Prod", [](const char* n) { return ReduceType{std::in_place_type_t<ge::ascir_op::Prod>{}, n}; }},
  {"Any",  [](const char* n) { return ReduceType{std::in_place_type_t<ge::ascir_op::Any>{}, n}; }},
  {"All",  [](const char* n) { return ReduceType{std::in_place_type_t<ge::ascir_op::All>{}, n}; }}
};

bool IsNotPartitionReduce(const ge::AscNodePtr &reduce_node, size_t threshold) {
  std::queue<ge::NodePtr> node_queue;
  size_t node_count = 0UL;
  std::unordered_set<const ge::Node*> visited;
  visited.insert(reduce_node.get());
  for (const auto &reduce_out_node : reduce_node->GetOutNodes()) {
    node_queue.emplace(reduce_out_node);
    visited.insert(reduce_out_node.get());
  }
  while (!node_queue.empty()) {
    const auto current_node = node_queue.front();
    node_queue.pop();

    if (current_node->GetInDataNodesSize() > 1UL) {
      for (const auto &in_current_node : current_node->GetInDataNodes()) {
        if (visited.find(in_current_node.get()) == visited.end()) {
          GELOGW("Node [%s] has multiple inputs, with input node [%s] not being a post-reduction node.",
                 current_node->GetNamePtr(), in_current_node->GetNamePtr());
          return false;
        }
      }
    }

    node_count += 1UL;
    if (node_count > threshold) {
      GELOGW("The total count of nodes after the reduce node[%s](including the store node) is above the threshold[%zu].",
             reduce_node->GetNamePtr(), threshold);
      return false;
    }
    const auto &asc_current_node = std::dynamic_pointer_cast<ge::AscNode>(current_node);
    GE_ASSERT_NOTNULL(asc_current_node);
    if (ge::ops::IsOps<ge::ascir_op::Store>(asc_current_node) ||
        asc_current_node->attr.api.type == ge::ApiType::kAPITypeBuffer) {
      continue;
    }
    
    if (!ScheduleUtils::IsElewise(asc_current_node)) {
      GELOGW("The node[%s] after the reduce node[%s] is not elewise type.",
             asc_current_node->GetNamePtr(), reduce_node->GetNamePtr());
      return false;
    }

    if (node_count > threshold - 1UL) {
      GELOGW("The count of nodes after the reduce node[%s] is above the threshold[%zu].",
             reduce_node->GetNamePtr(), threshold - 1UL);
      return false;
    }

    for (const auto &next_node : current_node->GetOutAllNodes()) {
      if (visited.find(next_node.get()) == visited.end()) {
        visited.insert(next_node.get());
        node_queue.emplace(next_node);
      }
    }
  }
  return true;
}
}

Status ReducePartitionCaseGenerator::GeneratorGeneralTask(ascir::HintGraph &optimize_graph,
                                                          std::vector<ScheduleTask> &tasks) {
  std::vector<ascir::ImplGraph> optimize_graphs;
  std::vector<std::string> score_funcs;
  GE_CHK_STATUS_RET(GenerateGeneralCase(optimize_graph, optimize_graphs, score_funcs), "GenerateScheduleCases failed");
  score_funcs.resize(optimize_graphs.size());
  for (size_t i = 0U; i < optimize_graphs.size(); ++i) {
    const auto &graph = optimize_graphs[i];
    ScheduleTask task{graph, {}, score_funcs[i], {}, ReduceTemplateType::kCommon};
    GE_CHK_STATUS_RET(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, task.grouped_graphs, node_order_),
                      "Failed to partition graph");
    tasks.emplace_back(std::move(task));
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GeneratorAllLoadTask(ascir::HintGraph &optimize_graph,
                                                          std::vector<ScheduleTask> &tasks) {
  if (!CanReduceFuse(optimize_graph)) {
    return ge::GRAPH_SUCCESS;
  }
  std::vector<ascir::ImplGraph> optimize_graphs;
  std::vector<std::string> score_funcs;
  GE_CHK_STATUS_RET(GenerateAllLoadCase(optimize_graph, optimize_graphs, score_funcs), "GenerateScheduleCases failed");
  score_funcs.resize(optimize_graphs.size());
  for (size_t i = 0U; i < optimize_graphs.size(); ++i) {
    const auto &graph = optimize_graphs[i];
    ScheduleTask task{graph, {}, score_funcs[i], {}, ReduceTemplateType::kAllLoad};
    GE_CHK_STATUS_RET(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, task.grouped_graphs, node_order_),
                      "Failed to partition graph");
    tasks.emplace_back(std::move(task));
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GeneratorRCoreTask(ascir::HintGraph &optimize_graph,
                                                        std::vector<ScheduleTask> &tasks) const {
  std::vector<ScheduleTask> new_tasks;
  for (const auto &task : tasks) {
    if (task.reduce_type != ReduceTemplateType::kCommon) {
      continue;
    }
    std::vector<::ascir::ImplGraph> new_task_grouped_graphs;
    std::map<size_t, std::vector<size_t>> map;
    size_t phase_2_graph_size = 0;
    for (size_t i = 0; i < task.grouped_graphs.size(); i++) {
      GE_ASSERT_TRUE(IsGroupGraphLegal(task.grouped_graphs[i]));
      if (!HasReduce(task.grouped_graphs[i])) {
        ::ascir::ImplGraph graph((task.grouped_graphs[i].GetName() + "_r_multicore").c_str());
        graph.CopyFrom(task.grouped_graphs[i]);
        new_task_grouped_graphs.emplace_back(std::move(graph));
        continue;
      }
      ascir::ImplGraph phase_graph((task.grouped_graphs[i].GetName() + "_r_multicore_phase_graph").c_str());
      phase_graph.CopyFrom(task.grouped_graphs[i]);
      ge::AscNodePtr reduce_node;
      for (auto node : phase_graph.GetAllNodes()) {
        if (ScheduleUtils::IsReduce(node)) {
          reduce_node = node;
          break;
        }
      }
      GE_CHECK_NOTNULL(reduce_node);
      GE_ASSERT_TRUE(IsNotPartitionReduce(reduce_node, NODE_COUNT_AFTER_REDUCE));
      ascir::ImplGraph phase_1_graph((task.grouped_graphs[i].GetName() + "_r_multicore_phase_1_graph").c_str());
      ascir::ImplGraph phase_2_graph((task.grouped_graphs[i].GetName() + "_r_multicore_phase_2_graph").c_str());
      GE_CHK_STATUS_RET(RMulticorePhase2Graph(phase_2_graph, phase_1_graph, phase_graph, reduce_node).Construct());
      ascir::utils::DumpGraph(phase_2_graph, "phase2graph_construct");
      new_task_grouped_graphs.emplace_back(std::move(phase_1_graph));
      new_task_grouped_graphs.emplace_back(std::move(phase_2_graph));
      map[i + phase_2_graph_size] = {i + phase_2_graph_size + 1};
      phase_2_graph_size++;
    }
    if (phase_2_graph_size == 0) {
      continue;
    }
    ScheduleTask new_task{optimize_graph, new_task_grouped_graphs, task.score_func, map, ReduceTemplateType::kRCore};
    new_tasks.push_back(new_task);
  }
  tasks.insert(tasks.end(), new_tasks.begin(), new_tasks.end());
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GeneratorTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks,
                                                   const OptimizerOptions &options) {
  GE_CHK_STATUS_RET(GeneratorGeneralTask(optimize_graph, tasks));
  // inductor 流程后融合场景存在workspace计算问题，暂不增加多核模板
  if (options.graph_type != GraphType::kFusedAscBackend) {
    GE_CHK_STATUS_RET(GeneratorRCoreTask(optimize_graph, tasks));
  }
  GE_CHK_STATUS_RET(GeneratorAllLoadTask(optimize_graph, tasks));
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::Generate([[maybe_unused]] ascir::HintGraph &graph,
                                              [[maybe_unused]] std::vector<ascir::ImplGraph> &graphs,
                                              [[maybe_unused]] std::vector<std::string> &score_functions) {
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GenerateGeneralCase(ascir::HintGraph &graph,
                                                         std::vector<ascir::ImplGraph> &graphs,
                                                         std::vector<std::string> &score_functions) {
  if (!HasReduce(graph)) {
    return ge::GRAPH_SUCCESS;
  }
  ascir::ImplGraph optimize_graph(graph.GetName().c_str());
  optimize_graph.CopyFrom(graph);

  // 以多输出节点为起点遍历，找环路的终点：如有环路则返回终点的列表
  std::vector<std::pair<ge::AscNodePtr, ge::AscNodePtr>> loop_start_end;
  for (auto node : optimize_graph.GetAllNodes()) {
    if (node->GetOutDataNodes().empty()) {
      node_order_.emplace_back(node);
    }
    if (node->GetOutNodes().size() <= 1UL) {
      continue;
    }
    std::vector<ge::AscNodePtr> loop_ends;
    FindNormLoop(node, loop_ends);
    for (const auto &end : loop_ends) {
      loop_start_end.emplace_back(node, end);
    }
  }
  std::sort(loop_start_end.begin(), loop_start_end.end(), [](
    const std::pair<ge::AscNodePtr, ge::AscNodePtr> &lhs, const std::pair<ge::AscNodePtr, ge::AscNodePtr> &rhs) {
    return lhs.second->GetOpDescBarePtr()->GetId() < rhs.second->GetOpDescBarePtr()->GetId();
  });

  // reduce 后融合切分
  GE_CHK_STATUS_RET(ReducePartitionPostFusion(optimize_graph));

  // 按照前面获取的环路起点、终点，进行norm的切分
  GE_CHK_STATUS_RET(PartitionNorm(optimize_graph, loop_start_end));

  // reduce 多引用结构需要切分的，切分开
  GE_CHK_STATUS_RET(ReducePartitionMultipleCitations(optimize_graph));

  if (partition_) {
    std::sort(node_order_.begin(), node_order_.end(), [](const ge::AscNodePtr &lhs, ge::AscNodePtr &rhs) {
      return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
    });

    ascir::utils::DumpGraph(graph, "before_partition");
    ascir::utils::DumpGraph(optimize_graph, "after_partition");
    graphs.emplace_back(optimize_graph);
    score_functions.resize(graphs.size());
  } else {
    node_order_.clear();
    graphs.emplace_back(graph);
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GenerateAllLoadCase(ascir::HintGraph &graph,
                                                         std::vector<ascir::ImplGraph> &graphs,
                                                         const std::vector<std::string> &score_functions) {
  (void)score_functions;
  if (!HasReduce(graph)) {
    return ge::GRAPH_SUCCESS;
  }
  node_order_.clear();
  graphs.emplace_back(graph);
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::ReducePartitionMultipleCitations(ascir::ImplGraph &impl_graph) {
  if (IsGroupGraphLegal(impl_graph)) {
    return ge::GRAPH_SUCCESS;
  }
  std::vector<ge::AscNodePtr> multi_output_nodes;
  for (auto node : impl_graph.GetAllNodes()) {
    if (node->GetOutNodes().size() > 1UL) {
      multi_output_nodes.emplace_back(node);
    }
  }
  std::sort(multi_output_nodes.begin(), multi_output_nodes.end(), [](const ge::AscNodePtr &lhs, ge::AscNodePtr &rhs) {
      return lhs->GetOpDescBarePtr()->GetId() > rhs->GetOpDescBarePtr()->GetId();
  });
  for (auto node : multi_output_nodes) {
    std::set<ge::AscNodePtr> reduce_nodes;
    for (const auto &output_node : node->GetOutNodes()) {
      ge::AscNodePtr out_asc_node = std::dynamic_pointer_cast<ge::AscNode>(output_node);
      if (ge::AscNodePtr reduce_node = nullptr; FindOutputReduce(out_asc_node, reduce_node)) {
        if (!reduce_nodes.empty() && reduce_nodes.find(reduce_node) == reduce_nodes.end()) {
          PartitionByNode(node, out_asc_node, impl_graph);
        }
        reduce_nodes.emplace(reduce_node);
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool ReducePartitionCaseGenerator::FindOutputReduce(const ge::AscNodePtr &node, ge::AscNodePtr &reduce_node) {
  if (ScheduleUtils::IsReduce(node)) {
    reduce_node = node;
    return true;
  }
  bool output_has_reduce = false;
  if (node->GetOutNodes().empty()) {
    return output_has_reduce;
  }
  for (const auto &output_node : node->GetOutNodes()) {
    auto output_asc_node = std::dynamic_pointer_cast<ge::AscNode>(output_node);
    output_has_reduce = output_has_reduce || FindOutputReduce(output_asc_node, reduce_node);
  }
  return output_has_reduce;
}

Status ReducePartitionCaseGenerator::PartitionReduce(ge::AscNodePtr &src_node, ascir::ImplGraph &impl_graph) {
  partition_ = true;
  node_order_.emplace_back(src_node);
  ge::ascir_op::Workspace workspace_pre((src_node->GetName() + "_Workspace").c_str());
  ge::ascir_op::Workspace workspace_post((src_node->GetName() + "_Workspace").c_str());
  ge::ascir_op::Load load((src_node->GetName() + "_Load").c_str());
  ge::ascir_op::Store store((src_node->GetName() + "_Store").c_str());
  auto workspace_pre_node = impl_graph.AddNode(workspace_pre);
  auto workspace_post_node = impl_graph.AddNode(workspace_post);
  auto load_node = impl_graph.AddNode(load);
  auto store_node = impl_graph.AddNode(store);
  GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(src_node, load_node));
  GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(src_node, store_node));
  GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(store_node, workspace_pre_node));
  GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(load_node, workspace_post_node));
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                     REPORT_INNER_ERR_MSG("E18888", "out data anchor is null, node:%s.", src_node->GetName().c_str());
                     return ge::GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s",
                            src_node->GetName().c_str());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      auto dst_node = peer_in_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(dst_node, "peer node is null, src node: %s", src_node->GetNamePtr());
      // remove src->dst
      GE_CHK_STATUS_RET(ge::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                   dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(load_node->GetOutAnchor(out_anchor->GetIdx()),
                                                dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
    }
  }
  // add src->store->workspace_pre_node
  GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(src_node->GetOutAnchor(0UL), store_node->GetInAnchor(0UL)));
  GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL), workspace_pre_node->GetInAnchor(0UL)));
  // add workspace_post_node->load->dst
  GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(workspace_post_node->GetOutAnchor(0UL), load_node->GetInAnchor(0UL)));
	  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::ReducePartitionPostFusion(ascir::ImplGraph &impl_graph) {
  for (auto node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsReduce(node)) {
      if (IsNotPartitionReduce(node, NODE_COUNT_AFTER_REDUCE)) {
        continue;
      }
      GE_CHK_STATUS_RET(PartitionReduce(node, impl_graph));
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::PartitionByNode(ge::AscNodePtr &src_node, ge::AscNodePtr &dst_node,
                                                     ascir::ImplGraph &impl_graph) {
  partition_ = true;
  node_order_.emplace_back(src_node);
  if (ScheduleUtils::IsLoad(src_node)) {
    return PartitionLoad(src_node, dst_node, impl_graph);
  }
  if (ge::ops::IsOps<ge::ascir_op::Scalar>(src_node)) {
    return PartitionScalar(src_node, dst_node, impl_graph);
  };

  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                     REPORT_INNER_ERR_MSG("E18888", "out data anchor is null, node:%s.", src_node->GetName().c_str());
                     return ge::GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s",
                            src_node->GetName().c_str());
    ge::ascir_op::Workspace workspace_pre(GetNewNodeName(src_node, dst_node, "Workspace", out_anchor->GetIdx()).c_str());
    ge::ascir_op::Workspace workspace_post(GetNewNodeName(src_node, dst_node, "Workspace", out_anchor->GetIdx()).c_str());
    ge::ascir_op::Load load(GetNewNodeName(src_node, dst_node, "Load", out_anchor->GetIdx()).c_str());
    ge::ascir_op::Store store(GetNewNodeName(src_node, dst_node, "Store", out_anchor->GetIdx()).c_str());
    auto workspace_pre_node = impl_graph.AddNode(workspace_pre);
    auto workspace_post_node = impl_graph.AddNode(workspace_post);
    auto load_node = impl_graph.AddNode(load);
    auto store_node = impl_graph.AddNode(store);
    GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(src_node, load_node));
    GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(src_node, store_node));
    GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(store_node, workspace_pre_node));
    GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(load_node, workspace_post_node));
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHK_BOOL_EXEC(peer_in_anchor->GetOwnerNodeBarePtr() != nullptr,
                       REPORT_INNER_ERR_MSG("E18888", "Peer in node:%s is null", src_node->GetName().c_str());
                       return ge::GRAPH_FAILED, "Peer in node:%s is null", src_node->GetName().c_str());
      if (peer_in_anchor->GetOwnerNodeBarePtr()->GetName() == dst_node->GetName()) {
        // remove src->dst
        GE_CHK_STATUS_RET(ge::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(load_node->GetOutAnchor(0UL),
                                                  dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
    // add src->store->workspace_pre_node
    GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                              store_node->GetInAnchor(0UL)));
    GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL),
                                              workspace_pre_node->GetInAnchor(0UL)));
    // add workspace_post_node->load->dst
    GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(workspace_post_node->GetOutAnchor(0UL),
                                              load_node->GetInAnchor(0UL)));
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::PartitionLoad(ge::AscNodePtr &src_node, ge::AscNodePtr &dst_node,
                                                   ascir::ImplGraph &impl_graph) {
  auto load_input_node = src_node->GetInNodes().at(0UL);
  auto load_input_asc_node = std::dynamic_pointer_cast<ge::AscNode>(load_input_node);
  GE_ASSERT_TRUE(ge::ops::IsOps<ge::ascir_op::Data>(load_input_asc_node) || ge::ops::IsOps<ge::ascir_op::Workspace>(load_input_asc_node));
  ge::ascir_op::Load load(("copy_from_" + src_node->GetName()).c_str());

  ge::AscNodePtr new_load_input_node;
  if (ge::ops::IsOps<ge::ascir_op::Data>(load_input_asc_node)) {
    ge::ascir_op::Data data(("copy_from_" + load_input_asc_node->GetName()).c_str());
    new_load_input_node = impl_graph.AddNode(data);
  } else {
    ge::ascir_op::Workspace workspace(("copy_from_" + load_input_asc_node->GetName()).c_str());
    new_load_input_node = impl_graph.AddNode(workspace);
  }
  auto load_node = impl_graph.AddNode(load);
  DoCopyAscNodeTensorAttr(load_input_asc_node, new_load_input_node);
  DoCopyAscNodeTensorAttr(src_node, load_node);
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                     REPORT_INNER_ERR_MSG("E18888", "out data anchor is null, node:%s.", src_node->GetName().c_str());
                     return ge::GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s", src_node->GetName().c_str());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHK_BOOL_EXEC(peer_in_anchor->GetOwnerNodeBarePtr() != nullptr,
                       REPORT_INNER_ERR_MSG("E18888", "Peer in node:%s is null", src_node->GetName().c_str());
                       return ge::GRAPH_FAILED, "Peer in node:%s is null", src_node->GetName().c_str());
      if (peer_in_anchor->GetOwnerNodeBarePtr()->GetName() == dst_node->GetName()) {
        // remove load->dst
        GE_CHK_STATUS_RET(ge::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        // add new_load_input->new_load->dst
        GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(new_load_input_node->GetOutAnchor(0UL),
                                                  load_node->GetInAnchor(0UL)));
        GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(load_node->GetOutAnchor(0UL),
                                                  dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        return ge::GRAPH_SUCCESS;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::PartitionScalar(ge::AscNodePtr &src_node, ge::AscNodePtr &dst_node,
                                                     ascir::ImplGraph &impl_graph) {
  ge::ascir_op::Scalar scalar(("copy_from_" + src_node->GetName()).c_str());
  auto scalar_node = impl_graph.AddNode(scalar);
  DoCopyAscNodeTensorAttr(src_node, scalar_node);
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                     REPORT_INNER_ERR_MSG("E18888", "out data anchor is null, node:%s.", src_node->GetName().c_str());
                     return ge::GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s", src_node->GetName().c_str());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHK_BOOL_EXEC(peer_in_anchor->GetOwnerNodeBarePtr() != nullptr,
                       REPORT_INNER_ERR_MSG("E18888", "Peer in node:%s is null", src_node->GetName().c_str());
                       return ge::GRAPH_FAILED, "Peer in node:%s is null", src_node->GetName().c_str());
      if (peer_in_anchor->GetOwnerNodeBarePtr()->GetName() == dst_node->GetName()) {
        // remove src->dst
        GE_CHK_STATUS_RET(ge::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        // add new_scalar->dst
        GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(scalar_node->GetOutAnchor(0UL),
                                                  dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        return ge::GRAPH_SUCCESS;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool ReducePartitionCaseGenerator::IsInputNodePartitioned(const std::shared_ptr<ge::Node>& start,
  const std::shared_ptr<ge::Node>& node) {
  // 向前找 start 节点，如找到则未经过切分，到根节点未找到则已切分或不需要切分
  if (node == start) {
    return false;
  }
  bool partitioned = true;
  if (node->GetInNodes().empty()) {
    return partitioned;
  }
  for (const auto &in_node : node->GetInNodes()) {
    partitioned = partitioned && IsInputNodePartitioned(start, in_node);
  }
  return partitioned;
}

Status ReducePartitionCaseGenerator::FindNormLoop(const ge::AscNodePtr &start, std::vector<ge::AscNodePtr> &ends) {
  std::set<ge::NodePtr> visited{start};
  std::list<ge::NodePtr> next_nodes{start};
  while (!next_nodes.empty()) {
    const auto node = next_nodes.front();
    next_nodes.pop_front();
    for (auto &out_node : node->GetOutDataNodes()) {
      if (visited.find(out_node) == visited.cend()) {
        next_nodes.emplace_back(out_node);
        visited.emplace(out_node);
      } else {
        auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(out_node);
        ends.emplace_back(asc_node);
      }
    }
  }
  ends.erase(
    std::remove_if(ends.begin(), ends.end(),
      [start, this](const ge::AscNodePtr &end) { return !IsNorm(start, end); }), ends.end()
    );
  return ge::GRAPH_SUCCESS;
}

void ReducePartitionCaseGenerator::FindAllPath(const ge::AscNodePtr& start, const ge::AscNodePtr& end,
                                               std::vector<ge::AscNodePtr> &path,
                                               std::vector<std::vector<ge::AscNodePtr>> &all_paths) {
  // 需保证图为有向无环图
  path.emplace_back(start);
  if (start == end) {
    all_paths.emplace_back(path);
  }
  for (const auto &out_node : start->GetOutNodes()) {
    const auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(out_node);
    FindAllPath(asc_node, end, path, all_paths);
  }
  path.pop_back();
}

bool ReducePartitionCaseGenerator::IsNorm(const ge::AscNodePtr &start, const ge::AscNodePtr &end) {
  std::vector<ge::AscNodePtr> path;
  std::vector<std::vector<ge::AscNodePtr>> all_paths;
  FindAllPath(start, end, path, all_paths);
  bool is_norm = false;
  std::set<ge::AscNodePtr> end_in_nodes;
  for (const auto &path_temp : all_paths) {
    for (const auto &node : path_temp) {
      if (ScheduleUtils::IsReduce(node)) {
        is_norm =  true;
        break;
      }
    }
    end_in_nodes.insert(path_temp.at(path_temp.size() - TWO));
  }
  return is_norm && end_in_nodes.size() > 1;
}

Status ReducePartitionCaseGenerator::PartitionNorm(ascir::ImplGraph &impl_graph, std::vector<std::pair<ge::AscNodePtr,
                                                   ge::AscNodePtr>> &loop_start_end) {
  for (auto loop : loop_start_end) {
    for (auto &in_node : loop.second->GetInNodes()) {
      if(IsInputNodePartitioned(loop.first, in_node)) {
        continue;
      }
      ge::AscNodePtr src_node = std::dynamic_pointer_cast<ge::AscNode>(in_node);
      GE_CHK_STATUS_RET(PartitionByNode(src_node, loop.second, impl_graph));
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool ReducePartitionCaseGenerator::HasReduce(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsReduce(node)) {
      return true;
    }
  }
  return false;
}

// 全载模板只支持reduce AR ARA
bool ReducePartitionCaseGenerator::CanReduceFuse(const ascir::ImplGraph &impl_graph) {
  std::vector<ascir::SizeExpr> temp_strides;
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (!ScheduleUtils::IsReduce(node)) {
      continue;
    }
    std::vector<ascir::SizeExpr> input_repeats = node->inputs[0].attr.repeats;
    std::vector<ascir::SizeExpr> output_repeats = node->outputs[0].attr.repeats;
    GE_ASSERT_TRUE(input_repeats.size() == output_repeats.size());
    if (output_repeats.empty() || (output_repeats.size() > kMaxFullLoadAxisSize)) {
      return false;
    }

    if (ge::SymbolicUtils::StaticCheckEq(input_repeats[0], output_repeats[0]) != ge::TriBool::kTrue) {
      return false;
    }
  }

  return true;
}

bool ReducePartitionCaseGenerator::IsGroupGraphLegal(const ascir::ImplGraph &impl_graph) {
  int reduce_count = 0;
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsReduce(node)) {
      reduce_count += 1;
    }
  }
  return reduce_count <= 1;
}

Status RMulticorePhase2Graph::Construct() {
  GE_ASSERT_TRUE(reducers.find(reduce_node->GetType()) != reducers.end());
  ReduceType phase2graph_reduce = reducers.find(reduce_node->GetType())->
                                  second((phase2graph.GetName() + "_" + reduce_node->GetName() + "_reduce").c_str());
  std::visit([](auto&& reduce_op) {
    reduce_op.attr.sched.axis = {0, 1};
  }, phase2graph_reduce);
  GE_CHK_STATUS_RET(CompletePhaseGraph(phase2graph_reduce));
  GE_CHK_STATUS_RET(CreateVarAxis());
  auto workspace_node = phase2graph.FindNode((phase2graph.GetName() + "_workspace").c_str());
  GE_ASSERT_NOTNULL(workspace_node);
  workspace_node->attr.sched.axis = {0, 1};
  GE_CHK_STATUS_RET(CompleteNodeAttr(workspace_node, true, reduce_node->outputs[0].attr.dtype));
  auto load_node = phase2graph.FindNode((phase2graph.GetName() + "_load").c_str());
  GE_ASSERT_NOTNULL(load_node);
  load_node->attr.sched.axis = {0, 1};
  GE_CHK_STATUS_RET(CompleteNodeAttr(load_node, true, reduce_node->outputs[0].attr.dtype));
  auto reduce_node_parse2graph = phase2graph.FindNode((phase2graph.GetName() + "_" + reduce_node->GetName() + "_reduce").c_str());
  GE_ASSERT_NOTNULL(reduce_node_parse2graph);
  std::set<ge::NodePtr> visited{reduce_node_parse2graph};
  std::list<ge::NodePtr> next_nodes{reduce_node_parse2graph};
  while (!next_nodes.empty()) {
    const auto node = next_nodes.front();
    next_nodes.pop_front();
    auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(node);
    asc_node->attr.sched.axis = {0, 1};
    GE_CHK_STATUS_RET(CompleteNodeAttr(asc_node, false, asc_node->outputs[0].attr.dtype));
    for (auto &out_node : node->GetOutDataNodes()) {
      if (visited.find(out_node) == visited.cend()) {
        next_nodes.emplace_back(out_node);
        visited.emplace(out_node);
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::CreateVarAxis() {
  // 创建符号：A轴符号：s1，R轴符号：s2
  auto  compute_graph = ge::AscGraphUtils::GetComputeGraph(phase2graph);
  GE_ASSERT_NOTNULL(compute_graph);
  auto attr = compute_graph->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
  GE_ASSERT_NOTNULL(attr);
  attr->axis.clear();
  attr->size_vars.clear();
  Rm_org_size = phase2graph.CreateSizeVar("Rm_org_size");
  A_org_size = phase2graph.CreateSizeVar("A_org_size");
  // 创建轴：a，r
  phase2graph.CreateAxis("Rm", Rm_org_size);
  phase2graph.CreateAxis("A", A_org_size);
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::CompleteNodeAttr(ge::AscNodePtr &node, bool before_reduce,
                                               const ge::AscTensorDataType& data_type) {
  node->outputs[0].attr.dtype = data_type;
  node->outputs[0].attr.axis = {0, 1};
  if (before_reduce) {
    node->outputs[0].attr.strides = {A_org_size, ge::ops::One};
    node->outputs[0].attr.repeats = {Rm_org_size, A_org_size};
  } else {
    node->outputs[0].attr.strides = {ge::ops::Zero, ge::ops::One};
    node->outputs[0].attr.repeats = {ge::ops::One, A_org_size};
  }
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::CompletePhaseGraph(ReduceType &phase2graph_reduce) {
  std::vector<ge::AscNodePtr> node_order;
  GE_ASSERT_GRAPH_SUCCESS(PartitionByReduce(phase_graph, phase2graph_reduce, node_order));
  GE_ASSERT_GRAPH_SUCCESS(SetNodeOrder(node_order));
  std::vector<::ascir::ImplGraph> sub_optimize_graphs;
  GE_ASSERT_GRAPH_SUCCESS(ScheduleGroupGraphPartitioner::PartitionByConnectivity(phase_graph, sub_optimize_graphs, node_order));
  GE_ASSERT_EQ(sub_optimize_graphs.size(), 2UL);
  phase1graph.CopyFrom(sub_optimize_graphs[0]);
  phase2graph.CopyFrom(sub_optimize_graphs[1]);
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::PartitionByReduce(ascir::ImplGraph &impl_graph,
                                                ReduceType &phase2graph_reduce,
                                                std::vector<ge::AscNodePtr> &node_order) {
  ge::ascir_op::Workspace workspace_pre((phase2graph.GetName() + "_workspace").c_str());
  ge::ascir_op::Workspace workspace_post((phase2graph.GetName() + "_workspace").c_str());
  ge::ascir_op::Load load((phase2graph.GetName() + "_load").c_str());
  ge::ascir_op::Store store((phase1graph.GetName() + "Store").c_str());
  ge::AscNodePtr new_reduce_node;
  std::visit([&new_reduce_node, &impl_graph](auto&& reduce_op) {
    new_reduce_node = impl_graph.AddNode(reduce_op);
  }, phase2graph_reduce);
  auto workspace_pre_node = impl_graph.AddNode(workspace_pre);
  node_order.emplace_back(workspace_pre_node);
  auto workspace_post_node = impl_graph.AddNode(workspace_post);
  auto load_node = impl_graph.AddNode(load);
  GE_ASSERT_NOTNULL(load_node);
  auto store_node = impl_graph.AddNode(store);
  GE_ASSERT_NOTNULL(store_node);
  GE_ASSERT_GRAPH_SUCCESS(DoCopyAscNodeTensorAttr(reduce_node, new_reduce_node));
  GE_ASSERT_GRAPH_SUCCESS(DoCopyAscNodeTensorAttr(reduce_node, store_node));
  GE_ASSERT_GRAPH_SUCCESS(DoCopyWorkspaceTensorAttr(reduce_node, workspace_pre_node));
  GE_ASSERT_GRAPH_SUCCESS(DoCopyWorkspaceTensorAttr(load_node, workspace_post_node));
  for (const auto &reduce_out_anchor : reduce_node->GetAllOutDataAnchors()) {
    GE_ASSERT_NOTNULL(reduce_out_anchor);
    for (const auto &peer_in_anchor : reduce_out_anchor->GetPeerInDataAnchors()) {
      GE_ASSERT_NOTNULL(peer_in_anchor);
      auto reduce_out_node = peer_in_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(reduce_out_node);
      GE_CHK_STATUS_RET(ge::GraphUtils::RemoveEdge(reduce_node->GetOutAnchor(reduce_out_anchor->GetIdx()),
                                                   reduce_out_node->GetInAnchor(peer_in_anchor->GetIdx())));
      GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(new_reduce_node->GetOutAnchor(reduce_out_anchor->GetIdx()),
                                                reduce_out_node->GetInAnchor(peer_in_anchor->GetIdx())));
    }
  }
  // add reduce->store->workspace_pre_node
  GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(reduce_node->GetOutAnchor(0UL),
                                            store_node->GetInAnchor(0UL)));
  GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL),
                                            workspace_pre_node->GetInAnchor(0UL)));
  // add workspace_post_node->load->new reduce
  GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(workspace_post_node->GetOutAnchor(0UL),
                                            load_node->GetInAnchor(0UL)));
  GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(load_node->GetOutAnchor(0UL),
                                            new_reduce_node->GetInAnchor(0UL)));
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::SetNodeOrder (std::vector<ge::AscNodePtr> &node_order) {
  auto new_reduce_node = phase_graph.FindNode((phase2graph.GetName() + "_" + reduce_node->GetName() + "_reduce").c_str());
  GE_ASSERT_NOTNULL(new_reduce_node);
  std::set<ge::NodePtr> visited{new_reduce_node};
  std::list<ge::NodePtr> next_nodes{new_reduce_node};
  while (!next_nodes.empty()) {
    const auto node = next_nodes.front();
    GE_ASSERT_NOTNULL(node);
    next_nodes.pop_front();
    if (node->GetOutDataNodes().empty()) {
      node_order.emplace_back(std::dynamic_pointer_cast<ge::AscNode>(node));
    }
    for (auto &out_node : node->GetOutDataNodes()) {
      if (visited.find(out_node) == visited.cend()) {
        next_nodes.emplace_back(out_node);
        visited.emplace(out_node);
      }
    }
  }
  return ge::SUCCESS;
}
}