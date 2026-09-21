/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/cube_schedule_case_generator.h"
#include <queue>
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/utils/graph_utils.h"
#include "ascir/meta/ascir_utils.h"
#include "ascir/meta/ascir_ops_utils.h"
#include "optimize/schedule_utils.h"
#include "platform/platform_factory.h"
#include "util/mem_utils.h"
#include "base/err_msg.h"

namespace optimize {
namespace {
const ge::Expression kSymbolZero = ge::Symbol(0);
const ge::Expression kSymbolOne = ge::Symbol(1);

std::string GetNewNodeName(const ge::AscNodePtr &src_node, const std::string &type) {
  return src_node->GetName() + "_" + type;
}

Status DoCopyAscNodeTensorAttr(const ge::AscNodePtr &cube_node, ge::AscNodePtr &store_node) {
  GE_ASSERT_NOTNULL(cube_node);
  GE_ASSERT_NOTNULL(store_node);
  auto op_desc = store_node->GetOpDesc();
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<ge::AscNodeAttr>();
  auto src_asc_node_attr = cube_node->GetOpDesc()->GetOrCreateAttrsGroup<ge::AscNodeAttr>();
  if ((src_asc_node_attr != nullptr) && (dst_asc_node_attr != nullptr)) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
  }
  GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
  auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::AscTensorAttr>();
  GE_CHECK_NOTNULL(tensor_attr_group);
  const auto &output_attr = cube_node->outputs[0].attr;
  tensor_attr_group->dtype = static_cast<ge::DataType>(output_attr.dtype);
  tensor_attr_group->axis = output_attr.axis;
  tensor_attr_group->repeats = output_attr.repeats;
  tensor_attr_group->strides = output_attr.strides;
  return ge::SUCCESS;
}

Status DoCopyAscNodeTensorAttr(const ge::AscNodePtr &cube_node, const ge::AscNodePtr &cube_next_node,
                               ge::AscNodePtr &load_node) {
  GE_ASSERT_NOTNULL(cube_next_node);
  GE_ASSERT_NOTNULL(load_node);
  GE_ASSERT_NOTNULL(cube_node);
  auto op_desc = load_node->GetOpDesc();
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<ge::AscNodeAttr>();
  auto src_asc_node_attr = cube_next_node->GetOpDesc()->GetOrCreateAttrsGroup<ge::AscNodeAttr>();
  if ((src_asc_node_attr != nullptr) && (dst_asc_node_attr != nullptr)) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
    if (src_asc_node_attr->ir_attr) {
      dst_asc_node_attr->ir_attr = src_asc_node_attr->ir_attr->Clone();
    }
  }
  GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
  auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::AscTensorAttr>();
  GE_CHECK_NOTNULL(tensor_attr_group);
  const auto &cube_next_output_attr = cube_next_node->outputs[0].attr;
  tensor_attr_group->axis = cube_next_output_attr.axis;

  const auto &cube_output_attr = cube_node->outputs[0].attr;
  tensor_attr_group->repeats = cube_output_attr.repeats;
  tensor_attr_group->strides = cube_output_attr.strides;
  tensor_attr_group->dtype = cube_output_attr.dtype;

  for (size_t i = 0U; i < cube_next_output_attr.axis.size(); i++) {
    auto it = std::find(cube_output_attr.axis.begin(), cube_output_attr.axis.end(), cube_next_output_attr.axis[i]);
    if (it == cube_output_attr.axis.end()) {
      // 给 tensor 补齐轴
      tensor_attr_group->repeats.insert(tensor_attr_group->repeats.begin() + i, kSymbolOne);
      tensor_attr_group->strides.insert(tensor_attr_group->strides.begin() + i, kSymbolZero);
    }
  }

  return ge::SUCCESS;
}

Status DoCopyWorkspaceTensorAttr(const ge::AscNodePtr &src_node, ge::AscNodePtr &workspace_node) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(workspace_node);
  GE_ASSERT_TRUE(src_node->outputs().size() > 0UL);
  GE_ASSERT_TRUE(workspace_node->outputs().size() > 0UL);
  workspace_node->outputs[0].attr.dtype = src_node->outputs[0].attr.dtype;
  return ge::SUCCESS;
}

Status UpdateHintGraphAttr(const ::ascir::ImplGraph &graph) {
  const auto &compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph, "compute_graph is null");
  const auto compute_graph_attr = compute_graph->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
  GE_CHECK_NOTNULL(compute_graph_attr, "compute_graph_attr is null");

  for (auto node : graph.GetAllNodes()) {
    if (!ScheduleUtils::IsCube(node)) {
      continue;
    }
    const auto &cube_output_attr = node->outputs[0].attr;

    std::vector<ge::AxisPtr> new_axis;
    for (size_t i = 0U; i < compute_graph_attr->axis.size(); i++) {
      auto it = std::find(cube_output_attr.axis.begin(), cube_output_attr.axis.end(), compute_graph_attr->axis[i]->id);
      if (it != cube_output_attr.axis.end()) {
        std::shared_ptr<ge::Axis> axis = ge::MakeShared<ge::Axis>();
        GE_CHECK_NOTNULL(axis, "create axis failed");
        axis->id = compute_graph_attr->axis[i]->id;
        axis->name = compute_graph_attr->axis[i]->name;
        axis->type = compute_graph_attr->axis[i]->type;
        axis->size = compute_graph_attr->axis[i]->size;
        axis->from = compute_graph_attr->axis[i]->from;
        new_axis.push_back(std::move(axis));
      }
    }
    compute_graph_attr->axis = new_axis;
  }
  return ge::SUCCESS;
}

bool HasCastBrc(const ascir::ImplGraph &graph) {
  for (auto node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (!ge::ops::IsOps<ge::ascir_op::Cast>(node)) {
      continue;
    }
    for (const auto &out_node : node->GetOutNodes()) {
      GE_ASSERT_NOTNULL(out_node);
      if (ge::ops::IsOps<ge::ascir_op::Broadcast>(out_node)) {
        return true;
      }
    }
  }
  return false;
}

bool IsCubeFixpip(const ge::AscNodePtr &cube_node) {
  for (auto out_node : cube_node->GetOutNodes()) {
    GE_ASSERT_NOTNULL(out_node);
    auto out_asc_node = std::dynamic_pointer_cast<ge::AscNode>(out_node);
    GE_ASSERT_NOTNULL(out_asc_node);
    if (!ScheduleUtils::IsStore(out_asc_node)) {
      return false;
    }
  }
  return true;
}

Status GetPrioritySequence(const ge::AscGraph &graph, std::unordered_set<ge::Node *> &priority_sequences,
                           std::unordered_set<ge::Node *> &store_sequences) {
  std::unordered_set<const ge::Node*> visited;
  std::queue<ge::NodePtr> node_queue;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetName().find("Cube_Load_") == std::string::npos) {
      continue;
    }
    visited.insert(node.get());
    for (const auto &cube_output_node : node->GetOutNodes()) {
      GE_ASSERT_NOTNULL(cube_output_node);
      if (ge::ops::IsOps<ge::ascir_op::Store>(cube_output_node)) {
        store_sequences.insert(cube_output_node.get());
        continue;
      }
      node_queue.emplace(cube_output_node);
      visited.insert(cube_output_node.get());
    }
    while (!node_queue.empty()) {
      const auto current_node = node_queue.front();
      node_queue.pop();
      priority_sequences.insert(current_node.get());
      for (const auto &next_node : current_node->GetOutAllNodes()) {
        if (visited.find(next_node.get()) == visited.end()) {
          visited.insert(next_node.get());
          node_queue.emplace(next_node);
        }
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status TopoSortByCubePriority(ge::AscGraph &graph) {
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  std::unordered_set<ge::Node *> priority_sequences;
  std::unordered_set<ge::Node *> store_sequences;
  GE_ASSERT_GRAPH_SUCCESS(GetPrioritySequence(graph, priority_sequences, store_sequences),
                          "Get priority sequence failed.");
  if (priority_sequences.empty()) {
    return ge::GRAPH_SUCCESS;
  }

  auto vec_node = *priority_sequences.begin();
  GE_ASSERT_NOTNULL(vec_node);
  GE_ASSERT_NOTNULL(vec_node->GetOpDesc());
  auto vec_node_id = vec_node->GetOpDesc()->GetId();
  int64_t store_id_min = std::numeric_limits<int64_t>::max();;
  ge::Node *store_node_id_min = nullptr;
  for (auto store_node : store_sequences) {
    GE_ASSERT_NOTNULL(store_node);
    GE_ASSERT_NOTNULL(store_node->GetOpDesc());
    auto store_id = store_node->GetOpDesc()->GetId();
    if (store_id < store_id_min) {
      store_id_min = store_id;
      store_node_id_min = store_node;
    }
  }
  if (store_id_min < vec_node_id) {
    GE_ASSERT_NOTNULL(store_node_id_min);
    GE_ASSERT_NOTNULL(store_node_id_min->GetOpDesc());
    vec_node->GetOpDesc()->SetId(store_id_min);
    store_node_id_min->GetOpDesc()->SetId(vec_node_id);
  }

  const auto func = [&priority_sequences](const ge::NodePtr &node1, const ge::NodePtr &node2) -> bool {
    bool is_node1_in_priority_seq = priority_sequences.find(node1.get()) != priority_sequences.end();
    bool is_node2_in_priority_seq = priority_sequences.find(node2.get()) != priority_sequences.end();
    if (is_node1_in_priority_seq && !is_node2_in_priority_seq) {
      return true;
    } else {
      return node1->GetOpDescBarePtr()->GetId() < node2->GetOpDescBarePtr()->GetId();
    }
  };

  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  compute_graph->TopologicalSorting(func);
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  return ge::SUCCESS;
}
bool HasBroadCastNode(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->attr.api.compute_type == ge::ComputeType::kComputeBroadcast &&
        !ScheduleUtils::IsScalarBrc(node)) {
      return true;
    }
  }
  return false;
}
}  // namespace

Status CubeFusionCaseGenerator::GenerateGeneralCase(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs) {
  ascir::ImplGraph optimize_graph(graph.GetName().c_str());
  optimize_graph.CopyFrom(graph);

  for (const auto &node : optimize_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetOutDataNodes().empty()) {
      node_order_.emplace_back(node);
    }
    if (!ScheduleUtils::IsCube(node)) {
      continue;
    }
    if (IsCubeFixpip(node)) {
      GELOGI("graph %s is fixpip, generate fixpip task.", optimize_graph.GetName().c_str());
      break;
    }
    node_order_.emplace_back(node);
    partition_ = true;
    bool first = true;
    ge::AscNodePtr workspace_pre_node;
    ge::AscNodePtr workspace_post_node;
    ge::AscNodePtr load_node;
    ge::AscNodePtr store_node;
    ge::ascir_op::Workspace workspace_pre(GetNewNodeName(node,  "Workspace").c_str());
    ge::ascir_op::Workspace workspace_post(GetNewNodeName(node, "Workspace").c_str());
    ge::ascir_op::Load load(("Cube_Load_" + GetNewNodeName(node, "Load")).c_str());
    ge::ascir_op::Store store(GetNewNodeName(node, "Store").c_str());
    for (const auto &cube_output_node : node->GetOutNodes()) {
      GE_CHECK_NOTNULL(cube_output_node);
      ge::AscNodePtr cube_node = std::dynamic_pointer_cast<ge::AscNode>(cube_output_node);
      GE_ASSERT_NOTNULL(cube_node);
      if (first) {
        workspace_pre_node = optimize_graph.AddNode(workspace_pre);
        GE_ASSERT_NOTNULL(workspace_pre_node);
        workspace_post_node = optimize_graph.AddNode(workspace_post);
        GE_ASSERT_NOTNULL(workspace_post_node);
        load_node = optimize_graph.AddNode(load);
        GE_ASSERT_NOTNULL(load_node);
        store_node = optimize_graph.AddNode(store);
        GE_ASSERT_NOTNULL(store_node);
        GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(node, cube_node, load_node));
        GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(node, store_node));
        GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(store_node, workspace_pre_node));
        GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(load_node, workspace_post_node));
      }
      for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
        GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                         REPORT_INNER_ERR_MSG("E18888", "out data anchor is null, node:%s.", node->GetName().c_str());
                         return ge::GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s",
                                node->GetName().c_str());
        for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
          GE_CHECK_NOTNULL(peer_in_anchor);
          GE_CHK_BOOL_EXEC(peer_in_anchor->GetOwnerNodeBarePtr() != nullptr,
                           REPORT_INNER_ERR_MSG("E18888", "Peer in node:%s is null", node->GetName().c_str());
                           return ge::GRAPH_FAILED, "Peer in node:%s is null", node->GetName().c_str());
          if (peer_in_anchor->GetOwnerNodeBarePtr()->GetName() == cube_node->GetName()) {
            // remove src->dst
            GE_CHK_STATUS_RET(ge::GraphUtils::RemoveEdge(node->GetOutAnchor(out_anchor->GetIdx()),
                                                         cube_node->GetInAnchor(peer_in_anchor->GetIdx())));
            // load->dst
            GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(load_node->GetOutAnchor(out_anchor->GetIdx()),
                                                      cube_node->GetInAnchor(peer_in_anchor->GetIdx())));
            break; // 这里加的break可能会导致add两个输入都来自cube时切图只切了一半
          }
        }
      }
      if (first) {
        // add src->store->workspace_pre_node
        GE_CHK_STATUS_RET(
            ge::GraphUtils::AddEdge(node->GetOutAnchor(0UL), store_node->GetInAnchor(0UL)));
        GE_CHK_STATUS_RET(
            ge::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL), workspace_pre_node->GetInAnchor(0UL)));
        // add workspace_post_node->load
        GE_CHK_STATUS_RET(
            ge::GraphUtils::AddEdge(workspace_post_node->GetOutAnchor(0UL), load_node->GetInAnchor(0UL)));
      }
      first = false;
    }
  }
  ascir::utils::DumpGraph(graph, "before_partition");
  ascir::utils::DumpGraph(optimize_graph, "after_partition");
  graphs.emplace_back(optimize_graph);
  return ge::GRAPH_SUCCESS;
}

Status CubeFusionCaseGenerator::GenNddmaNode(const ge::AscNodePtr &node_load, const ge::AscNodePtr &node_brc,
                                             ge::AscGraph &new_case) {
  GE_CHECK_NOTNULL(node_load);
  GE_CHECK_NOTNULL(node_load->GetOpDesc());
  GE_CHECK_NOTNULL(node_brc);
  // 继承自Load节点，把Load节点的offset信息和是否brc缓存信息同步继承过来
  node_load->GetOpDesc()->SetType("Nddma");
  node_load->attr.type = "Nddma";
  node_load->attr.sched = node_brc->attr.sched;
  node_load->outputs[0].attr = node_brc->outputs[0].attr;
  GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(new_case, std::dynamic_pointer_cast<ge::AscNode>(node_brc),
                                              node_load->GetOutDataAnchor(0)));
  return ge::SUCCESS;
}

ge::Status CubeFusionCaseGenerator::SwapCastBrcAndGenNddma(const ge::AscNodePtr &node_cast,
                                                           const ge::AscNodePtr &node_load, ge::AscGraph &new_case) {
  // 针对cast输出多引用的场景不做处理
  if (node_cast->GetOutNodesSize() != 1UL) {
    GELOGD("Node %s with single output and multiple refs, do not support gen nddma.", node_cast->GetNamePtr());
    return ge::UNSUPPORTED;
  }
  // 判断是否为load-cast-brc场景
  auto cast_out_anchor = node_cast->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(cast_out_anchor);
  auto next_in_anchor = cast_out_anchor->GetPeerInDataAnchors().at(0);
  GE_CHECK_NOTNULL(next_in_anchor);
  const auto &next_node = std::dynamic_pointer_cast<ge::AscNode>(next_in_anchor->GetOwnerNode());
  GE_CHECK_NOTNULL(next_node);
  if (!ge::ops::IsOps<ge::ascir_op::Broadcast>(next_node)) {
    GELOGD("The subgraph is not load-cast-brc, do not gen nddma.");
    return ge::UNSUPPORTED;
  }
  node_cast->attr.sched = next_node->attr.sched;
  node_cast->outputs[0].attr = next_node->outputs[0].attr;
  next_node->outputs[0].attr.dtype = node_load->outputs[0].attr.dtype;
  auto load_out_anchor = node_load->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(load_out_anchor);
  auto cast_in_anchor = load_out_anchor->GetPeerInDataAnchors().at(0);
  GE_CHECK_NOTNULL(cast_in_anchor);
  auto brc_out_anchor = next_node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(brc_out_anchor);
  // 将cast-->brc替换为load-->brc
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::ReplaceEdgeSrc(cast_out_anchor, next_in_anchor, load_out_anchor));
  // 将load-->cast替换为brc-->cast
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::ReplaceEdgeSrc(load_out_anchor, cast_in_anchor, brc_out_anchor));
  // 将brc-->others替换为cast-->others
  for (const auto &peer_in_anchor : brc_out_anchor->GetPeerInDataAnchors()) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    // 跳过cast_in_anchor这条边
    if (peer_in_anchor == cast_in_anchor) {
      continue;
    }
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::ReplaceEdgeSrc(brc_out_anchor, peer_in_anchor, cast_out_anchor));
  }

  GE_ASSERT_SUCCESS(GenNddmaNode(node_load, std::dynamic_pointer_cast<ge::AscNode>(next_node), new_case));
  return ge::SUCCESS;
}

Status CubeFusionCaseGenerator::GeneratorUbTask(const std::vector<::ascir::ImplGraph> &grouped_graphs,
                                                ScheduleTask &ub_task, std::vector<ScheduleTask> &tasks) {
  std::vector<::ascir::ImplGraph> tmp_grouped_graphs;
  for (auto &grouped_graph : grouped_graphs) {
    ascir::ImplGraph optimize_graph((grouped_graph.GetName() + "_ub").c_str());
    optimize_graph.CopyFrom(grouped_graph);
    if (ScheduleUtils::HasComputeType(optimize_graph, ge::ComputeType::kComputeCube)) {
      tmp_grouped_graphs.emplace_back(optimize_graph);
      continue;
    }
    for (auto node : optimize_graph.GetAllNodes()) {
      GE_CHECK_NOTNULL(node);
      if (!ge::ops::IsOps<ge::ascir_op::Load>(node)) {
        continue;
      }
      if (node->GetOutAllNodes().size() > 1UL) {
        GELOGD("Node %s with single output and multiple refs, do not support nddma.", node->GetNamePtr());
        continue;
      }
      auto load_out_anchor = node->GetOutDataAnchor(0);
      GE_CHECK_NOTNULL(load_out_anchor);
      auto peer_in_anchor = load_out_anchor->GetPeerInDataAnchors().at(0);
      GE_CHECK_NOTNULL(peer_in_anchor);
      const auto &out_node = std::dynamic_pointer_cast<ge::AscNode>(peer_in_anchor->GetOwnerNode());
      GE_CHECK_NOTNULL(out_node);
      if (ge::ops::IsOps<ge::ascir_op::Broadcast>(out_node)) {
        GE_ASSERT_SUCCESS(GenNddmaNode(node, std::dynamic_pointer_cast<ge::AscNode>(out_node),
                                       optimize_graph), "Generator nddma node failed.");
      }
      if (ge::ops::IsOps<ge::ascir_op::Cast>(out_node)){
        auto ret = SwapCastBrcAndGenNddma(std::dynamic_pointer_cast<ge::AscNode>(out_node), node,
                                                 optimize_graph);
        if (ret == ge::UNSUPPORTED) {
          if (HasCastBrc(optimize_graph)) {
            GELOGW("The graph %s not support generating ub task.", grouped_graph.GetName().c_str());
            return ge::GRAPH_SUCCESS;
          }
          continue;
        }
        if (ret != ge::GRAPH_SUCCESS) {
          GELOGE(ret, "Swap cast and brc, generator nddma node failed.");
          return ret;
        }
      }
    }
    tmp_grouped_graphs.emplace_back(optimize_graph);
    if (HasBroadCastNode(optimize_graph)) {
      GELOGW("The graph %s still contains broadcast nodes and not support generating ub task.",
             grouped_graph.GetName().c_str());
      return ge::GRAPH_SUCCESS;
    }
  }
  for (const auto &tmp_graph : tmp_grouped_graphs) {
    ub_task.grouped_graphs.emplace_back(tmp_graph);
  }
  tasks.push_back(std::move(ub_task));
  return ge::GRAPH_SUCCESS;
}

void MoveCubeGraphsToEnd(std::vector<::ascir::ImplGraph> &grouped_graphs) {
  (void)std::stable_partition(
    grouped_graphs.begin(),
    grouped_graphs.end(),
    [](const ::ascir::ImplGraph& graph) {
        // 返回true表示保留在前面，false表示移动到后面
        return !ScheduleUtils::HasComputeType(graph, ge::ComputeType::kComputeCube);
    }
  );
  // it现在指向第一个Cube类型元素的位置
}

Status CubeFusionCaseGenerator::GeneratorTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks,
                                              const OptimizerOptions &options) {
  (void)options;
  std::vector<ascir::ImplGraph> optimize_graphs;
  std::vector<std::string> score_funcs;
  if (!ScheduleUtils::HasComputeType(optimize_graph, ge::ComputeType::kComputeCube)) {
    return ge::GRAPH_SUCCESS;
  }
  GE_CHK_STATUS_RET(GenerateGeneralCase(optimize_graph, optimize_graphs), "GenerateScheduleCases failed");
  score_funcs.resize(optimize_graphs.size());
  if (partition_) {
    std::sort(node_order_.begin(), node_order_.end(), [](const ge::AscNodePtr &lhs, ge::AscNodePtr &rhs) {
      return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
    });
  } else {
    node_order_.clear();
  }
  for (size_t i = 0U; i < optimize_graphs.size(); ++i) {
    const auto &graph = optimize_graphs[i];
    ScheduleTask task{graph, {}, score_funcs[i], {},
                      ReduceTemplateType::kDefault, ascir::CubeTemplateType::kFixpip};
    GE_CHK_STATUS_RET(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, task.grouped_graphs,
                      node_order_), "Failed to partition graph");
    if (task.grouped_graphs.size() > 1U) {
      task.cube_type = ascir::CubeTemplateType::kCommon;
      MoveCubeGraphsToEnd(task.grouped_graphs);
      for (auto grouped_graph : task.grouped_graphs) {
        if (ScheduleUtils::HasComputeType(grouped_graph, ge::ComputeType::kComputeCube)) {
          GE_CHK_STATUS_RET(UpdateHintGraphAttr(grouped_graph), "UpdateHintGraphAttr failed");
        } else {
          GE_ASSERT_GRAPH_SUCCESS(TopoSortByCubePriority(grouped_graph), "Failed to topo sort by cube priority.");
        }
      }
      ScheduleTask ub_task{graph, {}, score_funcs[i], {},
                           ReduceTemplateType::kDefault, ascir::CubeTemplateType::kUBFuse};
      GE_ASSERT_SUCCESS(GeneratorUbTask(task.grouped_graphs, ub_task, tasks), "Generator ub task failed.");
    }
    tasks.emplace_back(std::move(task));
  }
  return ge::GRAPH_SUCCESS;
}

Status CubeFusionCaseGenerator::Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                         std::vector<std::string> &score_functions) {
  (void)graph;
  (void)graphs;
  (void)score_functions;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optimize
