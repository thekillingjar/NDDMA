/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_schedule_case_generator.h"
#include <queue>
#include <unordered_map>
#include "graph/utils/graph_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"
#include "ascir_ops.h"
#include "schedule_utils.h"


namespace optimize {
namespace {
constexpr size_t kExpectedTransposeNodeNum = 1UL;      // 当前支持Ascgraph中含单个Transpose，后续考虑多个的情况。
constexpr int32_t transposeNoNeedUBConvertSize = 512;  // 以512Byte作为是否需要消除Transpose的阈值
}

std::vector<ge::AscNodePtr> TransposeFusionCaseGenerator::FindTransposeNodes(const ascir::HintGraph &owner_graph) {
  std::vector<ge::AscNodePtr> transpose_nodes;
  for (const auto &node : owner_graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::Transpose>(node)) {
      transpose_nodes.emplace_back(node);
    }
  }
  return transpose_nodes;
}

Status TransposeFusionCaseGenerator::TransposeNodeInputsAndOutputsCheck(const ge::AscNodePtr &transpose_node) const {
  const auto &transpose_in_data_nodes = transpose_node->GetInDataNodes();
  GE_ASSERT_TRUE(transpose_in_data_nodes.size() == 1UL, "%zu nodes links to transpose node:%s",
                 transpose_in_data_nodes.size(), transpose_node->GetNamePtr());
  return ge::SUCCESS;
}

void TransposeFusionCaseGenerator::UpdateAxisByPath(::ascir::ImplGraph &owner_graph, const ge::NodePtr &input_node,
                                                    std::set<ge::Node *> &visited_nodes,
                                                    const std::vector<int64_t> &reordered_axis,
                                                    const std::vector<int64_t> &reordered_sched_axis) const {
  std::queue<ge::NodePtr> path_nodes;
  path_nodes.emplace(input_node);
  while (!path_nodes.empty()) {
    auto top = path_nodes.front();
    path_nodes.pop();
    visited_nodes.emplace(top.get());
    ::ascir::NodeView input_view = std::dynamic_pointer_cast<ge::AscNode>(top);
    if (!ScheduleUtils::IsBuffer(input_view)) {
      owner_graph.ApplyTensorAxisReorder(input_view, reordered_axis);
      owner_graph.ApplySchedAxisReorder(input_view, reordered_sched_axis);
    }
    // 向上遍历。 暂时不考虑transpose之上的节点有额外向下的分支，如果有额外分支，则不能将transpose消除。
    for (size_t idx = 0UL; idx < top->GetInDataNodesSize(); ++idx) {
      if (visited_nodes.count(top->GetInDataNodes().at(idx).get()) == 0UL) {
        path_nodes.push(top->GetInDataNodes().at(idx));
      }
    }
  }
}

void TransposeFusionCaseGenerator::UpdateAxis(ascir::HintGraph &graph, const ge::AscNodePtr &transpose_node) const {
  std::vector<std::pair<int64_t, int64_t>> transpose_info;
  const auto &reordered_axis = transpose_node->outputs[0].attr.axis;
  const auto &reordered_sched_axis = transpose_node->attr.sched.axis;
  std::set<ge::Node *> unused_visited_nodes;
  UpdateAxisByPath(graph, transpose_node, unused_visited_nodes, reordered_axis, reordered_sched_axis);
}

Status TransposeFusionCaseGenerator::TransposeConvertProcess(ascir::HintGraph &graph,
                                                             const ge::AscNodePtr &transpose_node) const {
  GE_CHK_STATUS_RET(TransposeNodeInputsAndOutputsCheck(transpose_node), "TransposeNode Check failed");
  UpdateAxis(graph, transpose_node);

  auto owner_compute_graph = transpose_node->GetOwnerComputeGraph();
  GE_CHK_STATUS_RET(owner_compute_graph->RemoveNode(transpose_node), "Failed to remote node: %s",
                    transpose_node->GetNamePtr());

  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  ascir::utils::DumpGraph(graph, "AfterConvertTranspose");
  return ge::SUCCESS;
}

Status TransposeFusionCaseGenerator::Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs, std::vector<std::string> &score_functions) {
  /*
  场景1： 尾轴转置， 需要UB重排，Transpose节点保留；
  场景2：非尾轴转置，尾轴大于等于512Byte，不需要UB重排，Transpose删除，刷新load/store表示；
  场景3：非尾轴转置，尾轴小于等于512Byte，需要UB重排，Transpose节点保留；
  同时提供两套模板，后续根据打分机制来选取
  */

  const auto transpose_nodes = FindTransposeNodes(graph);
  if (transpose_nodes.size() != kExpectedTransposeNodeNum) {
    GELOGI("Transpose node num = %zu, not equal to 1, skip", transpose_nodes.size());
    return ge::SUCCESS;
  }

  //生成场景1、3的模板，Transpose保留模板（不改图）
  graphs.emplace_back(graph);

  //生成场景2的模板，Transpose消除模板（改图）
  // 复制图用于生成改图模板
  ascir::ImplGraph optimized_graph((graph.GetName() + "_group_transpose").c_str());
  optimized_graph.CopyFrom(graph);

  auto transpose_node = FindTransposeNodes(optimized_graph).front();
  GE_CHK_STATUS_RET(TransposeConvertProcess(optimized_graph, transpose_node), "TransposeConvertProcess failed");
  graphs.emplace_back(optimized_graph);

  if (graphs.size() > 1U) {
    transpose_node = FindTransposeNodes(graphs[0]).front();
    score_functions.resize(2U);
    GE_CHK_STATUS_RET(GenerateScoreFuncForUbReorder(graph, transpose_node, score_functions[0]),
                      "Failed to generate score func");
  }
  return ge::SUCCESS;
}

Status TransposeFusionCaseGenerator::GenerateScoreFuncForUbReorder(const ascir::HintGraph &graph,
                                                                  const ge::AscNodePtr &transpose_node,
                                                                  std::string &score_func) {
  return TransposeScoreFunctionGenerator(graph, transpose_node).Generate(score_func);
}

TransposeScoreFunctionGenerator::TransposeScoreFunctionGenerator(const ascir::HintGraph &graph,
                                                                 ge::AscNodePtr transpose_node)
  : graph_(&graph), transpose_node_(std::move(transpose_node)){}

Status TransposeScoreFunctionGenerator::ParseRepeat() {
  const auto last_idx = transpose_node_->inputs[0].attr.axis.size() - 1;
  repeat_ = transpose_node_->inputs[0].attr.repeats[last_idx];
  return ge::SUCCESS;
}

Status TransposeScoreFunctionGenerator::Generate(std::string &score_func) {
  GE_CHK_STATUS_RET(ParseRepeat());
  ss_ << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {" << std::endl;
  int32_t score = 0;

  GE_CHK_STATUS_RET(GetScoreByExpr(score));
  GenerateReturnValue(score);
  score_func = ss_.str();
  return ge::SUCCESS;
}

void TransposeScoreFunctionGenerator::GenerateReturnValue(const int32_t score) {
  ss_ << "  return " << score << ";" << std::endl;
  ss_ << "}" << std::endl;
}

Status TransposeScoreFunctionGenerator::GetScoreByExpr(int32_t &score) const {
  // 使用表达式值获取分数值
  const auto last_idx = transpose_node_->inputs[0].attr.axis.size() - 1;
  const auto &input_tail_axis = transpose_node_->inputs[0].attr.axis[last_idx];
  const auto &output_tail_axis = transpose_node_->outputs[0].attr.axis[last_idx];

  if (input_tail_axis != output_tail_axis) {
    score = 1;
    return ge::SUCCESS;
  }
  // 非尾轴转置需要根据尾轴大小确定分数
  int32_t dim = -1;
  GE_ASSERT_TRUE(repeat_.GetHint(dim), "Failed to get int value, expr = %s", ge::SymbolicUtils::ToString(repeat_).c_str());
  const auto limited_size = transposeNoNeedUBConvertSize / GetSizeByDataType(transpose_node_->inputs[0].attr.dtype);
  score = dim < limited_size ? 1 : -1;
  return ge::SUCCESS;
}
}  // namespace optimize