/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nddma_template.h"
#include <stack>
#include "ascir_utils.h"
#include "graph_utils.h"
#include "un_alignment_strategy.h"
#include "tensor_layout_utils.h"
#include "autoschedule/alignment_handler.h"
#include "platform/common/base_alignment_strategy.h"

namespace optimize {
namespace {
constexpr uint32_t kAlignBytes = 32U;

bool IsNddma(const ge::AscNodePtr &node) {
  return ScheduleUtils::IsLoad(node) && node->attr.type == "Nddma";
}
}  // namespace

std::string NddmaTemplate::GenName(const std::string &general_case_name) {
  return general_case_name + "_nddma";
}

ge::Status NddmaTemplate::ReAlignVectorizedStrides(const ge::AscNodePtr &node) {
  auto &output_attr = node->outputs[0].attr;
  const auto dtype_size = GetSizeByDataType(output_attr.dtype);
  GE_ASSERT_TRUE(dtype_size > 0, "Node [%s] output tensor dtype is invalid.", node->GetNamePtr());
  const auto &output_vec_axis = output_attr.vectorized_axis;
  if (output_vec_axis.empty()) {
    GELOGD("Vectorized axis is empty, no need to be realigned.");
    return ge::SUCCESS;
  }
  ge::Expression size_product = ge::sym::kSymbolOne;
  for (auto vec_axis_id = static_cast<int64_t>(output_vec_axis.size() - 1); vec_axis_id >= 0; vec_axis_id--) {
    ge::Expression &vectorized_stride = output_attr.vectorized_strides.at(vec_axis_id);
    // 向量化轴的stride为0则不做处理
    if (ge::SymbolicUtils::StaticCheckEq(vectorized_stride, ge::sym::kSymbolZero) == ge::TriBool::kTrue) {
      continue;
    }
    // 获取当前轴的repeat
    const auto axis = output_vec_axis.at(vec_axis_id);
    auto axis_tensor_iter = std::find(output_attr.axis.begin(), output_attr.axis.end(), axis);
    GE_ASSERT_TRUE(axis_tensor_iter != output_attr.axis.end(),
           "Can not find vectorized axis [%ld] in [%s]'s output tensor.", axis, node->GetNamePtr());
    const int64_t axis_index = std::distance(output_attr.axis.begin(), axis_tensor_iter);
    const auto &repeat = output_attr.repeats.at(axis_index);
    // 对次尾轴做对齐
    if (vec_axis_id == static_cast<int64_t>(output_vec_axis.size() - 2)) { // 2 表示次尾轴偏移量
      vectorized_stride = ge::sym::Align(vectorized_stride, kAlignBytes / dtype_size);
      size_product = ge::sym::Mul(repeat, vectorized_stride);
      continue;
    }
    // 其他vectorized_stride不为0的向量化轴用前一根stride非0的轴的repeat*vectorized_stride来刷新
    vectorized_stride = size_product;
    size_product = size_product * repeat;
  }
  return ge::SUCCESS;
}

/**
 * Brc与load或NDDMA合并成新的NDDMA节点，具体逻辑如下：
 * 1. 先将brc输出attr中的repeats属性按前序节点的axis顺序进行重排序
 * 2. 将前序节点的type改为nddma、并将nddma输出attr中的repeats和向量化轴strides设为brc输出attr中对应的值
 * 3. 将nddma的输出连到brc的输出并删除brc节点
 */
Status NddmaTemplate::GenNddmaNode(const ge::AscNodePtr &node_pre, const ge::AscNodePtr &node_brc,
                                   ge::AscGraph &new_case, const bool is_need_realignment) {
  GE_CHECK_NOTNULL(node_pre);
  GE_CHECK_NOTNULL(node_pre->GetOpDesc());
  GE_CHECK_NOTNULL(node_brc);
  // 如果前序节点已经是带transpose属性的nddma，需要先对repeats进行排序
  GE_ASSERT_SUCCESS(ReorderRepeats(node_pre, node_brc));
  // 继承自Load/Nddma节点，把Load/Nddma节点的offset信息和是否brc缓存信息同步继承过来
  node_pre->GetOpDesc()->SetType("Nddma");
  node_pre->attr.type = "Nddma";
  const auto brc_output_attr = node_brc->outputs[0].attr;
  GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(new_case, std::dynamic_pointer_cast<ge::AscNode>(node_brc),
                                              node_pre->GetOutDataAnchor(0)));
  node_pre->outputs[0].attr.repeats = brc_output_attr.repeats;
  node_pre->outputs[0].attr.vectorized_strides = brc_output_attr.vectorized_strides;
  if (!is_need_realignment) {
    return ge::SUCCESS;
  }
  ReAlignVectorizedStrides(node_pre);
  return ge::SUCCESS;
}

Status NddmaTemplate::AddTransposeNodeAfter(ge::AscGraph &graph, const ge::AscNodePtr &node,
                                            ge::AscNodePtr &new_transpose_node,
                                            const ge::AscNodePtr &old_transpose_node){
  GE_ASSERT_NOTNULL(node);
  const auto &old_transpose_node_info = old_transpose_node->outputs[0].attr;
  const std::string node_name = node->GetName() + "_transpose_tmp";
  ge::ascir_op::Transpose transpose_op(node_name.c_str());
  new_transpose_node = graph.AddNode(transpose_op);
  GE_ASSERT_NOTNULL(new_transpose_node);
  // 插入到node后的transpose的dtype需要和load的dtype一致
  new_transpose_node->outputs[0].attr.dtype = node->outputs[0].attr.dtype;

  // repeats数据量和load保持一致，需要按transpose重排并重新计算strides，axis、Vector axis应与transpose后的保持一致
  new_transpose_node->outputs[0].attr.axis = node->outputs[0].attr.axis;
  new_transpose_node->outputs[0].attr.repeats = node->outputs[0].attr.repeats;
  GE_ASSERT_SUCCESS(ReorderRepeats(old_transpose_node, new_transpose_node));
  GE_ASSERT_SUCCESS(ScheduleUtils::RecalculateStridesFromRepeats(new_transpose_node->outputs[0].attr.repeats,
                                                              new_transpose_node->outputs[0].attr.strides));
  new_transpose_node->attr.sched.axis = old_transpose_node->attr.sched.axis;
  new_transpose_node->outputs[0].attr.axis = old_transpose_node_info.axis;
  new_transpose_node->outputs[0].attr.vectorized_axis = old_transpose_node_info.vectorized_axis;

  const auto out_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);
  for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_SUCCESS(ge::GraphUtils::ReplaceEdgeSrc(out_anchor, in_anchor, new_transpose_node->GetOutDataAnchor(0)));
  }
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(out_anchor, new_transpose_node->GetInDataAnchor(0)));
  return ge::SUCCESS;
}

Status NddmaTemplate::MergeLoadAndTranspose(const ge::AscNodePtr &load_node, ge::AscGraph& new_case) {
  auto load_out_anchor = load_node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(load_out_anchor);
  auto peer_in_anchor = load_out_anchor->GetPeerInDataAnchors().at(0);
  GE_CHECK_NOTNULL(peer_in_anchor);
  const auto &out_node = std::dynamic_pointer_cast<ge::AscNode>(peer_in_anchor->GetOwnerNode());
  GE_CHECK_NOTNULL(out_node);
  load_node->outputs[0].attr.vectorized_strides = out_node->outputs[0].attr.vectorized_strides;
  load_node->outputs[0].attr.vectorized_axis = out_node->outputs[0].attr.vectorized_axis;
  load_node->GetOpDesc()->SetType("Nddma");
  load_node->attr.type = "Nddma";
  // 删除transpose节点
  GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(new_case, out_node, load_node->GetOutDataAnchor(0)));
  return ge::SUCCESS;
}

Status NddmaTemplate::TransposeToNddmaNode(const ge::AscNodePtr &transpose_node, ge::AscGraph &new_case) {
  const auto &transpose_info = transpose_node->outputs[0].attr;
  std::stack<ge::AscNodePtr> node_stack;
  std::set<ge::AscNodePtr> visited_nodes;
  std::vector<ge::AscNodePtr> load_nodes;
  // 从transpose_node开始向前遍历，由前端保证transpose之上的节点不会有额外向下的分支
  node_stack.push(transpose_node);
  visited_nodes.insert(transpose_node);
  while (!node_stack.empty()) {
    auto current_node = node_stack.top();
    node_stack.pop();
    // 如果当前节点是Load节点，则不再向前遍历，在其后插入transpose节点
    if (ge::ops::IsOps<ge::ascir_op::Load>(current_node)) {
      ge::AscNodePtr new_transpose_node = nullptr;
      AddTransposeNodeAfter(new_case, current_node, new_transpose_node, transpose_node);
      load_nodes.push_back(current_node);
      continue;
    }
    // 更新非Scalar节点的相关属性、当前节点的axis、Vector axis应为转置后的axis、repeats对应需要重排、strides需要重新计算
    if (!ge::ops::IsOps<ge::ascir_op::Scalar>(current_node)) {
      GE_ASSERT_SUCCESS(ReorderRepeats(transpose_node, current_node));
      GE_ASSERT_SUCCESS(ScheduleUtils::RecalculateStridesFromRepeats(current_node->outputs[0].attr.repeats,
                                                                  current_node->outputs[0].attr.strides));
      current_node->attr.sched.axis = transpose_node->attr.sched.axis;
      current_node->outputs[0].attr.axis = transpose_info.axis;
      current_node->outputs[0].attr.vectorized_axis = transpose_info.vectorized_axis;
    }
    for (const auto &in_data_anchor : current_node->GetAllInDataAnchors()) {
      auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }
      auto input_node = std::dynamic_pointer_cast<ge::AscNode>(peer_out_anchor->GetOwnerNode());
      if (input_node == nullptr || visited_nodes.find(input_node) != visited_nodes.end()) {
        continue;
      }
      visited_nodes.insert(input_node);
      node_stack.push(input_node);
    }
  }
  // 删除旧的transpose节点
  auto in_node = std::dynamic_pointer_cast<ge::AscNode>(transpose_node->GetInDataNodes().at(0));
  if (in_node != nullptr) {
    GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(new_case, transpose_node, in_node->GetOutDataAnchor(0)));
  }
  GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(new_case));
  GE_ASSERT_SUCCESS(autoschedule::AlignmentHandler::AlignVectorizedStrides(new_case));
  // 将load + transpose节点合并成nddma节点
  for (const auto &load_node : load_nodes) {
    GE_ASSERT_SUCCESS(MergeLoadAndTranspose(load_node, new_case));
  }
  return ge::SUCCESS;
}

/**
 * 生成NDDMA模版，具体逻辑如下：
 * 1. 遍历图找到Transpose节点，将transpose前移并随路更新路过节点的属性，最终与load或nddma节点融合成新的nddma节点
 * 2. 遍历图找到load/nddma节点，判断是否为输出多引用，若不是则继续执行步骤3
 * 3. 判断load/nddma节点的输出是否为brc节点，若是则将load/nddma和brc继续合并为nddma节点
 */
ge::Status NddmaTemplate::Generate([[maybe_unused]] const ge::AscGraph &origin_graph, 
                                   [[maybe_unused]] const ge::AscGraph &based_case,
                                   ge::AscGraph &new_case) {
  bool is_nddma_generated = false;
  for  (const auto &node : new_case.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if (ge::ops::IsOps<ge::ascir_op::Transpose>(node)) {
      GE_ASSERT_SUCCESS(TransposeToNddmaNode(node, new_case));
      is_nddma_generated = true;
    }
  }
  for (const auto &node : new_case.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if (!ge::ops::IsOps<ge::ascir_op::Load>(node) && !ge::ops::IsOps<ge::ascir_op::Nddma>(node)) {
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
    bool is_nddma_generated_cur = false;
    if (ge::ops::IsOps<ge::ascir_op::Broadcast>(out_node)) {
      GE_ASSERT_SUCCESS(GenNddmaNode(node, std::dynamic_pointer_cast<ge::AscNode>(out_node), new_case));
      is_nddma_generated_cur = true;
    }
    if (ge::ops::IsOps<ge::ascir_op::Cast>(out_node) &&
        SwapCastBrcAndGenNddma(std::dynamic_pointer_cast<ge::AscNode>(out_node), node, new_case) == ge::SUCCESS) {
      is_nddma_generated_cur = true;
    }
    is_nddma_generated = is_nddma_generated || is_nddma_generated_cur;
    DiscontinuityInfo info;
    GE_ASSERT_SUCCESS(TensorLayoutUtils::AnalyzeLoadDiscontinuity(node->outputs[0].attr, info),
                      "Failed to analyze discontinuity info for node:[%s].", node->GetNamePtr());
    bool need_align_at_repeat1 = info.has_multiple_discontinuities && info.is_tail_axis_discontinuous;
    if (!is_nddma_generated_cur && (IsLoadNeedAlign(node) || need_align_at_repeat1)) {
      GE_ASSERT_SUCCESS(GenLoadToGenNddmaNode(node));
    }
  }
  if (!is_nddma_generated) {
    GELOGD("No nddma template generated.");
    return ge::FAILED;
  }
  GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(new_case));
  return ge::SUCCESS;
}

bool NddmaTemplate::IsSecondaryTailAxisAligned(const ge::AscNodePtr &node) {
  const auto &node_output_attr = node->outputs[0].attr;
  const auto &tail_axis_repeat = node_output_attr.repeats.back();
  const auto &tail_axis_vectorized_stride = node_output_attr.vectorized_strides.back();
  if (node_output_attr.vectorized_strides.size() <= 1UL) {
    GELOGD("Node [%s] has no or just one vectorized axis.", node->GetNamePtr());
    return false;
  }
  const auto &penultimate_axis_vectorized_stride =
    node_output_attr.vectorized_strides.at(node_output_attr.vectorized_strides.size() - 2);
  if (ge::SymbolicUtils::StaticCheckEq(penultimate_axis_vectorized_stride, ge::sym::kSymbolZero) == ge::TriBool::kTrue
    || ge::SymbolicUtils::StaticCheckEq(tail_axis_vectorized_stride, ge::sym::kSymbolZero) == ge::TriBool::kTrue) {
    GELOGD("Node [%s] penultimate or tail axis vectorized stride is zero, skip.", node->GetNamePtr());
    return false;
  }
  if (ge::SymbolicUtils::StaticCheckEq(penultimate_axis_vectorized_stride,
    ge::sym::Mul(tail_axis_repeat, tail_axis_vectorized_stride)) != ge::TriBool::kTrue) {
    GELOGD("Node [%s] tail axis has been aligned.", node->GetNamePtr());
    return true;
  }
  return false;
}

/**
 * 参照node_src对node_dst的repeats属性重排
 */
ge::Status NddmaTemplate::ReorderRepeats(const ge::AscNodePtr &node_src, const ge::AscNodePtr &node_dst){
  // 获取dst节点的axis和repeats
  const auto &dst_axis = node_dst->outputs[0].attr.axis;
  const auto &dst_repeats = node_dst->outputs[0].attr.repeats;
  
  // 获取src节点的axis
  const auto &src_axis = node_src->outputs[0].attr.axis;
  
  // 创建axis映射关系：从dst的axis到src的axis的索引映射
  std::map<size_t, size_t> axis_mapping;
  for (size_t i = 0; i < dst_axis.size(); ++i) {
    const auto dst_axis_id = dst_axis[i];
    auto it = std::find(src_axis.begin(), src_axis.end(), dst_axis_id);
    if (it != src_axis.end()) {
      size_t src_index = std::distance(src_axis.begin(), it);
      axis_mapping[i] = src_index;
    }
  }
  
  // 根据axis映射关系重新排列repeats
  std::vector<ge::Expression> new_dst_repeats(dst_repeats.size());
  for (size_t i = 0; i < dst_repeats.size(); ++i) {
    auto it = axis_mapping.find(i);
    if (it != axis_mapping.end()) {
      size_t load_index = it->second;
      if (load_index < new_dst_repeats.size()) {
        new_dst_repeats[load_index] = dst_repeats[i];
      }
    } else {
      new_dst_repeats[i] = dst_repeats[i];
    }
  }
  
  // 更新dst节点的repeats
  node_dst->outputs[0].attr.repeats = new_dst_repeats;
  
  return ge::SUCCESS;
}

ge::Status NddmaTemplate::SwapCastBrcAndGenNddma(const ge::AscNodePtr &node_cast, const ge::AscNodePtr &node_load,
  ge::AscGraph &new_case) {
  // 针对cast输出多引用的场景不做处理
  if (node_cast->GetOutNodesSize() != 1UL) {
    GELOGD("Node %s with single output and multiple refs, do not support gen nddma.", node_cast->GetNamePtr());
    return ge::FAILED;
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
    return ge::FAILED;
  }
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
  // 判断原brc次尾轴是否按cast之后的dtype对齐，若是则后续nddma的向量化轴需要按cast之前的dtype对齐
  // 当前针对reduce场景仅判断brc次尾轴是否对齐，针对中间轴对齐场景待后续补充
  const bool is_need_realignment = IsSecondaryTailAxisAligned(next_node);
  next_node->outputs[0].attr.dtype = node_load->outputs[0].attr.dtype;
  node_cast->outputs[0].attr.repeats = next_node->outputs[0].attr.repeats;
  node_cast->outputs[0].attr.strides = next_node->outputs[0].attr.strides;
  node_cast->outputs[0].attr.vectorized_axis = next_node->outputs[0].attr.vectorized_axis;
  node_cast->outputs[0].attr.vectorized_strides = next_node->outputs[0].attr.vectorized_strides;
  GE_ASSERT_SUCCESS(GenNddmaNode(node_load, std::dynamic_pointer_cast<ge::AscNode>(next_node), new_case,
                                 is_need_realignment));
  return ge::SUCCESS;
}

bool NddmaTemplate::NeedDropBasedCase(const ge::AscGraph &origin_graph, [[maybe_unused]] const ge::AscGraph &based_case,
                                      [[maybe_unused]] const ge::AscGraph &new_case) {
  if (ScheduleUtils::HasComputeType(origin_graph, ge::ComputeType::kComputeTranspose)) {
    return  true;
  }
  return false;
}

bool IsValidNode(const ge::AscNodePtr &node) {
  return (ScheduleUtils::IsIOBuffer(node) || ScheduleUtils::IsBuffer(node) || ScheduleUtils::IsElewise(node) ||
          ScheduleUtils::IsBroadcast(node) || ScheduleUtils::IsLoad(node) || ScheduleUtils::IsStore(node)) &&
         node->attr.type != "Split";
}

bool IsValidDataType(const ge::AscNodePtr &node) {
  const auto dsize = ge::GetSizeByDataType(node->outputs[0].attr.dtype);
  const int b8 = 1;
  const int b16 = 2;
  return dsize == b8 || dsize == b16;
}

bool IsTailBroadcastNddmaNode(const ge::AscNodePtr &node) {
  GE_WARN_ASSERT(!node->outputs().empty());
  GE_WARN_ASSERT(!node->outputs[0].attr.vectorized_axis.empty());
  uint32_t vec_last_axis_pos_in_axis = std::find(node->outputs[0].attr.axis.begin(), node->outputs[0].attr.axis.end(),
                                                 node->outputs[0].attr.vectorized_axis.back()) -
                                       node->outputs[0].attr.axis.begin();
  GE_WARN_ASSERT(vec_last_axis_pos_in_axis < static_cast<long>(node->outputs[0].attr.strides.size()));
  const auto axis_size = node->outputs[0].attr.vectorized_axis.size();
  // strides是原模板load的输出stride，vectorized_strides是原模板brc的输出vectorized_strides
  if (node->outputs[0].attr.strides[vec_last_axis_pos_in_axis] == 0 &&
      node->outputs[0].attr.vectorized_strides[axis_size - 1] == 1) {
    return true;
  }
  return false;
}

bool IsStaticGraph(const ge::AscGraph &origin_graph){
  for (const auto &var : origin_graph.GetAllSizeVar()) {
    if (!var->expr.IsConstExpr()) {
      return false;
    }
  }
  return true;
}

std::string GetStaticScoreFunc(const ge::AscNodePtr &nddma_node, std::stringstream &ss){
  // 默认打分为0，ATT根据性能公式选择模板
  int32_t score = 0;
  const int low_score = -1;
  // case1: B8,B16类型时，若尾轴对齐，低分
  if (IsValidDataType(nddma_node) && ScheduleUtils::IsTailAxisAlignedBy(nddma_node)) {
    GELOGD("Nddma Node [%s] has B8/B16 data type with aligned tail axis, assigning low score.",
           nddma_node->GetNamePtr());
    score = low_score;
  }
  // case2: 尾轴brc时，如果尾轴大于4KB，低分
  const uint32_t large_tail_size = 4096;
  uint32_t tail_size = 0;
  if (IsTailBroadcastNddmaNode(nddma_node) && ScheduleUtils::GetTailAxisDataSize(nddma_node, tail_size) &&
      tail_size > large_tail_size) {
    GELOGD("Nddma Node [%s] is tail broadcast and with large tail axis, assigning low score.",
           nddma_node->GetNamePtr());
    score = low_score;
  }
  GELOGD("Nddma Node [%s]: Assigning score [%d].", nddma_node->GetNamePtr(), score);
  ss << "  return " << score << ";" << std::endl << "}" << std::endl;
  return ss.str();
}

std::string GetDynamicScoreFunc(const ge::AscGraph &nddma_graph, const ge::AscNodePtr &nddma_node,
                                std::stringstream &ss) {
  GELOGD("Start to get score func for dynamic Nddma Graph");
  std::vector<std::pair<ge::Expression, ge::Expression>> replacements;
  for (const auto &size_var : nddma_graph.GetAllSizeVar()) {
    if (!size_var->expr.IsConstExpr()) {
      replacements.emplace_back(size_var->expr,
                                ge::Symbol((std::string("tiling_data.") + size_var->expr.Str().get()).c_str()));
    }
  }
  const auto &output_attr = nddma_node->outputs[0].attr;
  const auto dsize = ge::GetSizeByDataType(output_attr.dtype);
  const auto dim_expr = output_attr.repeats.back();
  ge::Expression last_dim_size = ge::Symbol(dsize);
  last_dim_size = last_dim_size * dim_expr;
  ss << "  const auto tail_size = static_cast<int64_t>(" << last_dim_size.Replace(replacements).Str().get() << ");"
     << std::endl;
  // case1
  if (IsValidDataType(nddma_node)) {
    GELOGD("Nddma Node [%s] has B8/B16 data type, check alignment.", nddma_node->GetNamePtr());
    ss << "  if (tail_size % 32 == 0) { return -1; }" << std::endl;
  }
  // case 2
  if (IsTailBroadcastNddmaNode(nddma_node)) {
    GELOGD("Nddma Node [%s] is tail broadcast, check tail size.", nddma_node->GetNamePtr());
    ss << "  if (tail_size > 4096) { return -1; }" << std::endl;
  }
  ss << "  return 0;" << std::endl << "}" << std::endl;
  return ss.str();
}

std::string NddmaTemplate::GetScoreFunc(const ge::AscGraph &origin_graph, const ge::AscGraph &nddma_graph) {
  nddma_graph.GetName().c_str();
  GELOGD("Start to get score func for Nddma Graph [%s]", nddma_graph.GetName().c_str());
  ge::AscNodePtr nddma_node;
  uint32_t nddma_node_cnt = 0;
  // 打分函数仅在 纯elementwise+brc && 单个nddma节点 场景下生效
  // 从原图判断节点类型，排除transpose转成nddma的场景
  for (const auto &node : origin_graph.GetAllNodes()) {
    if (!IsValidNode(node)) {
      GELOGD("Graph [%s]: Not elewise + broadcast graph, assigning default score.", origin_graph.GetName().c_str());
      return "";
    }
    // 判断load连续性，排除部分split场景
    // 无法排除首轴split / split单输出等连续load场景
    if (ScheduleUtils::IsLoad(node) && !ScheduleUtils::IsContinuesVecStrides(node)) {
      GELOGD("Graph [%s]: Not contiguous load in graph, assigning default score.", origin_graph.GetName().c_str());
      return "";
    }
  }

  for (const auto &node : nddma_graph.GetAllNodes()) {
    if (nddma_node_cnt > 1) {
      GELOGD("Graph [%s]: Has more than 1 Nddma ndoe, assigning default score.",
             nddma_graph.GetName().c_str());
      return "";
    }
    if (IsNddma(node)) {
      nddma_node = node;
      ++nddma_node_cnt;
    }
  }
  std::stringstream ss;
  ss << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {" << std::endl;
  GE_WARN_ASSERT(nddma_node != nullptr);

  // 静态图
  if (IsStaticGraph(origin_graph)) {
    return GetStaticScoreFunc(nddma_node, ss);
  }
  return GetDynamicScoreFunc(nddma_graph, nddma_node, ss);
}

}  // namespace optimize