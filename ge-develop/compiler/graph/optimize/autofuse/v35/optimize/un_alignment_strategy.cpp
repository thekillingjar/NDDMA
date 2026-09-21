/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "un_alignment_strategy.h"
#include "common_utils.h"
#include "tensor_layout_utils.h"
#include "can_fuse/backend/backend_utils.h"
#include "platform/v1/alignment_strategy.h"

namespace optimize {
AlignmentType UnAlignmentStrategy::GetDefaultAlignmentType() {
  return AlignmentType::kNotAligned;
}

ge::Status UnAlignmentStrategy::LoadAlignmentInferFunc(const ge::AscNodePtr &node) {
  const auto &output_attr = node->outputs[0].attr;
  if (!ge::ops::IsOps<ge::ascir_op::Load>(node)) {
    GELOGD("Node[%s] is continuous loading, input tensor does not needs to be aligned.", node->GetNamePtr());
    // vectorized_axis连续则可以连续搬运
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
    return ge::SUCCESS;
  }

  if (IsLoadNeedAlignForReduce(node)) {
    // 符合该条件的load节点会转为nddma，不必对齐
    GELOGD("Node[%s] is discontinuous loading, input tensor needs to be fixed not aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
    return ge::SUCCESS;
  }
  DiscontinuityInfo info;
  GE_ASSERT_SUCCESS(TensorLayoutUtils::AnalyzeLoadDiscontinuity(output_attr, info),
                    "Failed to analyze discontinuity info for node:[%s].", node->GetNamePtr());
  if (!info.has_multiple_discontinuities) {
    // 走compat模式搬运
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
  } else if (info.is_tail_axis_discontinuous) {
    // vectorized_axis在尾轴非连续, 且不能走compat模式, 需要转nddma
    GELOGD("Node[%s] is tail axis discontinuous loading, need to use nddma.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
  } else {
    // vectorized_axis 在gm上非连续,需要尾轴对齐搬运
    GELOGD("Node[%s] is discontinuous loading, input tensor needs to be aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kAligned};
  }
  return ge::SUCCESS;
}

ge::Status UnAlignmentStrategy::StoreAlignmentInferFunc(const ge::AscNodePtr &node) {
  const auto &output_attr = node->outputs[0].attr;
  AlignmentType input_align = tensor_to_align_type_[&node->inputs[0].attr].align_type;
  tensor_to_align_type_[&output_attr] = {input_align};
  auto owner_graph = node->GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph);
  auto graph_attr = owner_graph->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  const auto &axis = graph_attr->axis;
  size_t tile_inner_axis_size = 0UL;
  for (auto axis_id : node->inputs[0].attr.vectorized_axis) {
    auto iter =
        std::find_if(axis.begin(), axis.end(), [axis_id](const ge::AxisPtr &axis) { return axis->id == axis_id; });
    if (iter != axis.end() && (*iter)->type == ge::Axis::kAxisTypeTileInner) {
      ++tile_inner_axis_size;
    }
  }

  if (!ScheduleUtils::IsVectorizedAxisContinuousInGM(output_attr) && (tile_inner_axis_size > 1UL)) {
    GELOGD("Node[%s] is discontinuous writing, input tensor needs to be aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kAligned};
    GE_ASSERT_SUCCESS(BackPropagateAlignment(node, AlignmentType::kAligned));
  }
  return ge::SUCCESS;
}

bool IsTailBroadcastNode(ge::AscNode &node) {
  // 判断节点尾轴是否广播的节点
  if (node.attr.api.compute_type != ge::ComputeType::kComputeBroadcast) {
    return false;
  }

  const auto &in_nodes = node.GetInDataNodes();
  // scalar广播
  if (ge::ops::IsOps<ge::ascir_op::Scalar>(in_nodes.at(0UL))) {
    GELOGD("Input of Broadcast[%s] is Scalar[%s], support.", node.GetNamePtr(), in_nodes.at(0UL)->GetNamePtr());
    return true;
  }
  auto inputs = node.inputs();
  auto outputs = node.outputs();
  GE_ASSERT(!inputs.empty());
  GE_ASSERT(!outputs.empty());
  GE_ASSERT_NOTNULL(inputs[0]);
  GE_ASSERT_NOTNULL(outputs[0]);

  const auto &in_attr = inputs[0]->attr;
  std::vector<ge::Expression> in_vec_repeats;
  GE_ASSERT(ScheduleUtils::GetVectorRepeats(in_attr.repeats, in_attr.axis, in_attr.vectorized_axis, in_vec_repeats) ==
            ge::SUCCESS);

  const auto &out_attr = outputs[0]->attr;
  std::vector<ge::Expression> out_vec_repeats;
  GE_ASSERT(ScheduleUtils::GetVectorRepeats(out_attr.repeats, out_attr.axis, out_attr.vectorized_axis,
                                            out_vec_repeats) == ge::SUCCESS);

  // 输入输出vectorized_strides大小一致，且非空
  GE_ASSERT(in_vec_repeats.size() == out_vec_repeats.size());
  GE_ASSERT(!in_vec_repeats.empty());
  auto vec_size = in_vec_repeats.size();
  auto in_last_dim_size = in_vec_repeats[vec_size - 1];
  auto out_last_dim_size = out_vec_repeats[vec_size - 1];
  auto is_tail_brc = ascgen_utils::ExpressEq(in_last_dim_size, ge::Symbol(1)) &&
                     !ascgen_utils::ExpressEq(in_last_dim_size, out_last_dim_size);
  return is_tail_brc;
}

ge::Status UnAlignmentStrategy::SetAlignInfoForTailBrcNodes(AlignmentType aligned_type, ge::AscNode *node,
                                                            std::set<ge::Node *> &visited_nodes,
                                                            std::queue<ge::Node *> &node_queue) {
  for (const auto &output : node->outputs()) {
    auto &align_info = tensor_to_align_type_[&output->attr];
    if (align_info.align_type == aligned_type) {
      continue;
    }
    if (align_info.align_type == AlignmentType::kFixedNotAligned) {
      align_info.conflict_with_output = true;
      continue;
    }
    GELOGD("Node [%s]'s align type need to be changed.", node->GetNamePtr());
    align_info.align_type = aligned_type;

    for (const auto &peer_in : output->anchor.GetPeerInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(peer_in);
      auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(peer_in->GetOwnerNode());
      if (!ScheduleUtils::IsBuffer(asc_node) && visited_nodes.insert(asc_node.get()).second) {
        node_queue.push(asc_node.get());
      }
    }
  }
  visited_nodes.insert(node);
  return ge::SUCCESS;
}

ge::Status UnAlignmentStrategy::BackPropagateAlignment(const ge::AscNodePtr &node, AlignmentType aligned_type) {
  std::set<ge::Node *> visited_nodes;
  std::queue<ge::Node *> node_queue;
  visited_nodes.emplace(node.get());
  SetAlignInfoForNodeInputs(aligned_type, node.get(), visited_nodes, node_queue);
  while (!node_queue.empty()) {
    const auto curr_node = dynamic_cast<ge::AscNode *>(node_queue.front());
    node_queue.pop();
    GE_ASSERT_NOTNULL(curr_node);
    if (IsTailBroadcastNode(*curr_node)) {
      GE_ASSERT(SetAlignInfoForTailBrcNodes(aligned_type, curr_node, visited_nodes, node_queue) == ge::SUCCESS);
      continue;
    }

    bool alignment_changed = SetAlignInfoForNodeOutputs(aligned_type, curr_node, visited_nodes, node_queue);
    if (alignment_changed) {
      SetAlignInfoForNodeInputs(aligned_type, curr_node, visited_nodes, node_queue);
    }
  }
  return ge::SUCCESS;
}

ge::Status UnAlignmentStrategy::ConcatAlignmentInferFunc(const ge::AscNodePtr &node) {
  const auto &output_attr = node->outputs[0].attr;
  tensor_to_align_type_[&output_attr] = {AlignmentType::kFixedNotAligned};
  return ge::SUCCESS;
}

/**
 * 功能：将尾轴transpose的load节点替换为nddma节点；
 * 与naddma_template的区别：这里仅处理load，而nddma模板则处理load + brc或load + cast
 */
Status GenLoadToGenNddmaNode(const ge::AscNodePtr &node_load) {
  GE_CHECK_NOTNULL(node_load);
  GE_CHECK_NOTNULL(node_load->GetOpDesc());

  node_load->GetOpDesc()->SetType("Nddma");
  node_load->attr.type = "Nddma";
  return ge::SUCCESS;
}
}  // namespace optimize