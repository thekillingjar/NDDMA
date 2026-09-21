/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "unaligned_template.h"
#include "graph_utils.h"
namespace {
constexpr uint32_t kAlignWidth = 32U;
}

namespace optimize {

std::string UnalignedTemplate::GenName(const std::string &general_case_name) {
  return general_case_name + "_unaligned";
}

bool UnalignedTemplate::NeedRemovePad(const ge::AscNodePtr &node) {
  // 如果是非scalar的Broadcast节点，直接插RemovePad，结束循环
  if (ScheduleUtils::IsBroadcast(node) && !ScheduleUtils::IsScalarBroadcastNode(node)) {
    return true;
  }
  if (ScheduleUtils::IsLoad(node) && node->GetInDataNodesSize() == 1UL && node->GetOutDataNodesSize() > 0UL) {
    // 判断Load是否是非连续的
    const auto &repeats = node->outputs[0].attr.repeats;
    const auto &strides = node->outputs[0].attr.strides;
    return !ScheduleUtils::IsContinuesStrides(repeats, strides);
  }
  return false;
}

Status UnalignedTemplate::UnAlignVectorizedStrides(const ge::AscNodePtr &node) {
  GELOGD("Calc un-alignment vectorized strides for node: %s[%s]", node->GetTypePtr(), node->GetNamePtr());
  for (const auto &output_attr : node->outputs()) {
    GE_CHECK_NOTNULL(output_attr);
    auto &attr = output_attr->attr;
    std::vector<ge::Expression> vector_repeats;
    GE_ASSERT_SUCCESS(ScheduleUtils::GetVectorRepeats(attr.repeats, attr.axis, attr.vectorized_axis, vector_repeats));
    GE_ASSERT_EQ(vector_repeats.size(), attr.vectorized_strides.size());
    ge::Expression size_product = ge::sym::kSymbolOne;
    for (int64_t i = static_cast<int64_t>(attr.vectorized_strides.size()) - 1; i >= 0; i--) {
      // 如果vector_strides不是0，说明要么repeat不是1，要么是非连续写，此时stride需要改为乘积
      if (attr.vectorized_strides[i] != ge::sym::kSymbolZero) {
        attr.vectorized_strides[i] = size_product;
      }
      size_product = size_product * vector_repeats[i];
    }
  }
  return ge::SUCCESS;
}

Status UnalignedTemplate::ReverseDfsUnAlignNode(ge::AscGraph &impl_graph, const ge::NodePtr &ge_node,
                                                std::set<ge::NodePtr> &visited_nodes) {
  // 这些节点不需要对齐
  if (ScheduleUtils::IsIOBuffer(ge_node) || ScheduleUtils::IsRemovePad(ge_node)) {
    return ge::SUCCESS;
  }
  const auto &node = std::dynamic_pointer_cast<ge::AscNode>(ge_node);
  if (visited_nodes.find(node) != visited_nodes.end()) {
    return ge::SUCCESS;
  }
  visited_nodes.insert(node);
  // step3: 判断是否需要插入RemovePad，如果需要则插入RemovePad并结束
  if (NeedRemovePad(node)) {
    ge::AscNodePtr remove_pad_node = nullptr;
    GE_WARN_ASSERT(ScheduleUtils::AddRemovePadAfter(impl_graph, node, remove_pad_node) == ge::SUCCESS);
    GE_WARN_ASSERT(UnAlignVectorizedStrides(remove_pad_node) == ge::SUCCESS);
    visited_nodes.insert(remove_pad_node);
    return ge::SUCCESS;
  }
  // step4: 如果不需要插入RemovePad，则还原不对齐的vector_strides
  GE_WARN_ASSERT(UnAlignVectorizedStrides(node) == ge::SUCCESS);
  for (const auto &in_node : node->GetInDataNodes()) {
    GE_WARN_ASSERT(ReverseDfsUnAlignNode(impl_graph, in_node, visited_nodes) == ge::SUCCESS);
  }
  return ge::SUCCESS;
}

/**
 * 在optimized graph基础上，对已经对齐到32B的vector_stride进行还原。具体逻辑如下：
 * 1. broadcast以及broadcast之前的节点尾轴正常对齐，broadcast之后的节点尾轴不做32B对齐。
 * 2. 在broadcast之后新增一个RemovePad节点，其输出也不做32B对齐。
 * 3. 若整个TilingCase中没有broadcast，则表示原TilingCase中存在的Broadcast因为冗余被消除了，此时所有节点都不做32B对齐。
 *
 * 处理逻辑如下：先找到所有的output节点，然后倒序遍历，逐节点还原vector_strides，若遇到brc则停止，并在其后插入RemovePad节点。
 */
ge::Status UnalignedTemplate::Generate(const ge::AscGraph &origin_graph,
                                       [[maybe_unused]] const ge::AscGraph &based_case,
                                       ge::AscGraph &new_case) {
  if (!ScheduleUtils::NotNeedAlignVectorStride(origin_graph)) {
    GELOGD("Not need to generate unaligned template for TilingCase: %s", origin_graph.GetName().c_str());
    return ge::FAILED;
  }
  // step1: 收集所有的Store节点，因为Output只有1个输入且必定是Store
  std::vector<ge::NodePtr> store_nodes;
  for (const auto &node : new_case.GetAllNodes()) {
    if (ScheduleUtils::IsStore(node)) {
      store_nodes.push_back(node);
    }
  }

  std::set<ge::NodePtr> visited_nodes;
  // step2: 从Store节点倒序遍历，output节点本身不需要取消对齐
  GE_CHECK_GE(store_nodes.size(), 1UL);
  size_t continues_store_cnt = 0UL;
  // 此处如果有Concat，则必为小尾轴case，需要进行连续处理
  // 只判断Store会因为Concat改变了shape导致误判，所以加此判断
  for (const auto &node : store_nodes) {
    const auto &src_nodes = node->GetInDataNodes();
    const auto connect_to_concat = (!src_nodes.empty()) && (src_nodes.at(0U)->GetType() == ge::ascir_op::Concat::Type);
    if ((!connect_to_concat) && ScheduleUtils::IsContinuesVecStrides(std::dynamic_pointer_cast<ge::AscNode>(node))) {
      GELOGD("Graph[%s] Node[%s] is continues.", new_case.GetName().c_str(), node->GetNamePtr());
      continues_store_cnt++;
      continue;
    }
    GE_WARN_ASSERT(ReverseDfsUnAlignNode(new_case, node, visited_nodes) == ge::SUCCESS);
  }
  if (continues_store_cnt == store_nodes.size()) {
    GELOGD("Graph[%s] is continues, do not need generate un-aligned tiling case.", new_case.GetName().c_str());
    return ge::FAILED;
  }
  GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(new_case));
  visited_nodes.clear();
  return ge::SUCCESS;
}

bool UnalignedTemplate::NeedDropBasedCase([[maybe_unused]] const ge::AscGraph &origin_graph,
                                          [[maybe_unused]] const ge::AscGraph &based_case, 
                                          const ge::AscGraph &new_case) {
  // 场景1：有Concat节点并且没有RemovePad节点，此时一定是Concat小尾轴场景，所以一定走非对齐模板
  const auto has_concat_node = ScheduleUtils::FindFirstNodeOfType<ge::ascir_op::Concat>(new_case) != nullptr;
  const auto has_remove_pad_node = ScheduleUtils::FindFirstNodeOfType<ge::ascir_op::RemovePad>(new_case) != nullptr;
  if (has_concat_node && (!has_remove_pad_node)) {
    GELOGI("[%s] has concat node, and unaligned graph does not have RemovePad", new_case.GetName().c_str());
    return true;
  }

  // 场景2：静态shape，尾轴小于32B，一定走非对齐模板
  const auto store_node = ScheduleUtils::FindFirstNodeOfType<ge::ascir_op::Store>(new_case);
  if (ScheduleUtils::IsTailAxisLessThan(store_node, kAlignWidth)) {
    GELOGI("Graph[%s] Store[%s] tail axis size < %u Bytes.", new_case.GetName().c_str(), store_node->GetNamePtr(),
      kAlignWidth);
    return true;
  }

  return false;
}

}  // namespace optimize