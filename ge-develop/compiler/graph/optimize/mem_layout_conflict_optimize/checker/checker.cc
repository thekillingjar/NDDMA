/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checker.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "check_register.h"
#include "checker_log.h"
#include "common/checker.h"

namespace ge {
namespace {
static const std::set<AnchorAttribute> kNoConflictTypes{
    ANCHOR_ATTR_REFERENCE_OUTPUT};
}
std::set<std::pair<AnchorAttribute, AnchorAttribute>> RegAllAbsoluteNoConflict() {
  std::set<std::pair<AnchorAttribute, AnchorAttribute>> absolute_no_conflict_set;

  RegAbsoluteSet(ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT, ANCHOR_ATTR_NORMAL_OUTPUT, absolute_no_conflict_set);

  RegAbsoluteSet(ANCHOR_ATTR_USER_MEMORY_OUTPUT, ANCHOR_ATTR_USER_MEMORY_OUTPUT, absolute_no_conflict_set);

  RegAbsoluteSet(ANCHOR_ATTR_USER_MEMORY_INPUT, ANCHOR_ATTR_USER_MEMORY_OUTPUT, absolute_no_conflict_set);
  return absolute_no_conflict_set;
}

/*
 * 对于绝对冲突的场景，如果两个都是OutDataAnchor，需要单独提供checker函数。
 * 这里不支持两个都是OutDataAnchor的场景。
 */
std::set<std::pair<AnchorAttribute, AnchorAttribute>> RegAllAbsoluteConflict() {
  std::set<std::pair<AnchorAttribute, AnchorAttribute>> absolute_conflict_set;

  RegAbsoluteSet(ANCHOR_ATTR_USER_MEMORY_INPUT, ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_INPUT,
                 absolute_conflict_set);
  RegAbsoluteSet(ANCHOR_ATTR_USER_MEMORY_INPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT, absolute_conflict_set);

  RegAbsoluteSet(ANCHOR_ATTR_USER_MEMORY_OUTPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT, absolute_conflict_set);
  RegAbsoluteSet(ANCHOR_ATTR_USER_MEMORY_OUTPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT, absolute_conflict_set);
  RegAbsoluteSet(ANCHOR_ATTR_USER_MEMORY_OUTPUT, ANCHOR_ATTR_CONTINUOUS_OUTPUT, absolute_conflict_set);
  RegAbsoluteSet(ANCHOR_ATTR_USER_MEMORY_OUTPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT, absolute_conflict_set);

  RegAbsoluteSet(ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT, ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT, absolute_conflict_set);
  RegAbsoluteSet(ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT, ANCHOR_ATTR_CONTINUOUS_INPUT, absolute_conflict_set);

  RegAbsoluteSet(ANCHOR_ATTR_CONTINUOUS_INPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT, absolute_conflict_set);

  RegAbsoluteSet(ANCHOR_ATTR_CONTINUOUS_OUTPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT, absolute_conflict_set);

  return absolute_conflict_set;
}

Checker::Checker() {
  auto &allCheckFunc = GetRegister();
  checkers_ = allCheckFunc;
  absolute_conflict_set_ = RegAllAbsoluteConflict();
  absolute_no_conflict_set_ = RegAllAbsoluteNoConflict();
}

const std::map<AnchorAttribute, std::string> Checker::kAnchorTypeStr{
    {ANCHOR_ATTR_USER_MEMORY_INPUT, "USER_MEMORY_INPUT"},
    {ANCHOR_ATTR_USER_MEMORY_OUTPUT, "USER_MEMORY_OUTPUT"},
    {ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT, "IMMUTABLE_ADDRESS_OUTPUT"},
    {ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT, "UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT"},
    {ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_INPUT, "UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_INPUT"},
    {ANCHOR_ATTR_CONTINUOUS_INPUT, "CONTINUOUS_INPUT"},
    {ANCHOR_ATTR_CONTINUOUS_OUTPUT, "CONTINUOUS_OUTPUT"},
    {ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT, "NOPADDING_CONTINUOUS_INPUT"},
    {ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT, "NOPADDING_CONTINUOUS_OUTPUT"},
    {ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT, "RTS_SPECIAL_TYPE_INPUT"},
    {ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT, "RTS_SPECIAL_TYPE_OUTPUT"},
    {ANCHOR_ATTR_REFERENCE_OUTPUT, "REFERENCE_OUTPUT"},
    {ANCHOR_ATTR_NORMAL_INPUT, "NORMAL_INPUT"},
    {ANCHOR_ATTR_NORMAL_OUTPUT, "NORMAL_OUTPUT"},
    {ANCHOR_ATTR_RESERVED, "RESERVED"}};

bool Checker::CheckNoConflict(const AnchorAttribute &type_a, const AnchorAttribute &type_b) const {
  if ((kNoConflictTypes.count(type_a) > 0U) || (kNoConflictTypes.count(type_b) > 0U)) {
    return true;
  }
  return IsInAbsoluteSet(type_a, type_b, absolute_no_conflict_set_);
}

/*
 * 对于绝对冲突的场景，如果两个都是OutDataAnchor，需要单独提供checker函数。
 * 这里不支持两个都是OutDataAnchor的场景。
 */
Status Checker::CheckAbsoluteConflict(CheckFuncContext &context) const {
  const auto &node_a = context.node_a;
  const auto &node_b = context.node_b;
  const ge::AnchorPtr anchor_a = MemLayoutConflictUtil::GetAnchorFromIndexIo(node_a);
  const ge::AnchorPtr anchor_b = MemLayoutConflictUtil::GetAnchorFromIndexIo(node_b);
  GE_ASSERT_NOTNULL(anchor_a);
  GE_ASSERT_NOTNULL(anchor_b);
  // 如果两个都是InDataAnchor，为了避免随机，总是选择一个拓扑ID小的
  if (anchor_a->IsTypeIdOf<ge::InDataAnchor>() && anchor_b->IsTypeIdOf<ge::InDataAnchor>()) {
    if (node_a.node_->GetOpDesc()->GetId() < node_b.node_->GetOpDesc()->GetId()
        && (!MemLayoutConflictUtil::IsNodeOutRefFromInput(node_a, context.all_nodes))) {
      context.result.insert(node_a.node_ptr_->GetInDataAnchor(node_a.index_));
      GE_MEM_LAYOUT_CONFLICT_LOGI(context, node_a);
    } else {
      context.result.insert(node_b.node_ptr_->GetInDataAnchor(node_b.index_));
      GE_MEM_LAYOUT_CONFLICT_LOGI(context, node_b);
    }
    return SUCCESS;
  }

  if (anchor_a->IsTypeIdOf<ge::InDataAnchor>()) {
    context.result.insert(node_a.node_ptr_->GetInDataAnchor(node_a.index_));
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, node_a);
    return SUCCESS;
  }

  if (anchor_b->IsTypeIdOf<ge::InDataAnchor>()) {
    context.result.insert(node_b.node_ptr_->GetInDataAnchor(node_b.index_));
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, node_b);
    return SUCCESS;
  }
  GE_ASSERT_TRUE(false, "Scenarios where both anchors are OutDataAnchors are not supported.");
}

bool Checker::IsSkip(CheckFuncContext &context) const {
  const auto &node_a = context.node_a;
  const auto &node_b = context.node_b;

  // 相同节点的anchor间直接跳过。输出引用输入的节点会有这种场景
  if (context.node_a.node_ == context.node_b.node_) {
    return true;
  }

  // 1 有 ref_var_src_var_name 属性的，引用anchor和被引用anchor不需要判断冲突
  // 2 但是有一个例外场景，如果是连续内存，仍然属于冲突场景，不能跳过。
  //   具体场景见：MemConflictShareGraph::BuildContinuousOutAndContinuousOutHcomByRefGraph
  if ((node_a.io_type_ == kOut) && (node_b.io_type_ == kOut)
      && (!MemLayoutConflictUtil::IsContainTargetType(context.type_b, ANCHOR_ATTR_CONTINUOUS_OUTPUT))
      && (!MemLayoutConflictUtil::IsContainTargetType(context.type_a, ANCHOR_ATTR_CONTINUOUS_OUTPUT))) {
    const auto &node_a_src_node = GetRefSrcAnchor(node_a.node_->GetOutDataAnchor(node_a.index_));
    const auto &node_b_src_node = GetRefSrcAnchor(node_b.node_->GetOutDataAnchor(node_b.index_));
    const auto is_a_ref_b = (node_a_src_node == node_b.node_);
    const auto is_b_ref_a = (node_b_src_node == node_a.node_);
    if (is_a_ref_b || is_b_ref_a) {
      GELOGI("[MemConflict] reference by ref_var_src_var_name attr, skip."
             " node_a: %s, node_b:%s, is_a_ref_b: %d, is_b_ref_a:%d",
             node_a.ToString().c_str(), node_b.ToString().c_str(), is_a_ref_b, is_b_ref_a);
      return true;
    }
  }
  return false;
}

Status Checker::CheckConditionConflict(const AnchorAttribute &type_a,
                                       const AnchorAttribute &type_b,
                                       CheckFuncContext &context) const {
  const auto checker_func = checkers_[type_a][type_b];
  if ((checker_func != nullptr) && (!checker_func->IsCallAsSymbol())) {
    GE_ASSERT_SUCCESS(checker_func->GetFunc()(context), "call [%s, %s] checker func failed.",
                      CheckerLog::ToStr(type_a).c_str(), CheckerLog::ToStr(type_b).c_str());
  }
  return SUCCESS;
}

Status Checker::CheckConditionConflictAsSymbol(const AnchorAttribute &type_a,
                                               const AnchorAttribute &type_b,
                                               CheckFuncContext &context,
                                               std::vector<vector_bit_t> &visit_flag) const {
  const auto checker_func = checkers_[type_a][type_b];
  if ((checker_func != nullptr) && (checker_func->IsCallAsSymbol())
      && (!visit_flag[type_a][type_b])) {
    GE_ASSERT_SUCCESS(checker_func->GetFunc()(context), "call [%s, %s] checker func failed.",
                      CheckerLog::ToStr(type_a).c_str(), CheckerLog::ToStr(type_b).c_str());
    visit_flag[type_a][type_b] = true;
    visit_flag[type_b][type_a] = true;
  }
  return SUCCESS;
}

Status Checker::CheckConflict(CheckFuncContext &context, std::vector<vector_bit_t> &visit_flag) const {
  if (IsSkip(context)) {
    return SUCCESS;
  }
  for (size_t i = 0U; i < context.type_a.size(); i++) {
    context.type_a_index = i;
    for (size_t j = 0U; j < context.type_b.size(); j++) {
      if (CheckNoConflict(context.type_a[i], context.type_b[j])) {
        continue;
      }
      context.type_b_index = j;
      if (IsInAbsoluteSet(context.type_a[i], context.type_b[j], absolute_conflict_set_)) {
        GE_ASSERT_SUCCESS(CheckAbsoluteConflict(context), "call [%s, %s] checker func failed.",
                          CheckerLog::ToStr(context.type_a[i]).c_str(), CheckerLog::ToStr(context.type_b[i]).c_str());
        continue;
      }
      GE_ASSERT_SUCCESS(CheckConditionConflictAsSymbol(context.type_a[i], context.type_b[j], context, visit_flag),
                        "call [%s, %s] checker func failed.",
                        CheckerLog::ToStr(context.type_a[i]).c_str(), CheckerLog::ToStr(context.type_b[i]).c_str());
      GE_ASSERT_SUCCESS(CheckConditionConflict(context.type_a[i], context.type_b[j], context),
                        "call [%s, %s] checker func failed.",
                        CheckerLog::ToStr(context.type_a[i]).c_str(), CheckerLog::ToStr(context.type_b[i]).c_str());
    }
  }
  return SUCCESS;
}

void Checker::SaveTypes(const TypeVector &types, const AnchorPtr anchor) {
  anchor_type_vectors_[anchor] = types;
}

void Checker::SaveRefMap(const OutDataAnchorPtr &out_anchor, const NodePtr &src_node) {
  ref_map_[out_anchor] = src_node;
}

const NodePtr &Checker::GetRefSrcAnchor(const OutDataAnchorPtr &out_anchor) const {
  const auto it = ref_map_.find(out_anchor);
  if (it != ref_map_.end()) {
    return it->second;
  }

  static NodePtr dummy_node = nullptr;
  return dummy_node;
}

void Checker::ClearTypes() {
  anchor_type_vectors_.clear();
}

const TypeVector &Checker::GetTypes(const ge::NodeIndexIO &node) const {
  const auto anchor = MemLayoutConflictUtil::GetAnchorFromIndexIo(node);
  const auto it = anchor_type_vectors_.find(anchor);
  if (it != anchor_type_vectors_.end()) {
    return it->second;
  }

  // if a node can not find type, it is normal, because a node has no anchor, so return empty type
  static SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN> dummy_type;
  return dummy_type;
}
}  // namespace ge
