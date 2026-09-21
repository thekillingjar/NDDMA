/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "node_matcher.h"
#include "graph/utils/constant_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "common/math/math_util.h"
#include "graph/ir_definitions_recover.h"
#include "exe_graph/lowering/bg_ir_attrs.h"

namespace ge {
namespace fusion{
namespace {
template<typename T>
bool IsTensorDataEqualWith(ConstGeTensorPtr &a_tensor, ConstGeTensorPtr &b_tensor, size_t shape_size) {
  const auto *a_value = reinterpret_cast<const T *>(a_tensor->GetData().data());
  const auto *b_value = reinterpret_cast<const T *>(b_tensor->GetData().data());
  for (size_t i = 0U; i < shape_size; ++i) {
    if(!IsEqualWith(a_value[i], b_value[i])) {
      return false;
    }
  }
  return true;
}

bool IsTensorEqualWith(ConstGeTensorPtr &a_tensor, ConstGeTensorPtr &b_tensor) {
  const auto &a_tensor_desc = a_tensor->GetTensorDesc();
  const auto &b_tensor_desc = b_tensor->GetTensorDesc();

  if (!TensorUtils::IsShapeEqual(a_tensor_desc.GetShape(), b_tensor_desc.GetShape())) {
    return false;
  }

  if (a_tensor_desc.GetDataType() != b_tensor_desc.GetDataType()) {
    return false;
  }

  if (a_tensor->GetData().GetSize() != b_tensor->GetData().GetSize()) {
    return false;
  }
  return memcmp(a_tensor->GetData().data(), b_tensor->GetData().data(), a_tensor->GetData().GetSize()) == 0;
}

NodePtr GetRealConst(const NodePtr &node, bool enable_cross_subgraph) {
  if (ConstantUtils::IsRealConst(node->GetOpDesc())) {
    return node;
  } else if (OpTypeUtils::IsSubgraphInnerData(node->GetOpDesc())){
    if (!enable_cross_subgraph) {
      return nullptr;
    }
    auto real_node = ge::NodeUtils::GetInNodeCrossSubgraph(node);
    GE_ASSERT_NOTNULL(real_node);
    if (!ConstantUtils::IsRealConst(real_node->GetOpDesc())) {
      GELOGI("[Matching][Const]The type of %s is not constant.", real_node->GetTypePtr());
      return nullptr;
    }
    return real_node;
  }
  return nullptr;
}
std::string AttrNamesToString(const std::vector<std::string> &attr_names) {
  std::stringstream ss;
  ss << "[";
  for (const auto &attr_name : attr_names) {
    ss << "{" << attr_name << "}";
  }
  ss << "]";
  return ss.str();
}

bool IsAttrNamesMatch(const NodePtr &p_node, const NodePtr &t_node, const std::vector<std::string> &p_attr_names) {
  const auto &t_attr_names = t_node->GetOpDesc()->GetIrAttrNames();
  GE_WARN_ASSERT(
      p_attr_names.size() == t_attr_names.size(),
      "[AttrMiss] Ir attr num is not match. P node[%s][%s] attr names: %s attr num is %zu, T node [%s][%s] attr names"
      ":%s attr num is %zu.",
      p_node->GetNamePtr(), p_node->GetTypePtr(), AttrNamesToString(p_attr_names).c_str(), p_attr_names.size(),
      t_node->GetNamePtr(), t_node->GetTypePtr(), AttrNamesToString(t_attr_names).c_str(), t_attr_names.size());
  GE_WARN_ASSERT(
      std::is_permutation(p_attr_names.begin(), p_attr_names.end(), t_attr_names.begin()),
      "[AttrMiss] Ir attr names is not match. P node[%s][%s] attr names: %s, T node [%s][%s] attr names: %s.",
      p_node->GetNamePtr(), p_node->GetTypePtr(), AttrNamesToString(p_attr_names).c_str(), t_node->GetNamePtr(),
      t_node->GetTypePtr(), AttrNamesToString(t_attr_names).c_str());
  return true;
}

bool IsAttrValuesMatch(const NodePtr &p_node, const NodePtr &t_node, const std::vector<std::string> &ir_attr_names) {
  size_t attr_num = ir_attr_names.size();
  std::vector<std::vector<uint8_t>> p_attr_values, t_attr_values;
  GE_WARN_ASSERT(gert::bg::GetAllIrAttrs(p_node, p_attr_values));
  GE_WARN_ASSERT(gert::bg::GetAllIrAttrs(t_node, t_attr_values));
  GE_WARN_ASSERT(p_attr_values.size() == attr_num,
                 "P node[%][%s] ir attr value num:[%zu] is not equal with ir attr def num:[%zu]", p_node->GetNamePtr(),
                 p_node->GetTypePtr(), p_attr_values.size(), attr_num);
  GE_WARN_ASSERT(t_attr_values.size() == attr_num,
                 "T node[%][%s] ir attr value num:[%zu] is not equal with ir attr def num:[%zu]", t_node->GetNamePtr(),
                 t_node->GetTypePtr(), t_attr_values.size(), attr_num);

  for (size_t i = 0U; i < attr_num; ++i) {
    const auto &attr_name = ir_attr_names[i];
    const auto &p_attr_value_buf = p_attr_values[i];
    const auto &t_attr_value_buf = t_attr_values[i];
    if (p_attr_value_buf.size() != t_attr_value_buf.size()) {
      GELOGD(
          "[AttrMiss] Ir attr value size is not match. Attr name [%s], P node[%s][%s] attr value size: %zu, T node "
          "[%s][%s] attr value size: %zu.",
          attr_name.c_str(), p_node->GetNamePtr(), p_node->GetTypePtr(), p_attr_value_buf.size(), t_node->GetNamePtr(),
          t_node->GetTypePtr(), t_attr_value_buf.size());
      return false;
    }
    if (memcmp(p_attr_value_buf.data(), t_attr_value_buf.data(), p_attr_value_buf.size()) != 0) {
      // todo better to print attr value
      GELOGD("[AttrMiss] Ir attr value is not match. Attr name [%s], P node[%s][%s], T node [%s][%s].",
             attr_name.c_str(), p_node->GetNamePtr(), p_node->GetTypePtr(), t_node->GetNamePtr(), t_node->GetTypePtr());
      return false;
    }
  }
  return true;
}

bool IsIrAttrMatch(const NodePtr &p_node, const NodePtr &t_node) {
  const auto &p_attr_names = p_node->GetOpDesc()->GetIrAttrNames();
  if (!IsAttrNamesMatch(p_node, t_node, p_attr_names)) {
    return false;
  }

  return IsAttrValuesMatch(p_node, t_node, p_attr_names);
}
} // namespace

bool NormalNodeMatcher::IsMatch(const NodePtr &p_node, const NodePtr &t_node) const {
  // todo make op type fuzzy
  if (!enable_ir_attr_match_) {
    return strcmp(p_node->GetTypePtr(), t_node->GetTypePtr()) == 0;
  }
  // 恢复ir属性后，没被设置的可选属性将挂在节点上，并设置为默认值，因此预期p和t的IR属性个数一致, 且ir attr names顺序一致
  GE_ASSERT_GRAPH_SUCCESS(RecoverOpDescIrDefinition(p_node->GetOpDesc()));
  GE_ASSERT_GRAPH_SUCCESS(RecoverOpDescIrDefinition(t_node->GetOpDesc()));

  return IsIrAttrMatch(p_node, t_node);
}

bool ConstantMatcher::IsMatch(const NodePtr &p_node, const NodePtr &t_node) const {
  auto real_const = GetRealConst(t_node, enable_match_cross_subgraph_);
  if (real_const == nullptr) {
    return false;
  }

  if (!enable_value_match_) {
    return true;
  }

  ConstGeTensorPtr p_tensor;
  if (!ConstantUtils::GetWeight(p_node->GetOpDesc(), 0, p_tensor)) {
    // no weight in pattern, only match type in t_node
    return true;
  }

  ConstGeTensorPtr t_tensor;
  if (!ConstantUtils::GetWeight(real_const->GetOpDesc(), 0, t_tensor)) {
    GELOGW("[Matching][Const]%s has no weight which is invalid const.", t_node->GetNamePtr());
    return false;
  }
  return IsTensorEqualWith(p_tensor, t_tensor);
}
}  // namespace fusion
}  // namespace ge
