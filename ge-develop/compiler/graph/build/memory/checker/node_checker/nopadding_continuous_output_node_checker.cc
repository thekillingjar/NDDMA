/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stack>
#include "../node_checker_register.h"
#include "../node_checker_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_common/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/utils/tensor_utils.h"
#include "common/checker.h"
#include "math_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"

namespace ge {
namespace {
bool IsUseOffsetList(const Node *const node) {
  const auto &out_anchors = node->GetAllOutDataAnchorsPtr();
  for (const auto &out_anchor : out_anchors) {
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr());
      GE_ASSERT_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr());
      std::vector<int64_t> offset_list = {};
      (void)AttrUtils::GetListInt(peer_in_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr(),
                                  ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
      if (!offset_list.empty()) {
        return true;
      }
    }
  }
  return false;
}

Status GetInputOffsetForContinuous(const OutDataAnchor *const out_data_anchor, int64_t &offset) {
  for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
    const auto node = peer_in_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
    std::vector<int64_t> offset_list = {};
    const bool has_offset_list =
        AttrUtils::GetListInt(node->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
    if (has_offset_list && (!offset_list.empty())) {
      GE_ASSERT_TRUE(static_cast<size_t>(peer_in_anchor->GetIdx()) < offset_list.size(),
                     "input_index: %d, offset_list size: %zu, node: %s", peer_in_anchor->GetIdx(), offset_list.size(),
                     NodeCheckerUtils::NodeName(node).c_str());
      offset = offset_list.at(peer_in_anchor->GetIdx());
      break;
    }
  }
  return SUCCESS;
}

Status GetBaseOffset(const Node *node, int64_t &base_offset) {
  const auto &out_anchors = node->GetAllOutDataAnchorsPtr();
  GE_ASSERT_TRUE(!out_anchors.empty());
  const auto first_out_anchor = out_anchors.front();
  const auto out_tensor = node->GetOpDescBarePtr()->GetOutputDescPtr(0);
  GE_ASSERT_NOTNULL(out_tensor);
  int64_t offset = 0;
  GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(node->GetOpDescBarePtr(), 0, offset));
  int64_t continuous_offset = 0;
  // 首个out_anchor如果空悬，获得continuous_offset为0也符合预期
  GE_ASSERT_SUCCESS(GetInputOffsetForContinuous(first_out_anchor, continuous_offset), "node: %s, out_index: %s",
                    NodeCheckerUtils::NodeName(node).c_str(), 0);
  GE_ASSERT_TRUE(offset >= continuous_offset, "offset: %" PRId64 ", continuous_offset: %" PRId64 "", offset,
                 continuous_offset);
  base_offset = offset - continuous_offset;
  return SUCCESS;
}

Status NoPaddingContinuousOutputNodeOffsetChecker(const NodeCheckerParam &param) {
  int64_t expect_offset = -1;
  bool use_offset_list = IsUseOffsetList(param.node);
  int64_t base_offset = 0;
  if (use_offset_list) {
    GE_ASSERT_SUCCESS(GetBaseOffset(param.node, base_offset));
  }
  for (const auto &out_data_anchor : param.node->GetAllOutDataAnchorsPtr()) {
    const auto out_index = out_data_anchor->GetIdx();
    const auto out_tensor = param.node->GetOpDescBarePtr()->GetOutputDescPtr(out_index);
    GE_ASSERT_NOTNULL(out_tensor);
    int64_t offset = 0;
    GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(param.node->GetOpDescBarePtr(), out_index, offset));

    int64_t stride = 0;
    if (!use_offset_list) {
      GE_ASSERT_SUCCESS(NodeCheckerUtils::GetStrideForNoPaddingContinuousOutput(param.node, out_index, stride));
      if (expect_offset == -1) {
        GE_ASSERT_TRUE(!AddOverflow(offset, stride, expect_offset), "offset: %" PRId64 ", len: %" PRId64 "", offset,
                       stride);
        continue;
      }
    } else {
      // 如果输出anchor空悬，无法获取ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS值，无法检查
      if (out_data_anchor->GetPeerInDataAnchorsPtr().empty()) {
        continue;
      }
      int64_t continuous_offset = 0;
      GE_ASSERT_SUCCESS(GetInputOffsetForContinuous(out_data_anchor, continuous_offset), "node: %s, out_index: "
                        "%" PRId64 "", NodeCheckerUtils::NodeName(param.node).c_str(), out_index);
      GE_ASSERT_TRUE(!AddOverflow(base_offset, continuous_offset, expect_offset),
                     "offset: %" PRId64 ", len: %" PRId64 "", base_offset, continuous_offset);
    }
    if (expect_offset != offset) {
      REPORT_INNER_ERR_MSG("E19999", "nopadding continuous output node memory check failed. node: %s, out_index: "
                         "%d, offset: %" PRId64 ", expect_offset: %" PRId64 "",
                         NodeCheckerUtils::NodeName(param.node).c_str(), out_index, offset, expect_offset);
      GELOGE(FAILED, "nopadding continuous output node memory check failed. node: %s, out_index: %d,"
          " offset: %" PRId64 ", expect_offset: %" PRId64 "", NodeCheckerUtils::NodeName(param.node).c_str(), out_index,
          offset, expect_offset);
      GE_ASSERT_SUCCESS(NodeCheckerUtils::ErrorLogAllOutputs(param));
      return FAILED;
    }
    if (!use_offset_list) {
      GE_ASSERT_TRUE(!AddOverflow(offset, stride, expect_offset), "offset: %" PRId64 ", len: %" PRId64 "", offset,
                     stride);
    }
  }
  return SUCCESS;
}

bool InputNodeIsPhonySplit(const Node *const node) {
  if ((!node->GetAllInDataAnchorsPtr().empty()) &&
      (node->GetAllInDataAnchorsPtr().front()->GetPeerOutAnchor() != nullptr) &&
      (node->GetAllInDataAnchorsPtr().front()->GetPeerOutAnchor()->GetOwnerNodeBarePtr() != nullptr)) {
    return MemLayoutConflictUtil::IsNoPaddingContinuousOutput(
        node->GetAllInDataAnchorsPtr().front()->GetPeerOutAnchor()->GetOwnerNodeBarePtr());
  }
  return false;
}

Status FindLastOutputPhonySplit(const Node *const node, OutDataAnchor **out_data_anchor) {
  std::stack<const Node *> node_stack;
  node_stack.push(node);
  while (!node_stack.empty()) {
    const auto ps = node_stack.top();
    node_stack.pop();
    GE_ASSERT_TRUE(!ps->GetAllOutDataAnchorsPtr().empty());
    *out_data_anchor = ps->GetAllOutDataAnchorsPtr().back();
    for (const auto peer_in_anchor : ps->GetAllOutDataAnchorsPtr().back()->GetPeerInDataAnchorsPtr()) {
      if (MemLayoutConflictUtil::IsNoPaddingContinuousOutput(peer_in_anchor->GetOwnerNodeBarePtr())) {
        node_stack.push(peer_in_anchor->GetOwnerNodeBarePtr());
        break;
      }
    }
  }
  return SUCCESS;
}

Status GetOutputOffsetRange(const OutDataAnchor *const out_data_anchor, int64_t &start, int64_t &end) {
  GE_ASSERT_NOTNULL(out_data_anchor);
  const auto node = out_data_anchor->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(node);
  const auto out_index = out_data_anchor->GetIdx();

  GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(node->GetOpDescBarePtr(), out_index, start), "node: %s",
                    NodeCheckerUtils::NodeName(node).c_str());
  int64_t out_size = 0;
  GE_ASSERT_SUCCESS(TensorUtils::GetSize(node->GetOpDescBarePtr()->GetOutputDesc(out_index), out_size),
                    "node: %s, out_index: %d", NodeCheckerUtils::NodeName(node).c_str(), out_index);
  GE_ASSERT_TRUE(!AddOverflow(start, out_size, end), "start: %" PRId64 ", out_size: %" PRId64 "", start, out_size);
  return SUCCESS;
}

Status GetNoPaddingOutputOffsetRange(const OutDataAnchor *const out_data_anchor, int64_t &start, int64_t &end) {
  GE_ASSERT_NOTNULL(out_data_anchor);
  const auto node = out_data_anchor->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(node);
  const auto out_index = out_data_anchor->GetIdx();

  GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(node->GetOpDescBarePtr(), out_index, start), "node: %s",
                    NodeCheckerUtils::NodeName(node).c_str());
  int64_t out_size = 0;
  GE_ASSERT_SUCCESS(NodeCheckerUtils::GetStrideForNoPaddingContinuousOutput(node, out_index, out_size),
                    "node: %s, out_index: %d", NodeCheckerUtils::NodeName(node).c_str(), out_index);
  GE_ASSERT_TRUE(!AddOverflow(start, out_size, end), "start: %" PRId64 ", out_size: %" PRId64 "", start, out_size);
  return SUCCESS;
}

Status ErrorLogPhonySplitNode(const Node *const node) {
  std::stack<const Node *> node_stack;
  node_stack.push(node);
  while (!node_stack.empty()) {
    const auto ps = node_stack.top();
    node_stack.pop();
    NodeCheckerParam param{ps->GetOwnerComputeGraph(), ps, kSpecialNodeTypeNoPaddingContinuousOutput};
    GE_ASSERT_SUCCESS(NodeCheckerUtils::ErrorLogAllOutputs(param));
    GE_ASSERT_TRUE(!ps->GetAllOutDataAnchorsPtr().empty());
    for (const auto peer_in_anchor : ps->GetAllOutDataAnchorsPtr().back()->GetPeerInDataAnchorsPtr()) {
      if (MemLayoutConflictUtil::IsNoPaddingContinuousOutput(peer_in_anchor->GetOwnerNodeBarePtr())) {
        node_stack.push(peer_in_anchor->GetOwnerNodeBarePtr());
        break;
      }
    }
  }
  return SUCCESS;
}

Status NoPaddingContinuousOutputNodeSizeChecker(const NodeCheckerParam &param) {
  if (InputNodeIsPhonySplit(param.node)) {
    return SUCCESS;
  }
  OutDataAnchor *last_ps_out_anchor = nullptr;
  GE_ASSERT_SUCCESS(FindLastOutputPhonySplit(param.node, &last_ps_out_anchor), "node: %s",
                    NodeCheckerUtils::NodeName(param.node).c_str());
  GE_ASSERT_NOTNULL(last_ps_out_anchor);
  int64_t out_offset = 0;
  int64_t out_offset_end = 0;
  GE_ASSERT_SUCCESS(GetNoPaddingOutputOffsetRange(last_ps_out_anchor, out_offset, out_offset_end),
                    "node: %s, out_index: %d",
                    NodeCheckerUtils::NodeName(last_ps_out_anchor->GetOwnerNodeBarePtr()).c_str(),
                    last_ps_out_anchor->GetIdx());

  int64_t continuous_mem_start = 0;
  int64_t continuous_mem_end = 0;
  GE_ASSERT_TRUE(!param.node->GetAllInDataAnchorsPtr().empty(), "node: %s",
                 NodeCheckerUtils::NodeName(param.node).c_str());
  const auto in_data_anchor = param.node->GetAllInDataAnchorsPtr().front();
  GE_ASSERT_NOTNULL(in_data_anchor);
  GE_ASSERT_SUCCESS(
      GetOutputOffsetRange(in_data_anchor->GetPeerOutAnchor().get(), continuous_mem_start, continuous_mem_end),
      "node: %s", NodeCheckerUtils::NodeName(param.node).c_str());

  if (out_offset_end > continuous_mem_end) {
    REPORT_INNER_ERR_MSG("E19999", "nopadding continuous output node memory size check failed. valid memory range: "
        "[%" PRId64 ", %" PRId64 "), last PhonySplit: %s, out_index: %d, out_offset: %" PRId64 ", out_offset_end: "
        "%" PRId64 ", first PhonySplit input node: %s,  out_index: %d", continuous_mem_start, continuous_mem_end,
        NodeCheckerUtils::NodeName(last_ps_out_anchor->GetOwnerNodeBarePtr()).c_str(), last_ps_out_anchor->GetIdx(),
        out_offset, out_offset_end,
        NodeCheckerUtils::NodeName(in_data_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr()).c_str(),
        in_data_anchor->GetPeerOutAnchor()->GetIdx());
    GELOGE(FAILED, "nopadding continuous output node memory size check failed. valid memory range: [%" PRId64 ", "
        "%" PRId64 "), last  PhonySplit: %s, out_index: %d, out_offset: %" PRId64 ", out_offset_end: %" PRId64 ","
        " first PhonySplit input node: %s,  out_index: %d", continuous_mem_start, continuous_mem_end,
        NodeCheckerUtils::NodeName(last_ps_out_anchor->GetOwnerNodeBarePtr()).c_str(), last_ps_out_anchor->GetIdx(),
        out_offset, out_offset_end,
        NodeCheckerUtils::NodeName(in_data_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr()).c_str(),
        in_data_anchor->GetPeerOutAnchor()->GetIdx());
    GE_ASSERT_SUCCESS(ErrorLogPhonySplitNode(param.node));
    return FAILED;
  }
  return SUCCESS;
}

Status NoPaddingContinuousOutputNodeChecker(const NodeCheckerParam &param) {
  GE_ASSERT_SUCCESS(NoPaddingContinuousOutputNodeOffsetChecker(param));
  GE_ASSERT_SUCCESS(NoPaddingContinuousOutputNodeSizeChecker(param));
  return SUCCESS;
}
}  // namespace
REGISTER_SPECIAL_NODE_CHECKER(kSpecialNodeTypeNoPaddingContinuousOutput, NoPaddingContinuousOutputNodeChecker);
}  // namespace ge
