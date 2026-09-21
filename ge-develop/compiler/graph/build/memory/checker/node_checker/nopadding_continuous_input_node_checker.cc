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
#include "ge_common/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "common/checker.h"
#include "graph/utils/tensor_utils.h"
#include "math_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
/*
 * 在分配内存前需要校验，输入节点上ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS and
 * ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION 只能有一个种属性
 */
bool IsUseOffsetList(const Node *const node) {
  const auto in_nodes = node->GetInDataNodes();
  GE_ASSERT_TRUE(!in_nodes.empty());
  std::vector<int64_t> offset_list = {};
  (void)AttrUtils::GetListInt(in_nodes.at(0)->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS,
                              offset_list);
  return !offset_list.empty();
}

/*
 * 在分配内存前需要校验，每个输入节点上都有这个属性，或者都没有这个属性
 */
Status GetOutputOffsetForContinuous(const Node *const node, const int32_t out_index, int64_t &offset) {
  std::vector<int64_t> offset_list = {};
  const bool has_offset_list =
      AttrUtils::GetListInt(node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  GE_ASSERT_TRUE(has_offset_list && (!offset_list.empty()), "expect node %s has %s attr.",
                 NodeCheckerUtils::NodeName(node).c_str(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS.c_str());
  GE_ASSERT_TRUE(static_cast<size_t>(out_index) < offset_list.size(), "out_index: %d, offset_list size: %zu, node: %s",
                 out_index, offset_list.size(), NodeCheckerUtils::NodeName(node).c_str());
  offset = offset_list.at(out_index);
  return SUCCESS;
}

Status GetExpectOffsetWithOutputOffsetList(const Node *const peer_node, const int32_t out_index,
                                           const int64_t base_offset, int64_t &expect_offset) {
  int64_t continuous_offset = 0;
  GE_ASSERT_SUCCESS(GetOutputOffsetForContinuous(peer_node, out_index, continuous_offset),
                    "peer_node: %s, out_index: %d", NodeCheckerUtils::NodeName(peer_node).c_str(), out_index);
  GE_ASSERT_TRUE(!AddOverflow(base_offset, continuous_offset, expect_offset),
                 "base_offset: %" PRId64 ", continuous_offset: %" PRId64 "", base_offset, continuous_offset);
  return SUCCESS;
}

// 只能在use_offset_list为true的时候才调用
Status GetBaseOffsetForOffsetList(const Node *const node, int64_t &base_offset) {
  GE_ASSERT_TRUE(!node->GetAllInDataAnchorsPtr().empty());
  const auto first_in_data_anchor = node->GetAllInDataAnchorsPtr().at(0);

  GE_ASSERT_NOTNULL(first_in_data_anchor->GetPeerOutAnchor());
  const auto peer_node = first_in_data_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(peer_node);
  GE_ASSERT_NOTNULL(peer_node->GetOpDescBarePtr());
  const auto out_index = first_in_data_anchor->GetPeerOutAnchor()->GetIdx();

  int64_t offset = 0;
  GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(peer_node->GetOpDescBarePtr(), out_index, offset));

  int64_t out_offset_for_continuous = 0;
  GE_ASSERT_SUCCESS(GetOutputOffsetForContinuous(peer_node, out_index, out_offset_for_continuous));
  GE_ASSERT_TRUE(offset >= out_offset_for_continuous, "offset: %" PRId64 ", out_offset_for_continuous: %" PRId64 "",
                 offset, out_offset_for_continuous);
  base_offset = offset - out_offset_for_continuous;
  return SUCCESS;
}

/*
 * 计算逻辑地址有两种方式，
 * 方式一：使用上一个节点的逻辑地址+偏移（ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION或者上一个节点非对齐内存大小)，
 * 方式二：使用offset list，列表中的offset都是相对于第1个节点的（ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS）
 */
Status NoPaddingContinuousInputNodeOffsetChecker(const NodeCheckerParam &param) {
  int64_t expect_offset = -1;
  bool use_offset_list = IsUseOffsetList(param.node);
  int64_t base_offset = 0;
  if (use_offset_list) {
    GE_ASSERT_SUCCESS(GetBaseOffsetForOffsetList(param.node, base_offset), "node: %s",
                      NodeCheckerUtils::NodeName(param.node).c_str());
  }
  for (const auto &in_data_anchor : param.node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(in_data_anchor->GetPeerOutAnchor());
    const auto peer_node = in_data_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(peer_node);
    GE_ASSERT_NOTNULL(peer_node->GetOpDescBarePtr());
    const auto out_index = in_data_anchor->GetPeerOutAnchor()->GetIdx();

    int64_t offset = 0;
    int64_t stride = 0;
    GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(peer_node->GetOpDescBarePtr(), out_index, offset));
    if (!use_offset_list) {
      GE_ASSERT_SUCCESS(
          NodeCheckerUtils::GetStrideForNoPaddingContinuousInput(param.node, peer_node, out_index, stride));
      if (expect_offset == -1) {
        GE_ASSERT_TRUE(!AddOverflow(offset, stride, expect_offset), "offset: %" PRId64 ", len: %" PRId64 "", offset,
                       stride);
        continue;
      }
    } else {
      GE_ASSERT_SUCCESS(GetExpectOffsetWithOutputOffsetList(peer_node, out_index, base_offset, expect_offset),
                        "node: %s, input_index: %d", NodeCheckerUtils::NodeName(param.node).c_str(),
                        in_data_anchor->GetIdx());
    }

    if (expect_offset != offset) {
      REPORT_INNER_ERR_MSG("E19999", "nopadding continuous input node memory check failed. node: %s, input_index: %d,"
          " input node %s, out_index: %d, offset: %" PRId64 ", expect_offset: %" PRId64 ", use_offset_list: %d,"
          " base_offset: %" PRId64 "", NodeCheckerUtils::NodeName(param.node).c_str(), in_data_anchor->GetIdx(),
          NodeCheckerUtils::NodeName(peer_node).c_str(), out_index, offset, expect_offset, use_offset_list,
          base_offset);
      GELOGE(FAILED, "nopadding continuous input node memory check failed. node: %s, input_index: %d, input node %s, "
          "out_index: %d, offset: %" PRId64 ", expect_offset: %" PRId64 ", use_offset_list: %d, base_offset: "
          "%" PRId64 "", NodeCheckerUtils::NodeName(param.node).c_str(), in_data_anchor->GetIdx(),
          NodeCheckerUtils::NodeName(peer_node).c_str(), out_index, offset, expect_offset, use_offset_list,
          base_offset);
      GE_ASSERT_SUCCESS(NodeCheckerUtils::ErrorLogAllInputs(param));
      return FAILED;
    }
    if (!use_offset_list) {
      GE_ASSERT_TRUE(!AddOverflow(offset, stride, expect_offset), "offset: %" PRId64 ", len: %" PRId64 "", offset,
                     stride);
    }
  }
  return SUCCESS;
}

Status FindLastRealInputNode(const Node *const node, OutDataAnchor **last_real_out_anchor,
                             InDataAnchor **peer_in_anchor) {
  std::stack<const Node *> node_stack;
  node_stack.push(node);

  while (!node_stack.empty()) {
    const auto pc = node_stack.top();
    node_stack.pop();
    GE_ASSERT_TRUE(!pc->GetAllInDataAnchorsPtr().empty(), "node: %s", NodeCheckerUtils::NodeName(pc).c_str());
    *peer_in_anchor = pc->GetAllInDataAnchorsPtr().back();
    GE_ASSERT_NOTNULL(*peer_in_anchor);
    *last_real_out_anchor = pc->GetAllInDataAnchorsPtr().back()->GetPeerOutAnchor().get();
    GE_ASSERT_NOTNULL(*last_real_out_anchor);
    const auto last_in_node = pc->GetAllInDataAnchorsPtr().back()->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(last_in_node);
    if (MemLayoutConflictUtil::IsNoPaddingContinuousInput(last_in_node)) {
      node_stack.push(last_in_node);
    }
  }
  return SUCCESS;
}

Status ErrorLogAllPhonyConcat(const Node *const node) {
  std::stack<const Node *> node_stack;
  node_stack.push(node);

  while (!node_stack.empty()) {
    const auto pc = node_stack.top();
    node_stack.pop();
    NodeCheckerParam param{node->GetOwnerComputeGraph(), pc, kSpecialNodeTypeNoPaddingContinuousInput};
    GE_ASSERT_SUCCESS(NodeCheckerUtils::ErrorLogAllInputs(param), "pc: %s", NodeCheckerUtils::NodeName(pc).c_str());
    GE_ASSERT_TRUE(!pc->GetAllInDataAnchorsPtr().empty(), "pc: %s", NodeCheckerUtils::NodeName(pc).c_str());
    GE_ASSERT_NOTNULL(pc->GetAllInDataAnchorsPtr().back()->GetPeerOutAnchor());
    const auto last_in_node = pc->GetAllInDataAnchorsPtr().back()->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(last_in_node);
    if (MemLayoutConflictUtil::IsNoPaddingContinuousInput(last_in_node)) {
      node_stack.push(last_in_node);
    }
  }
  return SUCCESS;
}

// PhonyConcat means no padding continuous input node
bool OutputNodeIsPhonyConcat(const Node *const node) {
  if (!node->GetAllOutDataAnchorsPtr().empty()) {
    for (const auto peer_in_anchor : node->GetAllOutDataAnchorsPtr().back()->GetPeerInDataAnchorsPtr()) {
      if (MemLayoutConflictUtil::IsNoPaddingContinuousInput(peer_in_anchor->GetOwnerNodeBarePtr())) {
        return true;
      }
    }
  }
  return false;
}

Status GetNoAlignedSize(const Node *const node, const int32_t out_index, int64_t &no_aligned_size) {
  const auto &output_desc = node->GetOpDescBarePtr()->GetOutputDesc(out_index);
  const Format out_format = output_desc.GetFormat();
  const DataType data_type = output_desc.GetDataType();
  const auto &shape = output_desc.GetShape();
  GE_ASSERT_SUCCESS(ge::TensorUtils::CalcTensorMemSize(shape, out_format, data_type, no_aligned_size));
  GE_ASSERT_TRUE(no_aligned_size >= 0,
                 "After calculating, tensor memory size:%" PRId64 " invalid, less than 0. "
                 "new shape:%s, format:%s, dtype:%s, maybe has dynamic shape",
                 no_aligned_size, shape.ToString().c_str(), TypeUtils::FormatToSerialString(out_format).c_str(),
                 TypeUtils::DataTypeToSerialString(data_type).c_str());
  return SUCCESS;
}

Status NoPaddingContinuousInputNodeSizeChecker(const NodeCheckerParam &param) {
  if (OutputNodeIsPhonyConcat(param.node)) {
    return SUCCESS;
  }
  OutDataAnchor *last_real_out_anchor = nullptr;
  InDataAnchor *peer_in_anchor = nullptr;
  GE_ASSERT_SUCCESS(FindLastRealInputNode(param.node, &last_real_out_anchor, &peer_in_anchor),
                    "node: %s, find last input node failed.", NodeCheckerUtils::NodeName(param.node).c_str());

  const auto last_in_node = last_real_out_anchor->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(last_in_node);
  const auto out_index = last_real_out_anchor->GetIdx();
  int64_t out_offset;
  GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(last_in_node->GetOpDescBarePtr(), out_index, out_offset),
                    "get offset failed, last_in_node: %s, index: %d", NodeCheckerUtils::NodeName(last_in_node).c_str(),
                    out_index);

  int64_t write_len;
  GE_ASSERT_SUCCESS(GetNoAlignedSize(last_in_node, out_index, write_len),
                    "get write length failed, last_in_node: %s, index: %d",
                    NodeCheckerUtils::NodeName(last_in_node).c_str(), out_index);
  int64_t out_offset_end;
  GE_ASSERT_TRUE(!AddOverflow(out_offset, write_len, out_offset_end), "offset: %" PRId64 ", len: %" PRId64 "",
                 out_offset, write_len);

  int64_t continuous_mem_base;
  GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(param.node->GetOpDescBarePtr(), 0, continuous_mem_base),
                    "get offset failed, node: %s, index: %d", NodeCheckerUtils::NodeName(param.node).c_str(), 0);
  int64_t continuous_mem_size;
  GE_ASSERT_SUCCESS(TensorUtils::GetSize(param.node->GetOpDescBarePtr()->GetOutputDesc(0), continuous_mem_size),
                    "get size failed, node: %s", NodeCheckerUtils::NodeName(param.node).c_str());
  int64_t max_offset;
  GE_ASSERT_TRUE(!AddOverflow(continuous_mem_base, continuous_mem_size, max_offset),
                 "offset: %" PRId64 ", len: %" PRId64 "", continuous_mem_base, continuous_mem_size);
  if (max_offset < out_offset_end) {
    REPORT_INNER_ERR_MSG("E19999", "nopadding continuous input node memory size check failed. valid offset[%" PRId64 ","
        " %" PRId64 "), last real input node out_offset: %" PRId64 ", out_offset_end: %" PRId64 ", last real input"
        " node: %s, phony concat node: %s", continuous_mem_base, max_offset, out_offset, out_offset_end,
        NodeCheckerUtils::NodeName(last_in_node).c_str(), NodeCheckerUtils::NodeName(param.node).c_str());
    GELOGE(FAILED, "nopadding continuous input node memory size check failed. valid offset[%" PRId64 ", %" PRId64 "),"
        " last real input node out_offset: %" PRId64 ", out_offset_end: %" PRId64 ", last real input node: %s,"
        " phony concat node: %s", continuous_mem_base, max_offset, out_offset, out_offset_end,
        NodeCheckerUtils::NodeName(last_in_node).c_str(), NodeCheckerUtils::NodeName(param.node).c_str());
    GE_ASSERT_SUCCESS(ErrorLogAllPhonyConcat(param.node), "node: %s", NodeCheckerUtils::NodeName(param.node).c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status NoPaddingContinuousInputNodeChecker(const NodeCheckerParam &param) {
  GE_ASSERT_SUCCESS(NoPaddingContinuousInputNodeOffsetChecker(param));
  GE_ASSERT_SUCCESS(NoPaddingContinuousInputNodeSizeChecker(param));
  return SUCCESS;
}
}  // namespace
REGISTER_SPECIAL_NODE_CHECKER(kSpecialNodeTypeNoPaddingContinuousInput, NoPaddingContinuousInputNodeChecker);
}  // namespace ge
