/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/format_optimize/cast_remove_pass.h"
#include <string>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "common/op/transop_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"
#include "common/checker.h"
#include "base/err_msg.h"

namespace ge {
namespace {
// 获取输出有连边的anchor
// 入参中的node只有1个消费者
OutDataAnchorPtr GetOutDataAnchorWithConsumer(const NodePtr &node) {
  auto out_node_2_in_anchors = node->GetOutDataNodesAndAnchors();
  GE_ASSERT_TRUE(out_node_2_in_anchors.size() == 1u);
  auto out_node_2_in_anchor = *out_node_2_in_anchors.begin();
  return out_node_2_in_anchor.second->GetPeerOutAnchor();
}

GeTensorDescPtr MutableOutTensorDescWithConsumer(const NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto end_out_data_anchor = GetOutDataAnchorWithConsumer(node);
  GE_ASSERT_NOTNULL(end_out_data_anchor);
  return op_desc->MutableOutputDesc(end_out_data_anchor->GetIdx());
}
} // namespace
Status CastRemovePass:: Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Param [node] must not be null.");
    return PARAM_INVALID;
  }
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param op_desc of node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] OpDesc of param [node] must not be null.");
    return PARAM_INVALID;
  }

  if (IsRedundantCastNode(node)) {
    GE_CHK_STATUS_RET(IsolateAndDeleteNode(node, {0}), "Failed to remove redundant Cast node:%s", node->GetName().c_str());
    GELOGI("Successfully remove redundant Cast node %s, datatype %s.", node->GetName().c_str(),
      TypeUtils::DataTypeToSerialString(node->GetOpDesc()->GetInputDesc(0).GetDataType()).c_str());
    return SUCCESS;
  }

  // begin with not trans op, and only has one out data node
  if (TransOpUtil::IsTransOp(node) || node->GetOutDataNodes().size() != 1) {
    return SUCCESS;
  }

  std::vector<NodePtr> nodes_to_fuse;
  NodePtr end_node = GetTheEndNode(node, nodes_to_fuse);
  if (nodes_to_fuse.empty()) {
    return SUCCESS;
  }
  GE_ASSERT_NOTNULL(end_node);
  OpDescPtr end_op_desc = end_node->GetOpDesc();
  if (end_op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "op_desc of end_node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] OpDesc of end node must not be null.");
    return PARAM_INVALID;
  }

  if (!CheckPrecisionLoss(nodes_to_fuse)) {
    return SUCCESS;
  }

  DataType type = DT_UNDEFINED;
  if (!HasSameDataType(node, end_node, type)) {
    return SUCCESS;
  }
  if (RemoveCast(type, nodes_to_fuse) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

bool CastRemovePass::CheckPrecisionLoss(const std::vector<NodePtr> &nodes_to_fuse) const {
  for (const NodePtr &node : nodes_to_fuse) {
    if (TransOpUtil::IsPrecisionLoss(node)) {
      return false;
    }
  }
  return true;
}

bool CastRemovePass::HasSameDataType(const NodePtr &begin_node, const NodePtr &end_node, DataType &type) const {
  const auto &begin_op_desc = begin_node->GetOpDesc();
  const auto &end_op_desc = end_node->GetOpDesc();
  if (begin_op_desc->GetName() == end_op_desc->GetName()) {
    return false;
  }
  // end op 一定是转换算子，因此取输出0即可
  auto end_out_desc = end_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(end_out_desc);
  DataType end_out_datatype = end_out_desc->GetDataType();

  const auto begin_out_desc = MutableOutTensorDescWithConsumer(begin_node);
  GE_ASSERT_NOTNULL(begin_out_desc);
  DataType begin_out_datatype = begin_out_desc->GetDataType();
  if (begin_out_datatype == end_out_datatype) {
    type = begin_out_datatype;
    return true;
  }
  return false;
}

// op1->TransData->Cast->TransposeD->Cast->TransData->op2
// change to be
// op1->TransData->TransposeD->TransData->op2
Status CastRemovePass::RemoveCast(DataType &type, std::vector<NodePtr> &nodes_to_fuse) {
  std::string cast_name;
  for (NodePtr &node : nodes_to_fuse) {
    if (node->GetType() == CAST) {
      GELOGI("CastRemovePass, remove Cast %s.", node->GetName().c_str());
      cast_name = node->GetName();
      if (IsolateAndDeleteNode(node, {0}) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Isolate and delete node:%s(%s) failed",
                          node->GetName().c_str(), node->GetType().c_str());
        GELOGE(FAILED, "[IsolateAndDelete][Node] %s failed.", node->GetName().c_str());
        return FAILED;
      }
    }
  }

  if (cast_name.empty()) {
    return SUCCESS;
  }
  for (auto &node : nodes_to_fuse) {
    if (node->GetType() == CAST) {
      continue;
    }
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Find nullptr op_desc in node, check invalid");
      GELOGE(FAILED, "[Get][OpDesc] OpDesc must not be null.");
      return FAILED;
    }

    // change node name for recompile cache, will be abandoned in April
    std::string new_node_name = cast_name + op_desc->GetName();
    op_desc->SetName(new_node_name);
    auto in_desc = op_desc->MutableInputDesc(0);
    auto out_desc = op_desc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(in_desc);
    GE_ASSERT_NOTNULL(out_desc);
    in_desc->SetDataType(type);
    out_desc->SetDataType(type);
    GELOGI("CastRemovePass, change %s %s input output datatype to be %s.", node->GetName().c_str(),
           node->GetType().c_str(), TypeUtils::DataTypeToSerialString(type).c_str());
  }
  return SUCCESS;
}

NodePtr CastRemovePass::GetTheEndNode(NodePtr begin_node, std::vector<NodePtr> &nodes_to_fuse) const {
  while (begin_node->GetOutDataNodes().size() == 1) {
    auto out_node = begin_node->GetOutDataNodes().at(0);
    OutDataAnchorPtr out_data_anchor = GetOutDataAnchorWithConsumer(begin_node);
    GE_ASSERT_NOTNULL(out_data_anchor);
    auto peer_in_anchor = out_data_anchor->GetFirstPeerAnchor();
    GE_ASSERT_NOTNULL(peer_in_anchor);
    if (!TransOpUtil::IsTransOp(out_node) ||
        TransOpUtil::GetTransOpDataIndex(out_node) != peer_in_anchor->GetIdx()) {
      return begin_node;  // when seen not trans op
    }
    begin_node = out_node;
    nodes_to_fuse.emplace_back(begin_node);
  }
  return begin_node;  // when seen branch
}

// Cast1(FP16->FP16)
bool CastRemovePass::IsRedundantCastNode(const NodePtr &node) const {
  if (node->GetType() != CAST) {
    return false;
  }

  const OpDescPtr op_desc = node->GetOpDesc();
  const auto &input_desc = op_desc->MutableInputDesc(0);
  GE_ASSERT_NOTNULL(input_desc);
  const auto &output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  return input_desc->GetDataType() == output_desc->GetDataType();
}

REG_PASS_OPTION("CastRemovePass").LEVELS(OoLevel::kO3);
}  // namespace ge
