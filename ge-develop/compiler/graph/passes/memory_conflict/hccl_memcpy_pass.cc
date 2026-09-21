/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/memory_conflict/hccl_memcpy_pass.h"

#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace {
const int32_t kAnchorSize = 1;
const int32_t kAnchorAssignRefIndex = 0;
const int32_t kAnchorAssignValueIndex = 1;
const int32_t kAnchorIdentityIndex = 0;
const std::set<std::string> kShapeComputingOp = {"Shape", "Rank"};

bool IsShapeComputingOp(const std::string &op_type) {
  return kShapeComputingOp.count(op_type) > 0;
}

/*
      op                    op
    /   \         or       /  \
  hccl shape            hccl  reshape
                                 \
                                 shape
 左图中shape，右图中reshape及其下游均为shape计算的分支
*/
bool IsShapeComputingBranch(const ge::NodePtr &sibling_node) {
  if (IsShapeComputingOp(sibling_node->GetType())) {
    return true;
  } else if (sibling_node->GetType() == ge::RESHAPE) {
    const auto out_data_nodes = sibling_node->GetOutDataNodes();
    return std::all_of(out_data_nodes.begin(), out_data_nodes.end(), [](const ge::NodePtr &node) -> bool {
      return IsShapeComputingOp(node->GetType());
    });
  }
  return false;
};
/*
 * 1、如果一个输出同时给到局部写算子和读算子，这是一个读写冲突场景。
      若先读后写，不插入identity，也可以保证精度
      若先写后读，需要插入identity，避免读算子读取的数据是错误的。
      该函数判断兄弟节点的拓扑序是否在hccl算子前面
   2、如果一个输出同时给到局部写算子和shape计算算子，因shape计算不受内存被改写的影响
      此时可以不插入identity
*/
bool NeedInsertIdentity(const ge::OutDataAnchorPtr &src_out_anchor, const ge::InDataAnchorPtr &hccl_in_anchor) {
  const auto hccl_node = hccl_in_anchor->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(hccl_node);
  for (const auto &sibling_in_anchor : src_out_anchor->GetPeerInDataAnchors()) {
    if (sibling_in_anchor == hccl_in_anchor) {
      continue;
    }
    const auto sibling_node = sibling_in_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(sibling_node);
    // 若sibling为新插入算子，其id和src_out_anchor对应节点一样。表示sibling先于hccl执行，并且已经作了保护。此时自己不保护也可以
    if (sibling_node->GetOpDescBarePtr()->GetId() == src_out_anchor->GetOwnerNode()->GetOpDesc()->GetId()) {
      continue;
    }
    if (IsShapeComputingBranch(sibling_node->shared_from_this())) {
      GELOGD("Scope write node [%s][%s] has sibling node [%s][%s] with shape computing branch.",
             hccl_node->GetNamePtr(), hccl_node->GetTypePtr(), sibling_node->GetNamePtr(), sibling_node->GetTypePtr());
      continue;
    }
    if (sibling_node->GetOpDescBarePtr()->GetId() > hccl_node->GetOpDescBarePtr()->GetId()) {
      return true;
    }
  }
  // 临时方案： 此处已经处理过hccl算子的读写冲突，后续读写冲突可以跳过处理
  // 正式方案：将hccl memcpy pass收编到mem rw confilct中
  ge::AttrUtils::SetBool(hccl_in_anchor->GetOwnerNode()->GetOpDesc(), "_skip_rw_conflict", true);
  return false;
}
} // namespace
namespace ge {
Status HcclMemcpyPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  bool is_topo_sorted = false;
  // only one node or empty in graph can skip optimize
  if (graph->GetDirectNodesSize() < 2u) {
    return ge::GRAPH_SUCCESS;
  }
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Node with nullptr op_desc exist in Param graph:%s, check invalid",
                         graph->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, Node with nullptr op_desc exist in Param graph:%s.",
             graph->GetName().c_str());
      return INTERNAL_ERROR;
    }

    bool node_input_mutable = false;
    (void)AttrUtils::GetBool(op_desc, ATTR_NAME_MODIFY_INPUT, node_input_mutable);
    if (!node_input_mutable) {
      continue;
    }
    if (!is_topo_sorted) {
      GE_ASSERT_GRAPH_SUCCESS(graph->TopologicalSorting());
      is_topo_sorted = true;
    }

    Status ret = MutableInputProcess(graph, node);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Call][MutableInputProcess] failed, node_name:%s.", node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

// If node has _input_mutable attr, means input mem may be modified when op execute.
// In order to avoid to affect another op execute with same input when data modified,
// need to inset memcpy node between.
// also works on situation that input is variable or const.
Status HcclMemcpyPass::MutableInputProcess(const ComputeGraphPtr &graph, const NodePtr node) const {
  GELOGI("input mutable hcom op is:%s.", node->GetName().c_str());
  for (auto &hccl_in_anchor : node->GetAllInDataAnchors()) {
    if (hccl_in_anchor == nullptr) {
      continue;
    }
    auto src_out_anchor = hccl_in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(src_out_anchor);
    // Identity needs to be inserted between constant (/variable) and hcomallreduce to avoid constant being cleared.
    if (NotAllowedModifyMem(src_out_anchor->GetOwnerNode()->GetType())) {
      Status ret = ModifyEdgeConnection(graph, src_out_anchor, hccl_in_anchor);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Modify][EdgeConnection] between %s and %s failed.",
               src_out_anchor->GetOwnerNode()->GetName().c_str(), node->GetName().c_str());
        return ret;
      }
    } else {
      int32_t src_out_anchor_size = src_out_anchor->GetPeerInDataAnchors().size();
      if (src_out_anchor_size == kAnchorSize) {
        continue;
      }
      if (!NeedInsertIdentity(src_out_anchor, hccl_in_anchor)) {
        continue;
      }
      GE_ASSERT_GRAPH_SUCCESS(ModifyEdgeConnection(graph, src_out_anchor, hccl_in_anchor),
                              "[Modify][EdgeConnection] between %s and %s failed.",
                              src_out_anchor->GetOwnerNode()->GetName().c_str(), node->GetName().c_str());
    }
  }
  return SUCCESS;
}

bool HcclMemcpyPass::NotAllowedModifyMem(const std::string &node_type) const {
  return (node_type == CONSTANTOP) || (node_type == CONSTANT) || OpTypeUtils::IsVariableNode(node_type);
}

///
/// @brief Add Identity Node
/// @param [in] ge::ComputeGraphPtr graph
/// @param [in] ge::OutDataAnchorPtr in_node
/// @return ge::NodePtr
///
OpDescPtr HcclMemcpyPass::CreateIdentityOpDesc(const OutDataAnchorPtr &out_data_anchor,
                                               const InDataAnchorPtr &hccl_in_anchor) const {
  NodePtr pre_node = out_data_anchor->GetOwnerNode();
  OpDescPtr pre_op_desc = pre_node->GetOpDesc();
  if (pre_op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "OpDesc in node is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, OpDesc of pre node is invalid.");
    return nullptr;
  }
  NodePtr hccl_node = hccl_in_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(hccl_node);

  std::string node_name =
      hccl_node->GetName() + "_HcclMemcpyPass_" + IDENTITY + std::to_string(hccl_in_anchor->GetIdx());
  OpDescBuilder op_desc_builder(node_name, IDENTITY);
  auto data_desc = pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx());
  auto identity_op_desc = op_desc_builder.AddInput("x", data_desc).AddOutput("y", data_desc).Build();
  if (identity_op_desc == nullptr) {
    return nullptr;
  }
  // because history reason ,this pass can not do work after constant fold so mark it
  (void)AttrUtils::SetBool(identity_op_desc, ATTR_NO_NEED_CONSTANT_FOLDING, false);
  return identity_op_desc;
}

///
/// @brief Modify edge connection
/// @param [in] ComputeGraphPtr graph
/// @param [in] OutDataAnchorPtr src_out_anchor
/// @param [in] InDataAnchorPtr hccl_in_anchor
/// @return status
///
Status HcclMemcpyPass::ModifyEdgeConnection(const ComputeGraphPtr &graph, const OutDataAnchorPtr &src_out_anchor,
                                            const InDataAnchorPtr &hccl_in_anchor) const {
  GE_CHECK_NOTNULL(src_out_anchor->GetOwnerNode());
  GE_CHECK_NOTNULL(hccl_in_anchor->GetOwnerNode());

  Status ret = InsertIdentityBeforeHccl(src_out_anchor, hccl_in_anchor);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Add][Identity] failed, var_node:%s, hccl_node:%s.",
           src_out_anchor->GetOwnerNode()->GetName().c_str(),
           hccl_in_anchor->GetOwnerNode()->GetName().c_str());
    return ret;
  }

  ret = InsertAssignAfterBroadcastIfNeed(graph, src_out_anchor, hccl_in_anchor);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Add][Assign] failed, var_node:%s, hccl_node:%s.",
           src_out_anchor->GetOwnerNode()->GetName().c_str(),
           hccl_in_anchor->GetOwnerNode()->GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

///
/// @brief Insert Identity node Between Hccl node and variable
/// @param [in] ComputeGraphPtr graph
/// @param [in] OutDataAnchorPtr src_out_anchor
/// @param [in] InDataAnchorPtr hccl_in_anchor
/// @return status
///
Status HcclMemcpyPass::InsertIdentityBeforeHccl(const OutDataAnchorPtr &src_out_anchor,
                                                const InDataAnchorPtr &hccl_in_anchor) const {
  GELOGI("Between op %s and op %s need insert identity op.", src_out_anchor->GetOwnerNode()->GetName().c_str(),
         hccl_in_anchor->GetOwnerNode()->GetName().c_str());
  auto identity_op = CreateIdentityOpDesc(src_out_anchor, hccl_in_anchor);
  GE_CHECK_NOTNULL(identity_op);

  auto identity_node = GraphUtils::InsertNodeBefore(hccl_in_anchor,
      identity_op, kAnchorIdentityIndex, kAnchorIdentityIndex);
  if (identity_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Op:Fail to insert %s(%s) before %s(%s) on index:%d input anchor.",
                      identity_op->GetName().c_str(), identity_op->GetType().c_str(),
                      hccl_in_anchor->GetOwnerNode()->GetName().c_str(),
                      hccl_in_anchor->GetOwnerNode()->GetType().c_str(),
                      hccl_in_anchor->GetIdx());
    GELOGE(INTERNAL_ERROR, "[Insert][Node] %s(%s) before %s(%s) on index:%d input anchor failed.",
           identity_op->GetName().c_str(), identity_op->GetType().c_str(),
           hccl_in_anchor->GetOwnerNode()->GetName().c_str(),
           hccl_in_anchor->GetOwnerNode()->GetType().c_str(),
           hccl_in_anchor->GetIdx());
    return FAILED;
  }
  return SUCCESS;
}

///
/// @brief Insert assign node after broadcast node and variable to refresh variable data
/// @param [in] ComputeGraphPtr graph
/// @param [in] OutDataAnchorPtr var_out_anchor
/// @param [in] InDataAnchorPtr hccl_in_anchor
/// @return status
///
Status HcclMemcpyPass::InsertAssignAfterBroadcastIfNeed(const ComputeGraphPtr &graph,
                                                        const OutDataAnchorPtr &var_out_anchor,
                                                        const InDataAnchorPtr &hccl_in_anchor) const {
  if (hccl_in_anchor->GetOwnerNode()->GetType() != HCOMBROADCAST) {
    GELOGD("%s not broadcast, no need to insert assign node", hccl_in_anchor->GetOwnerNode()->GetName().c_str());
    return SUCCESS;
  }

  if (!OpTypeUtils::IsVarLikeNode(var_out_anchor->GetOwnerNode()->GetType())) {
    GELOGD("%s not variable, no need to insert assign node", var_out_anchor->GetOwnerNode()->GetName().c_str());
    return SUCCESS;
  }

  GELOGI("after op %s and op %s need insert assign op.", var_out_anchor->GetOwnerNode()->GetName().c_str(),
         hccl_in_anchor->GetOwnerNode()->GetName().c_str());

  for (auto peer_in_anchor : var_out_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor->GetOwnerNode()->GetType() == ASSIGN) {
      GELOGD("variable %s out assign node is exist.", var_out_anchor->GetOwnerNode()->GetName().c_str());
      return SUCCESS;
    }
  }

  NodePtr assign_node = CreateAssignNode(graph, var_out_anchor, hccl_in_anchor);
  GE_CHECK_NOTNULL(assign_node);

  OutDataAnchorPtr hccl_out_anchor = hccl_in_anchor->GetOwnerNode()->GetOutDataAnchor(hccl_in_anchor->GetIdx());
  GE_CHECK_NOTNULL(hccl_out_anchor);

  Status ret = hccl_out_anchor->LinkTo(assign_node->GetInDataAnchor(kAnchorAssignValueIndex));
  GE_ASSERT_SUCCESS(ret, "Op:%s(%s) out index:%d link to op:%s(%s) in index:%d failed",
                    hccl_out_anchor->GetOwnerNode()->GetName().c_str(),
                    hccl_out_anchor->GetOwnerNode()->GetType().c_str(), hccl_out_anchor->GetIdx(),
                    assign_node->GetName().c_str(), assign_node->GetType().c_str(), kAnchorAssignValueIndex);

  ret = var_out_anchor->LinkTo(assign_node->GetInDataAnchor(kAnchorAssignRefIndex));
  GE_ASSERT_SUCCESS(ret, "Op:%s(%s) out index:%d link to op:%s(%s) in index:%d failed",
                    var_out_anchor->GetOwnerNode()->GetName().c_str(),
                    var_out_anchor->GetOwnerNode()->GetType().c_str(), var_out_anchor->GetIdx(),
                    assign_node->GetName().c_str(), assign_node->GetType().c_str(), kAnchorAssignRefIndex);

  // add control edge between assign node and node after broadcast node
  OutControlAnchorPtr assign_out_control_anchor = assign_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(assign_out_control_anchor);

  for (auto in_data_anchor : hccl_out_anchor->GetPeerInDataAnchors()) {
    if (in_data_anchor->GetOwnerNode()->GetName() == assign_node->GetName()) {
      continue;
    }
    ret = assign_out_control_anchor->LinkTo(in_data_anchor->GetOwnerNode()->GetInControlAnchor());
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Op:%s(%s) out index:%d link to op:%s(%s) in index:%d failed",
                        assign_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                        assign_out_control_anchor->GetOwnerNode()->GetType().c_str(),
                        assign_out_control_anchor->GetIdx(),
                        in_data_anchor->GetOwnerNode()->GetName().c_str(),
                        in_data_anchor->GetOwnerNode()->GetType().c_str(),
                        in_data_anchor->GetIdx());
      GELOGE(INTERNAL_ERROR, "[Add][Edge] Op:%s(%s) out index:%d link to op:%s(%s) in index:%d failed",
             assign_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             assign_out_control_anchor->GetOwnerNode()->GetType().c_str(),
             assign_out_control_anchor->GetIdx(),
             in_data_anchor->GetOwnerNode()->GetName().c_str(),
             in_data_anchor->GetOwnerNode()->GetType().c_str(),
             in_data_anchor->GetIdx());
      return FAILED;
    }
  }

  for (auto in_control_anchor : hccl_out_anchor->GetOwnerNode()->GetOutControlAnchor()->GetPeerInControlAnchors()) {
    if (in_control_anchor->GetOwnerNode()->GetName() == assign_node->GetName()) {
      continue;
    }
    ret = assign_out_control_anchor->LinkTo(in_control_anchor);
      if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Op:%s(%s) link control to op:%s(%s) failed",
                        assign_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                        assign_out_control_anchor->GetOwnerNode()->GetType().c_str(),
                        in_control_anchor->GetOwnerNode()->GetName().c_str(),
                        in_control_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Add][Edge] Op:%s(%s) link control to op:%s(%s) failed",
             assign_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             assign_out_control_anchor->GetOwnerNode()->GetType().c_str(),
             in_control_anchor->GetOwnerNode()->GetName().c_str(),
             in_control_anchor->GetOwnerNode()->GetType().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

///
/// @brief create assign Node, add to graph
/// @param [in] ge::ComputeGraphPtr graph
/// @param [in] ge::OutDataAnchorPtr variable node out anchor
/// @return ge::NodePtr
///
NodePtr HcclMemcpyPass::CreateAssignNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor,
                                         const InDataAnchorPtr &hccl_in_anchor) const {
  NodePtr pre_node = out_data_anchor->GetOwnerNode();
  OpDescPtr pre_op_desc = pre_node->GetOpDesc();
  if (pre_op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "OpDesc in node is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, OpDesc of pre node is invalid.");
    return nullptr;
  }

  NodePtr hccl_node = hccl_in_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(hccl_node);

  std::string node_name = hccl_node->GetName() + "_" + ASSIGN + "_" + std::to_string(hccl_in_anchor->GetIdx());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name.c_str(), ASSIGN);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(INTERNAL_ERROR, "[New][OpDesc] failed.");
    return nullptr;
  }
  GELOGI("Create Assign op:%s.", op_desc->GetName().c_str());

  graphStatus ret = op_desc->AddInputDesc("ref", pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add input desc to op:%s(%s) failed, name:ref",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] to op:%s(%s) failed, name:ref",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  ret = op_desc->AddInputDesc("value", pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add input desc to op:%s(%s) failed, name:value",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] to op:%s(%s) failed, name:value",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  ret = op_desc->AddOutputDesc("ref", pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx()));
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add output desc to op:%s(%s) failed, name:ref",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] to op:%s(%s) failed, name:ref",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  NodePtr assign_node = graph->AddNode(op_desc);
  if (assign_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Node] %s(%s) to graph:%s failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }

  return assign_node;
}

REG_PASS_OPTION("HcclMemcpyPass").LEVELS(OoLevel::kO0);
}  // namespace ge
