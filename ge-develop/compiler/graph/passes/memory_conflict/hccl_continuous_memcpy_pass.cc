/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/memory_conflict/hccl_continuous_memcpy_pass.h"

#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_type_utils.h"
#include "runtime/rt.h"
#include "checker.h"

namespace {
const int32_t kAnchorAssignRefIndex = 0;
const int32_t kAnchorAssignValueIndex = 1;
const int32_t kAnchorIdentityIndex = 0;

bool IsInputContinuous(const ge::OpDescPtr &op_desc) {
  bool is_input_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
  return is_input_continuous;
}

bool HasMultiPeerNodeRequireInputContinuous(const ge::OutDataAnchorPtr &out_data_anchor) {
  size_t count = 0U;
  for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor == nullptr || peer_in_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    const auto &peer_in_node = peer_in_anchor->GetOwnerNode();
    if (IsInputContinuous(peer_in_node->GetOpDesc())) {
      count++;
    }
  }
  return (count > 1U);
}
}  // namespace
namespace ge {
Status HcclContinuousMemcpyPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Node with nullptr op_desc exist in Param graph:%s, check invalid",
                         graph->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, Node with nullptr op_desc exist in Param graph:%s.",
             graph->GetName().c_str());
      return INTERNAL_ERROR;
    }

    Status ret = ContinuousInputProcess(graph, node);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Call][ContinuousInputProcess] failed, node_name:%s.", node->GetName().c_str());
      return ret;
    }

    ret = P2pmemInputProcess(graph, node);
    if (ret != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Call][P2pmemInputProcess] failed, node_name:%s.", node->GetName().c_str());
      return ret;
    }

  }
  return SUCCESS;
}

// If broadcast input size is bigger than 1, and input from variable,
// cause by broadcast input memory should be continuous,
// another featuremap mem will be allocated for broadcast input.
// In this condition, move data from variable mem to broadcast input featuremap mem will be executed each step.
// In order to avoid move action out of model, use memcpy node instead of move action code.
Status HcclContinuousMemcpyPass::ContinuousInputProcess(const ComputeGraphPtr &graph, const NodePtr node) {
  auto op_desc = node->GetOpDesc();
  if (IsInputContinuous(op_desc)) {
    GELOGI("continuous input op is:%s.", op_desc->GetName().c_str());
    // if input size bigger than one, insert memcpy between var data for support continous mem alloc
    for (auto &hccl_in_anchor : node->GetAllInDataAnchors()) {
      if (hccl_in_anchor == nullptr) {
        continue;
      }
      auto src_out_anchor = hccl_in_anchor->GetPeerOutAnchor();
      if (src_out_anchor == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) input:%d anchor, peer anchor is nullptr, check invalid",
                           node->GetName().c_str(), node->GetType().c_str(),
                           hccl_in_anchor->GetIdx());
        GELOGE(INTERNAL_ERROR, "[Get][PeerOutAnchor] failed, Node:%s(%s) input:%d anchor, peer anchor is nullptr",
               node->GetName().c_str(), node->GetType().c_str(), hccl_in_anchor->GetIdx());
        return INTERNAL_ERROR;
      }

      if (IsDataNode(src_out_anchor->GetOwnerNode()->GetType()) ||
          HasMultiPeerNodeRequireInputContinuous(src_out_anchor)) {
        Status ret = ModifyEdgeConnection(graph, src_out_anchor, hccl_in_anchor);
        if (ret != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Modify][EdgeConnection] between %s and %s failed.",
                 src_out_anchor->GetOwnerNode()->GetName().c_str(), node->GetName().c_str());
          return ret;
        }
      }
    }
  }
  return SUCCESS;
}

// if input is var type, and node input need p2p mem, then memcpy should be insert between the two
Status HcclContinuousMemcpyPass::P2pmemInputProcess(const ComputeGraphPtr &graph, const NodePtr node) {
  auto op_desc = node->GetOpDesc();

  std::vector<int64_t> input_memory_types;
  (void) ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, input_memory_types);

  if (input_memory_types.empty()) {
    return SUCCESS;
  }

  for (uint32_t index = 0; index < input_memory_types.size() && index < op_desc->GetInputsSize(); index++) {
    if (input_memory_types[index] != RT_MEMORY_P2P_DDR) {
      continue;
    }

    GELOGD("p2p input op is:%s.", op_desc->GetName().c_str());
    auto hccl_in_anchor = node->GetInDataAnchor(index);
    if (hccl_in_anchor == nullptr) {
      continue;
    }
    auto src_out_anchor = hccl_in_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) input:%u anchor, peer anchor is nullptr, check invalid",
                         node->GetName().c_str(), node->GetType().c_str(), index);
      GELOGE(INTERNAL_ERROR, "[Get][PeerOutAnchor] failed, Node:%s(%s) input:%u anchor, peer anchor is nullptr",
             node->GetName().c_str(), node->GetType().c_str(), index);
      return INTERNAL_ERROR;
    }

    if (IsDataNode(src_out_anchor->GetOwnerNode()->GetType())) {
      Status ret = ModifyEdgeConnection(graph, src_out_anchor, hccl_in_anchor);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Modify][EdgeConnection] between %s and %s failed.",
               src_out_anchor->GetOwnerNode()->GetName().c_str(), node->GetName().c_str());
        return ret;
      }
    }
  }
  return SUCCESS;
}

bool HcclContinuousMemcpyPass::IsDataNode(const std::string& node_type) const {
  return (node_type == CONSTANTOP) || OpTypeUtils::IsDataNode(node_type) || (node_type == CONSTANT) ||
         OpTypeUtils::IsVarLikeNode(node_type);
}

///
/// @brief Add Identity Node
/// @param [in] ge::ComputeGraphPtr graph
/// @param [in] ge::OutDataAnchorPtr in_node
/// @return ge::NodePtr
///
OpDescPtr HcclContinuousMemcpyPass::CreateIdentityOpDesc(const OutDataAnchorPtr &out_data_anchor) {
  NodePtr pre_node = out_data_anchor->GetOwnerNode();
  OpDescPtr pre_op_desc = pre_node->GetOpDesc();
  if (pre_op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "OpDesc in node is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, OpDesc of pre node is invalid.");
    return nullptr;
  }

  std::string node_name = pre_node->GetName() + "_HcclContinuousMemcpyPass_" + IDENTITY;
  node_name = CheckDuplicateName(node_name);
  OpDescBuilder op_desc_builder(node_name, IDENTITY);
  auto data_desc = pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx());
  auto identity_op_desc = op_desc_builder.AddInput("x", data_desc).AddOutput("y", data_desc).Build();
  if (identity_op_desc == nullptr) {
    return nullptr;
  }
  // because history reason ,this pass can not do work after constant fold so mark it
  (void)AttrUtils::SetBool(identity_op_desc, ATTR_NO_NEED_CONSTANT_FOLDING, false);
  (void)AttrUtils::SetBool(identity_op_desc, ATTR_NAME_CANNOT_BE_DELETED, true);
  return identity_op_desc;
}

///
/// @brief Check duplicate node_name
/// @param [in] std::string& node_name
/// @return std::string
///
std::string HcclContinuousMemcpyPass::CheckDuplicateName(const std::string &node_name) {
  std::string tmp_name = node_name;
  auto iter = node_num_map_.find(tmp_name);
  if (iter != node_num_map_.end()) {
    tmp_name = tmp_name + "_" + std::to_string(iter->second);
    (iter->second)++;
  } else {
    node_num_map_[tmp_name] = 1;
  }
  return tmp_name;
}

///
/// @brief Modify edge connection
/// @param [in] ComputeGraphPtr graph
/// @param [in] OutDataAnchorPtr src_out_anchor
/// @param [in] InDataAnchorPtr hccl_in_anchor
/// @return status
///
Status HcclContinuousMemcpyPass::ModifyEdgeConnection(const ComputeGraphPtr &graph,
                                                      const OutDataAnchorPtr &src_out_anchor,
                                                      const InDataAnchorPtr &hccl_in_anchor) {
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
Status HcclContinuousMemcpyPass::InsertIdentityBeforeHccl(const OutDataAnchorPtr &src_out_anchor,
                                                          const InDataAnchorPtr &hccl_in_anchor) {
  GELOGI("Between op %s and op %s need insert identity op.", src_out_anchor->GetOwnerNode()->GetName().c_str(),
         hccl_in_anchor->GetOwnerNode()->GetName().c_str());
  OpDescPtr identity_op = CreateIdentityOpDesc(src_out_anchor);
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
           hccl_in_anchor->GetOwnerNode()->GetName().c_str(), hccl_in_anchor->GetOwnerNode()->GetType().c_str(),
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
Status HcclContinuousMemcpyPass::InsertAssignAfterBroadcastIfNeed(const ComputeGraphPtr &graph,
                                                                  const OutDataAnchorPtr &var_out_anchor,
                                                                  const InDataAnchorPtr &hccl_in_anchor) {
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

  NodePtr assign_node = CreateAssignNode(graph, var_out_anchor);
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
      REPORT_INNER_ERR_MSG("E19999", "Op:%s(%s) link control to op:%s(%s) failed",
                        assign_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                        assign_out_control_anchor->GetOwnerNode()->GetType().c_str(),
                        in_data_anchor->GetOwnerNode()->GetName().c_str(),
                        in_data_anchor->GetOwnerNode()->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Add][Edge] Op:%s(%s) link control to op:%s(%s) failed",
             assign_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             assign_out_control_anchor->GetOwnerNode()->GetType().c_str(),
             in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetOwnerNode()->GetType().c_str());
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
NodePtr HcclContinuousMemcpyPass::CreateAssignNode(const ComputeGraphPtr &graph,
                                                   const OutDataAnchorPtr &out_data_anchor) {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);
  NodePtr pre_node = out_data_anchor->GetOwnerNode();
  OpDescPtr pre_op_desc = pre_node->GetOpDesc();
  if (pre_op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "OpDesc in node is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, OpDesc of pre node is invalid.");
    return nullptr;
  }

  std::string node_name = pre_node->GetName() + "_" + ASSIGN;
  node_name = CheckDuplicateName(node_name);
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name.c_str(), ASSIGN);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(INTERNAL_ERROR, "[New][OpDesc] failed.");
    return nullptr;
  }
  GELOGI("Create Assign op:%s.", op_desc->GetName().c_str());

  if (!AttrUtils::SetBool(op_desc, ATTR_NEED_COMPILE, true)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NEED_COMPILE.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NEED_COMPILE.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

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


///
/// @brief Clear Status, used for subgraph pass
/// @return SUCCESS
///
Status HcclContinuousMemcpyPass::ClearStatus() {
  node_num_map_.clear();
  return SUCCESS;
}

REG_PASS_OPTION("HcclContinuousMemcpyPass").LEVELS(OoLevel::kO0);
}  // namespace ge
