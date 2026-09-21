/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/memory_conflict/atomic_addr_clean_pass.h"

#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <thread>

#include "framework/common/ge_inner_error_codes.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"
#include "api/gelib/gelib.h"
#include "graph/ge_context.h"
#include "graph/common/trans_op_creator.h"

namespace {
constexpr const char *kCleanSeparately = "1";
}
namespace ge {
Status AtomicAddrCleanPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);

  std::string atomic_clean_policy;
  bool need_clean_separately = (GetContext().GetOption(ge::ATOMIC_CLEAN_POLICY, atomic_clean_policy) == SUCCESS) &&
                               (atomic_clean_policy == kCleanSeparately);

  GELOGD("AtomicAddrCleanPass begin, need_clean_separately=%d", static_cast<int32_t>(need_clean_separately));
  // 1.Recoginze atomic and loop mark
  std::vector<NodePtr> atomic_node_vec;
  for (NodePtr &node : graph->GetDirectNode()) {
    if (IsNeedCleanNode(node, need_clean_separately)) {
      atomic_node_vec.push_back(node);
    }
    if (!is_loop_graph_ && node->GetType() == LOOPCOND) {
      is_loop_graph_ = true;
    }
  }

  if (!need_gentask_atomic_node_.empty()) {
    GE_CHK_STATUS_RET(CallCompileOp(need_gentask_atomic_node_));
  }

  if (atomic_node_vec.empty()) {
    GELOGD("There is no atomic node. Ignore atomicAddrClean pass.");
    return SUCCESS;
  }

  bool is_unknown_graph = graph->GetGraphUnknownFlag();
  if (is_unknown_graph) {
    IsNeedCallCompileForUnknownGraph(atomic_node_vec);
    GELOGD("Graph[%s] is unknown graph, only know nodes will call fe interface to compile op.",
           graph->GetName().c_str());
    GE_CHK_STATUS_RET(CallCompileOp(atomic_node_vec));
    return SUCCESS;
  }

  // 2.Insert clean node and link to atomic node
  Status ret;
  if (is_loop_graph_) {
    ret = HandleLoopGraph(graph, atomic_node_vec);
    if (ret != SUCCESS) {
      return ret;
    }
  } else {
    ret = HandleNormalGraph(graph, atomic_node_vec);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  GELOGD("AtomicAddrCleanPass end");
  return SUCCESS;
}

void AtomicAddrCleanPass::IsNeedCallCompileForUnknownGraph(std::vector<NodePtr> &atomic_node_vec) const {
  auto iter = atomic_node_vec.cbegin();
  while (iter != atomic_node_vec.cend()) {
    bool is_unknown_node = false;
    (void)ge::NodeUtils::GetNodeUnknownShapeStatus(**iter, is_unknown_node);
    if (is_unknown_node && !IsHcomAtomicNode(*iter)) {
      iter = atomic_node_vec.erase(iter);
    } else {
      iter++;
    }
  }
}

// just hccl may mark atomic from ops kernel now, and hccl's atomic if for all input
bool AtomicAddrCleanPass::CheckAtomicFromOpsKernel(const NodePtr &node) {
  // 1.Check if isAtomic attrs exist for HCOM
  std::shared_ptr<GELib> instance_ptr = GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGW("GELib not initialized, atomic from ops kernel judge false, node_name: %s", node->GetName().c_str());
    return false;
  }

  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  std::vector<OpInfo> op_info_vec = ops_kernel_manager.GetOpsKernelInfo(node->GetType());
  for (const auto &op_info : op_info_vec) {
    if (op_info.isAtomic) {
      // check peer input is DATA
      for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
        if (in_data_anchor->GetPeerOutAnchor() != nullptr &&
            in_data_anchor->GetPeerOutAnchor()->GetOwnerNode() != nullptr) {
          auto peer_in_node = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
          if (OpTypeUtils::IsDataNode(peer_in_node->GetType())) {
            GELOGI("Recognized atomic op %s from %s engine and input is DATA.", node->GetName().c_str(),
                   op_info.engine.c_str());
            return false;
          }
        }
      }
      GELOGI("Recognized atomic op %s from %s engine.", node->GetName().c_str(), op_info.engine.c_str());
      hcom_node_vec_.push_back(node);
      return true;
    }
  }
  return false;
}

bool AtomicAddrCleanPass::IsOutputIndexPeerInputAtomic(const NodePtr &node, int64_t output_index) {
  auto out_data_anchor = node->GetAllOutDataAnchors().at(output_index);
  if (out_data_anchor == nullptr) {
    return false;
  }

  for (auto input_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    auto output_node = input_anchor->GetOwnerNode();
    // just hccl may mark atomic from ops kernel now, and hccl's atomic if for all input
    // hccl's attr ATOMIC_ATTR_INPUT_INDEX mark on CalcOpRunningParam, can't be get here
    if (CheckAtomicFromOpsKernel(output_node)) {
      return true;
    }
  }
  return false;
}

bool AtomicAddrCleanPass::CheckSkipInsertInLoopGraph(const NodePtr &node) {
  OpDescPtr op_desc = node->GetOpDesc();
  std::map<std::string, std::map<int64_t, int64_t>> atomic_workspace_index_size;
  bool has_atomic_input = op_desc->HasAttr(ATOMIC_ATTR_INPUT_INDEX);
  bool has_atomic_output = op_desc->HasAttr(ATOMIC_ATTR_OUTPUT_INDEX);
  atomic_workspace_index_size = op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_index_size);
  if (!has_atomic_input && has_atomic_output && atomic_workspace_index_size.empty()) {
    std::vector<int64_t> atomic_output_index;
    (void) ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
    bool is_all_output_peer_also_atomic = true;
    for (const auto &output_index : atomic_output_index) {
      if (!IsOutputIndexPeerInputAtomic(node, output_index)) {
        is_all_output_peer_also_atomic = false;
        break;
      }
    }
    if (is_all_output_peer_also_atomic) {
      GELOGI("all out peer node input atomic, skip this out atomic process, node name: %s", node->GetName().c_str());
      return true;
    }
  }
  return false;
}

Status AtomicAddrCleanPass::HandleLoopGraph(ComputeGraphPtr &graph, const std::vector<NodePtr> &atomic_node_vec) {
  // Loop graph , insert clean node follow atomic node
  int32_t index = 0;
  for (const auto &node : atomic_node_vec) {
    // ATOMIC - HCCL, the node do not need atomic clen when the next node is hccl
    if (CheckSkipInsertInLoopGraph(node) || IsAllConnectHcclNode(node)) {
      continue;
    }

    // Insert atomic memset op
    NodePtr clean_addr_node = InsertAtomicMemsetNode(graph);
    if (clean_addr_node == nullptr) {
      GELOGE(FAILED, "[Insert][AtomicAddrCleanNode] to graph:%s failed. Ignore atomicAddrClean pass.",
             graph->GetName().c_str());
      return FAILED;
    }

    GE_CHECK_NOTNULL(clean_addr_node->GetOpDesc());
    std::string node_name = clean_addr_node->GetOpDesc()->GetName();
    std::ostringstream oss;
    oss << node_name << index;
    node_name = oss.str();
    clean_addr_node->GetOpDesc()->SetName(node_name);  // [Cascade Pointer]
    GELOGD("Inserted atomic clean node name is %s", node_name.c_str());

    auto ret = LinkToAtomicNode(node, clean_addr_node);
    if (ret != SUCCESS) {
      GELOGE(ret,
             "[Call][LinkToAtomicNode] Link control anchor failed from atomic node:%s to atomic_addr_clean:%s node.",
             node->GetName().c_str(), clean_addr_node->GetName().c_str());
      return ret;
    }
    index++;
  }
  return SUCCESS;
}

Status AtomicAddrCleanPass::HandleNormalGraph(ComputeGraphPtr &graph,
                                              const std::vector<NodePtr> &atomic_node_vec) const {
  GELOGD("Not loop graph and unknown graph. It will insert atomic clean nodes.");

  std::vector<NodePtr> common_atomic_nodes;
  std::vector<NodePtr> dispersed_atomic_nodes;
  auto ret = HandleDispersedAtomicNodes(graph, atomic_node_vec, common_atomic_nodes, dispersed_atomic_nodes);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][HandleDispersedAtomicNodes] failed, graph name is %s.", graph->GetName().c_str());
    return ret;
  }

  if (common_atomic_nodes.empty()) {
    GELOGI("common_atomic_nodes is empty");
    return SUCCESS;
  }

  // not loop graph , insert only one memset node in graph
  NodePtr clean_addr_node = InsertAtomicMemsetNode(graph);
  if (clean_addr_node == nullptr) {
    GELOGE(FAILED, "[Insert][AtomicAddrCleanNode] in graph:%s failed. Ignore atomicAddrClean pass.",
           graph->GetName().c_str());
    return FAILED;
  }
  for (const auto &node : common_atomic_nodes) {
    ret = LinkToAtomicNode(node, clean_addr_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Link][ControlAnchor] failed from atomic node:%s to atomic_addr_clean node:%s.",
             node->GetName().c_str(), clean_addr_node->GetName().c_str());
      return ret;
    }
  }

  GE_CHK_STATUS_RET(LinkToHcomPeerInNode(clean_addr_node),
                    "[Link][ControlAnchor] failed link atomic node:%s to Hcom peer in node.",
                    clean_addr_node->GetName().c_str());
  return LinkToPotentialPrecedenceNode(graph, clean_addr_node, dispersed_atomic_nodes);
}

// Add control edges from atomic clean node to all potential precedence nodes which may execute before atomic clean
// node. We hope that atomic clean node can execute with the highest priority in the entire graph. Because of stream
// concurrency mechanism, only placing it at the head can not ensure that priority. Therefore, we need to add control
// edges from atomic clean node to the nodes that may be the first node on each stream. Generally, the first nodes on
// each stream are successors of Data/Variable, and Data/Variable won't generate task or execute, so we link to the
// successors of Data/Variable.
Status AtomicAddrCleanPass::LinkToPotentialPrecedenceNode(ComputeGraphPtr &graph, NodePtr &atomic_clean_node,
                                                          const std::vector<NodePtr> &dispersed_atomic_nodes) const {
  GELOGD("Start to add control edges from %s to all second-nodes behind first-nodes which have no input.",
         atomic_clean_node->GetName().c_str());
  auto out_ctrl_anchor = atomic_clean_node->GetOutControlAnchor();
  const auto &out_control_nodes = atomic_clean_node->GetOutControlNodes();
  GE_CHECK_NOTNULL(out_ctrl_anchor);

  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    bool need_handle = (OpTypeUtils::IsDataNode(node->GetType()) || OpTypeUtils::IsVariableNode(node->GetType())) &&
                       node->GetInAllNodes().empty();
    if (!need_handle) {
      continue;
    }
    auto second_nodes = node->GetOutAllNodes();
    for (const auto &second_node : second_nodes) {
      GE_CHECK_NOTNULL(second_node);
      if ((std::find(dispersed_atomic_nodes.begin(), dispersed_atomic_nodes.end(), second_node) !=
           dispersed_atomic_nodes.end()) ||
          (second_node->GetType() == NETOUTPUT)) {
        continue;
      }
      if (std::find(need_gentask_atomic_node_.begin(), need_gentask_atomic_node_.end(), second_node) !=
          need_gentask_atomic_node_.end()) {
        GELOGD("Node %s need gen atomic task, skip link it to %s",
               second_node->GetName().c_str(), atomic_clean_node->GetName().c_str());
        continue;
      }
      if (std::find(out_control_nodes.begin(), out_control_nodes.end(), second_node) != out_control_nodes.end()) {
        GELOGD("Node %s has linked to atomic clean node %s", second_node->GetName().c_str(),
               atomic_clean_node->GetName().c_str());
        continue;
      }
      auto in_ctrl_anchor = second_node->GetInControlAnchor();
      GE_CHECK_NOTNULL(in_ctrl_anchor);
      GE_CHK_STATUS_RET(out_ctrl_anchor->LinkTo(in_ctrl_anchor));
      GELOGD("Add control edge from %s to %s.", atomic_clean_node->GetName().c_str(), second_node->GetName().c_str());
    }
  }

  return SUCCESS;
}

Status AtomicAddrCleanPass::HandleDispersedAtomicNodes(ComputeGraphPtr &graph,
                                                       const std::vector<NodePtr> &atomic_node_vec,
                                                       std::vector<NodePtr> &common_atomic_nodes,
                                                       std::vector<NodePtr> &dispersed_atomic_nodes) const {
  int32_t index = 0;
  for (const auto &node : atomic_node_vec) {
    if (IsAllConnectHcclNode(node)) {
      continue;
    }
    std::vector<int32_t> node_anchors_connect_netoutput;
    // If GetBool fail, attr is_connect_netoutput is an empty vector.
    (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_OUTPUT, node_anchors_connect_netoutput);
    // 背景：内存分配阶段atomic clean的内存分配在前，零拷贝在后，零拷贝会改输出内存地址，导致clean的内存不连续。
    // 因此直连netoutput的，且需要清理输出的算子，需要单独插入clean算子
    // 而hccl当前只需要清理输入，因此无须考虑直连输出与零拷贝的冲突场景。
    if (!node_anchors_connect_netoutput.empty() && !IsHcomAtomicNode(node)) {
      NodePtr dispersed_clean_addr_node = InsertAtomicMemsetNode(graph);
      if (dispersed_clean_addr_node == nullptr) {
        GELOGE(FAILED, "[Insert][AtomicAddrCleanNode] in graph:%s failed. Ignore atomicAddrClean pass.",
               graph->GetName().c_str());
        return FAILED;
      }

      auto dispersed_node_op_desc = dispersed_clean_addr_node->GetOpDesc();
      GE_CHECK_NOTNULL(dispersed_node_op_desc);
      std::string node_name = dispersed_node_op_desc->GetName();
      std::ostringstream oss;
      oss << node_name << "_" << index;
      node_name = oss.str();
      dispersed_node_op_desc->SetName(node_name);
      GELOGD("Inserted dispersed atomic clean node [%s] before [%s]", node_name.c_str(), node->GetName().c_str());
      ++index;
      Status ret = LinkToAtomicNode(node, dispersed_clean_addr_node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Link][ControlAnchor] failed from atomic node:%s to atomic_addr_clean node:%s.",
               node->GetName().c_str(), dispersed_clean_addr_node->GetName().c_str());
        return ret;
      }
      dispersed_atomic_nodes.emplace_back(node);

      GE_CHK_STATUS_RET(LinkToHcomPeerInNode(dispersed_clean_addr_node),
                        "[Link][ControlAnchor] failed link atomic node:%s to Hcom peer in node.",
                        dispersed_clean_addr_node->GetName().c_str());
    } else {
      common_atomic_nodes.emplace_back(node);
    }
  }

  return SUCCESS;
}

bool AtomicAddrCleanPass::CheckAccuracySupported(const OpDescPtr &op_desc) {
  auto instance = GELib::GetInstance();
  GE_ASSERT_NOTNULL(instance, "GELib is not initialized!");
  GE_ASSERT_TRUE(instance->InitFlag(), "GELib is not initialized!");
  OpsKernelManager &ops_kernel_manager = instance->OpsKernelManagerObj();
  std::vector<OpInfo> op_infos = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  if (op_infos.empty()) {
    GELOGI("Can not get op info by op type:%s", op_desc->GetType().c_str());
    return false;
  }
  std::string unsupported_reason;
  for (const auto &it : op_infos) {
    auto kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
    auto &kernel_name = it.opKernelLib;
    auto kernel_info_store = kernel_map.find(kernel_name);
    if (kernel_info_store != kernel_map.end()) {
      if (kernel_info_store->second != nullptr &&
          kernel_info_store->second->CheckSupported(op_desc, unsupported_reason)) {
        GELOGI("OpKernelLibName %s and engine name %s into op_desc %s", kernel_name.c_str(), it.engine.c_str(),
               op_desc->GetName().c_str());
        return true;
      }
    }
  }
  GELOGI("op:%s CheckSupported result: %s", op_desc->GetName().c_str(), unsupported_reason.c_str());
  return false;
}

NodePtr AtomicAddrCleanPass::InsertAtomicMemsetNode(ComputeGraphPtr &graph) const {
  static std::atomic<int64_t> node_count{};
  OpDescPtr op_desc = MakeShared<OpDesc>(NODE_NAME_ATOMIC_MEMSET, MEMSET);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(INTERNAL_ERROR, "[New][OpDesc] failed.");
    return nullptr;
  }
  if (!ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_ATOMIC_MEMSET_SIZES, {})) {
    GELOGE(FAILED, "[Set][Attr] attr[%s] to op[%s] failed.", ATTR_NAME_ATOMIC_MEMSET_SIZES.c_str(),
           op_desc->GetName().c_str());
    return nullptr;
  }
  // try add memset node, if failed, add atomic addr clean node
  if (!CheckAccuracySupported(op_desc)) {
    op_desc = MakeShared<OpDesc>(NODE_NAME_ATOMIC_ADDR_CLEAN, ATOMICADDRCLEAN);
    GE_ASSERT_NOTNULL(op_desc);
  }
  std::string session_graph_id;
  if (!AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Get graph session_graph_id attr failed.");
  }
  if (!session_graph_id.empty()) {
    (void) AttrUtils::SetStr(op_desc, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  }
  std::string node_name = op_desc->GetName();
  // Only flush subgraph name
  if (graph->GetParentGraph() != nullptr) {
    node_name = graph->GetName() + "_" + node_name;
  }

  std::string name = node_name + session_graph_id + std::to_string(++node_count);
  op_desc->SetName(name);
  GELOGI("Create cleanAddr op:%s", op_desc->GetName().c_str());
  // To avoid same name between graphs, set session graph id to this node
  NodePtr clean_addr_node = graph->AddNodeFront(op_desc);
  return clean_addr_node;
}

Status AtomicAddrCleanPass::LinkToAtomicNode(const NodePtr &atomic_node, NodePtr &atomic_clean_node) const {
  GE_IF_BOOL_EXEC(atomic_node == nullptr || atomic_clean_node == nullptr,
                  REPORT_INNER_ERR_MSG("E19999", "Param atomic_node or atomic_clean_node is nullptr, "
                                     "check invalid");
                  DOMI_LOGE("[Check][Param] param [atomic_node][atomic_clean_node] must not be null.");
                  return PARAM_INVALID);
  InControlAnchorPtr in_ctrl_anchor = atomic_node->GetInControlAnchor();
  OutControlAnchorPtr out_ctrl_anchor = atomic_clean_node->GetOutControlAnchor();
  if (in_ctrl_anchor == nullptr || out_ctrl_anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "in_ctrl_anchor of op:%s(%s) or out_ctrl_anchor of op:%s(%s) is nullptr, "
                       "check invalid",
                       atomic_node->GetName().c_str(), atomic_node->GetType().c_str(),
                       atomic_clean_node->GetName().c_str(), atomic_clean_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] in_ctrl_anchor of op:%s(%s) or out_ctrl_anchor of op:%s(%s) is nullptr.",
           atomic_node->GetName().c_str(), atomic_node->GetType().c_str(),
           atomic_clean_node->GetName().c_str(), atomic_clean_node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (!out_ctrl_anchor->IsLinkedWith(in_ctrl_anchor)) {
    graphStatus status = GraphUtils::AddEdge(out_ctrl_anchor, in_ctrl_anchor);
    GE_ASSERT_GRAPH_SUCCESS(
        status, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
        out_ctrl_anchor->GetOwnerNode()->GetName().c_str(), out_ctrl_anchor->GetOwnerNode()->GetType().c_str(),
        in_ctrl_anchor->GetOwnerNode()->GetName().c_str(), in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
    GELOGD("Graph add cleanAddrNode op out ctrl edge, dst node: %s.", atomic_node->GetName().c_str());
  }

  std::string stream_label;
  if (is_loop_graph_ && AttrUtils::GetStr(atomic_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
    if (!AttrUtils::SetStr(atomic_clean_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
      GELOGW("LinkToAtomicNode: SetStr failed");
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

bool AtomicAddrCleanPass::IsNeedCleanNode(const NodePtr &node, const bool need_clean_separately) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(FAILED, "[Check][Param] node is nullptr.");
                  return false);
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  // 1.Check if isAtomic attrs exist for HCOM
  if ((!need_clean_separately) && CheckAtomicFromOpsKernel(node)) {
    return true;
  }

  // 2.Check atomic attr in node
  std::map<std::string, std::map<int64_t, int64_t>> atomic_workspace_index_size;
  bool has_atomic_input = op_desc->HasAttr(ATOMIC_ATTR_INPUT_INDEX);
  bool has_atomic_output = op_desc->HasAttr(ATOMIC_ATTR_OUTPUT_INDEX);
  atomic_workspace_index_size = op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_index_size);
  if (!has_atomic_input && !has_atomic_output && atomic_workspace_index_size.empty()) {
    return false;
  }

  graphStatus ret = op_desc->SetAttr(ATOMIC_ATTR_IS_ATOMIC_NODE, GeAttrValue::CreateFrom<bool>(true));
  if (ret != GRAPH_SUCCESS) {
    GELOGW("set attr ATOMIC_ATTR_IS_ATOMIC_NODE fail.");
  }
  GELOGD("Recognized atomic op %s from FE engine.", op_desc->GetName().c_str());
  if (need_clean_separately) {
    ge::AttrUtils::SetBool(op_desc, "need_gentask_atomic", true);
    return false;
  }
  const auto connect_to_no_task = IsConnectNoTaskNode(node);
  const auto ref_variable = RefVariable(node);
  if (connect_to_no_task || ref_variable) {
    ge::AttrUtils::SetBool(op_desc, "need_gentask_atomic", true);
    need_gentask_atomic_node_.push_back(node);
    GELOGI("set node: %s need_gentask_atomic true, connect_to_no_task: %d, ref_variable: %d", node->GetNamePtr(),
           connect_to_no_task, ref_variable);
    return false;
  }

  return true;
}

bool AtomicAddrCleanPass::IsAllConnectHcclNode(const NodePtr &node) const {
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    bool next_node_have_hccl = false;
    for (size_t i = 0U; i < out_anchor->GetPeerInDataAnchors().size(); i++) {
      auto peerAnchor = out_anchor->GetPeerInDataAnchors().at(i);
      if (peerAnchor == nullptr) {
        GELOGD("Node %s in anchor is null", node->GetName().c_str());
        continue;
      }
      auto next_node = peerAnchor->GetOwnerNode();
      if (IsHcomAtomicNode(next_node)) {
        next_node_have_hccl = true;
        break;
      }
    }
    if (!next_node_have_hccl) {
       return false;
    }
  }
  return true;
}

// transdata有的需要清零，有的属于回边机制，输出地址引用变量内存
bool AtomicAddrCleanPass::RefVariable(const NodePtr &node) const {
  std::string ref_var_src_var_name;
  for (const auto &tensor_desc : node->GetOpDescBarePtr()->GetAllOutputsDescPtr()) {
    if (ge::AttrUtils::GetStr(tensor_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name)) {
      return true;
    }
  }
  return false;
}

bool AtomicAddrCleanPass::IsConnectNoTaskNode(const NodePtr &node) const {
  bool is_connect_notask = false;
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    for (size_t i = 0U; i < out_anchor->GetPeerInDataAnchors().size(); i++) {
      auto peerAnchor = out_anchor->GetPeerInDataAnchors().at(i);
      if (peerAnchor == nullptr) {
        GELOGD("Node %s in anchor is null", node->GetName().c_str());
        continue;
      }
      auto next_node = peerAnchor->GetOwnerNode();
      OpDescPtr op_desc = next_node->GetOpDesc();
      (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, is_connect_notask);
      if (is_connect_notask) {
        GELOGD("Node %s is linked with no task node %s", node->GetName().c_str(), op_desc->GetName().c_str());
        return true;
      }
    }
  }
  return false;
}

/// @brief Clear Status, used for subgraph pass
/// @return SUCCESS
Status AtomicAddrCleanPass::ClearStatus() {
  hcom_node_vec_.clear();
  need_gentask_atomic_node_.clear();
  return SUCCESS;
}

Status AtomicAddrCleanPass::CallCompileOp(const std::vector<NodePtr> &atomic_node_vec) {
  GE_TIMESTAMP_CALLNUM_START(UnknownGraphCompileOp);
  std::unordered_map<std::string, std::vector<ge::NodePtr>> node_vector_map;
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if ((instance == nullptr) || !instance->InitFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "GeLib is not init before, check invalid");
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GeLib is not init before.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }

  for (auto &atomic_node: atomic_node_vec) {
    auto op_desc = atomic_node->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGW("op desc is nullptr.");
      continue;
    }
    std::string kernel_lib_name = op_desc->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      REPORT_INNER_ERR_MSG("E19999", "Find ops kernel by name of op:%s(%s) failed",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(ge::INTERNAL_ERROR, "[Get][OpKernelLibName] of op:%s(%s) failed.", op_desc->GetName().c_str(),
             op_desc->GetType().c_str());
      return ge::INTERNAL_ERROR;
    }

    OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
    GE_CHECK_NOTNULL(kernel_info);
    node_vector_map[kernel_lib_name].emplace_back(atomic_node);
    GELOGI("node %s %s start to call engine %s to compile.", atomic_node->GetName().c_str(),
           atomic_node->GetType().c_str(), kernel_lib_name.c_str());
  }

  for (auto &it : node_vector_map) {
    auto &kernel_lib_name = it.first;
    auto &node_vector = it.second;
    OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
    GE_CHECK_NOTNULL(kernel_info);
    GE_TIMESTAMP_RESTART(UnknownGraphCompileOp);
    auto ret = kernel_info->CompileOp(node_vector);
    GELOGI("The atomic node size of compile op of %s is %zu", kernel_lib_name.c_str(), node_vector.size());
    GE_TIMESTAMP_ADD(UnknownGraphCompileOp);
    GE_ASSERT_SUCCESS(ret, "Call CompileOp failed, kernel_lib_name:%s, ret:%u", kernel_lib_name.c_str(), ret);
  }
  GE_TIMESTAMP_CALLNUM_END(UnknownGraphCompileOp, "AtomicAddrCleanPass::CallCompileOp");
  return SUCCESS;
}

// for HCOM atomic node, add one more control link to peer-in node
Status AtomicAddrCleanPass::LinkToHcomPeerInNode(NodePtr &atomic_clean_node) const {
  for (const auto &node : hcom_node_vec_) {
    for (auto &in_anchor : node->GetAllInDataAnchors()) {
      GE_CHECK_NOTNULL(in_anchor->GetPeerOutAnchor());
      NodePtr peer_in_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
      auto ret = LinkToAtomicNode(peer_in_node, atomic_clean_node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Link][ControlAnchor] from node:%s to node:%s failed",
               peer_in_node->GetName().c_str(), atomic_clean_node->GetName().c_str());
        return ret;
      }
    }
  }
  GELOGD("[Link] Link to hcom peer in node success.");
  return SUCCESS;
}
bool AtomicAddrCleanPass::IsHcomAtomicNode(const NodePtr &node) const {
  const auto it = std::find(hcom_node_vec_.cbegin(), hcom_node_vec_.cend(), node);
  return (it != hcom_node_vec_.cend());
}

REG_PASS_OPTION("AtomicAddrCleanPass").LEVELS(OoLevel::kO1);
}  // namespace ge
