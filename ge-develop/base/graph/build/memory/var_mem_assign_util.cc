/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/var_mem_assign_util.h"
#include <vector>
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "common/op/transop_util.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/op/ge_op_utils.h"
#include "base/err_msg.h"

namespace {
constexpr uint32_t kVarMemMaxRecursion = 16U;
}

namespace ge {
Status VarMemAssignUtil::AssignVarMemory(const ge::ComputeGraphPtr &compute_graph) {
  return AssignStaticMemory2Node(compute_graph);
}

Status VarMemAssignUtil::AssignConstantOpMemory(const ge::ComputeGraphPtr &compute_graph) {
  return AssignStaticMemory2Node(compute_graph);
}

static Status GetSizeByNodeType(const OpDescPtr &op_desc, const GeTensorDescPtr &tensor_desc, int64_t &size) {
  const auto node_type = op_desc->GetType();
  if (tensor_desc->GetDataType() == DT_STRING) {
    if ((node_type == CONSTANT) || (node_type == CONSTANTOP)) {
      GE_ASSERT_SUCCESS(OpUtils::GetConstantStrMemSize(op_desc, size), "node: %s get size failed",
                        op_desc->GetNamePtr());
      GELOGD("node: %s get size from value attr success. node_type is %s, data_type is DT_STRING, size: %lld",
             op_desc->GetNamePtr(), node_type.c_str(), size);
      return SUCCESS;
    } else if ((node_type == FILECONSTANT) && AttrUtils::GetInt(op_desc, ATTR_NAME_LENGTH, size)) {
      GELOGD("node: %s get length attr success. node type is FileConstant and data type is DT_STRING, size: %lld",
             op_desc->GetNamePtr(), size);
      return SUCCESS;
    }
  }
  GE_ASSERT_SUCCESS(TensorUtils::GetTensorMemorySizeInBytes(*tensor_desc, size),
                    "[Get][TensorMemorySize] In Bytes failed for op:%s(%s)",
                    op_desc->GetNamePtr(), node_type.c_str());
  return SUCCESS;
}

Status VarMemAssignUtil::AssignStaticMemory2Node(const ge::ComputeGraphPtr &compute_graph) {
  GE_IF_BOOL_EXEC(compute_graph == nullptr, return FAILED);
  for (const ge::NodePtr &n : compute_graph->GetAllNodes()) {
    const auto node_type = n->GetType();
    if ((node_type != VARIABLE) && (node_type != CONSTANTOP) && (node_type != FILECONSTANT) &&
        (node_type != CONSTPLACEHOLDER)) {
        continue;
    }
    const auto op_desc = n->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string ref_var_src_var_name;
    (void)ge::AttrUtils::GetStr(op_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);

    GE_IF_BOOL_EXEC(ge::AttrUtils::GetStr(op_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name), continue);
    std::string node_name = GetNameForVarManager(op_desc);
    GE_IF_BOOL_EXEC(op_desc->GetAllOutputsDesc().empty(),
                    REPORT_INNER_ERR_MSG("E19999", "check node:%s has no OutputDesc", n->GetName().c_str());
                    GELOGE(FAILED, "[Check][Param] node:%s has no OutputDesc.", n->GetName().c_str());
                    return FAILED);
    auto tensor_desc = op_desc->MutableOutputDesc(0U);
    GE_CHECK_NOTNULL(tensor_desc);
    rtMemType_t memory_type = RT_MEMORY_HBM;
    uint32_t mem_type = 0U;
    if (AttrUtils::GetInt(op_desc, ATTR_OUTPUT_MEMORY_TYPE, mem_type) && (mem_type == 1U)) {
      memory_type = RT_MEMORY_RDMA_HBM;
    }
    int64_t size_temp = 0;
    GE_ASSERT_SUCCESS(GetSizeByNodeType(op_desc, tensor_desc, size_temp), "node: %s get size failed",
                      op_desc->GetNamePtr());
    TensorUtils::SetSize(*tensor_desc, size_temp);
    std::vector<int64_t> output_offsets(op_desc->GetOutputsSize(), 0);
    op_desc->SetOutputOffset(output_offsets);

    GE_CHECK_NOTNULL(VarManager::Instance(compute_graph->GetSessionID()));
    if (!VarManager::Instance(compute_graph->GetSessionID())->IsVarExist(node_name, *tensor_desc)) {
      GE_CHK_STATUS_RET(VarManager::Instance(compute_graph->GetSessionID())
                            ->AssignVarMem(node_name, op_desc, *tensor_desc, memory_type));
      GE_IF_BOOL_EXEC(n->GetType() == VARIABLE,
                      GE_CHK_STATUS_RET(AssignData2Fp32Var(n, compute_graph->GetSessionID())));
      GE_CHK_STATUS_RET(VarManager::Instance(compute_graph->GetSessionID())
                            ->SetAllocatedGraphId(node_name, compute_graph->GetGraphID()));
    }

    uint8_t *dev_ptr = nullptr;
    GE_CHK_STATUS_RET(VarManager::Instance(compute_graph->GetSessionID())
                          ->GetVarAddr(node_name, *tensor_desc, dev_ptr, memory_type));
    std::vector<int64_t> output_list = op_desc->GetOutputOffset();
    GE_IF_BOOL_EXEC(output_list.empty(), return FAILED);
    output_list[0U] = static_cast<int64_t>(PtrToValue(dev_ptr));
    op_desc->SetOutputOffset(output_list);
  }
  return SUCCESS;
}

std::string VarMemAssignUtil::GetNameForVarManager(const OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    return "";
  }
  std::string src_const_name;
  if (ge::AttrUtils::GetStr(op_desc, ATTR_NAME_SRC_CONST_NAME, src_const_name) && (!src_const_name.empty())) {
    return src_const_name;
  }
  return op_desc->GetName();
}

Status VarMemAssignUtil::AssignData2Fp32Var(const ge::NodePtr &node, const uint64_t session_id) {
  std::string src_var_name;
  GE_CHECK_NOTNULL(node->GetOpDesc());
  if (ge::AttrUtils::GetStr(node->GetOpDesc(), VAR_ATTR_SRC_VAR_NAME, src_var_name)) {
    ge::GeTensorDesc cur_tensor_desc;
    uint8_t *dev_ptr = nullptr;
    rtMemType_t memory_type = RT_MEMORY_HBM;
    GE_CHECK_NOTNULL(VarManager::Instance(session_id));
    GE_CHK_STATUS_RET(VarManager::Instance(session_id)->GetCurVarDesc(src_var_name, cur_tensor_desc));
    GE_CHK_STATUS_RET(
        VarManager::Instance(session_id)->GetVarAddr(src_var_name, cur_tensor_desc, dev_ptr, memory_type));
    GE_CHK_STATUS_RET(VarManager::Instance(session_id)
                          ->SetVarAddr(node->GetName(), cur_tensor_desc, dev_ptr, memory_type, node->GetOpDesc()));
  }
  return SUCCESS;
}

Status VarMemAssignUtil::AssignVarAttr2Nodes(const ge::ComputeGraphPtr &compute_graph) {
  for (const ge::NodePtr &node : compute_graph->GetAllNodes()) {
    GE_IF_BOOL_EXEC(node->GetType() != VARIABLE, continue);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string ref_var_src_var_name;
    GE_IF_BOOL_EXEC(ge::AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name), continue);
    GE_CHK_STATUS_RET(DealVariableNode(compute_graph->GetGraphID(), node, compute_graph->GetSessionID()));
  }
  return SUCCESS;
}

Status VarMemAssignUtil::SetOutVariableAttr(const ge::NodePtr &node, const ge::NodePtr &var_node, const size_t index,
                                            const uint64_t session_id) {
  std::vector<int64_t> output_list;
  uint8_t *dev_ptr = nullptr;
  GE_CHECK_NOTNULL(node->GetOpDesc());
  output_list = node->GetOpDesc()->GetOutputOffset();
  if (output_list.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "check node:%s output_offset_list is empty", node->GetName().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] node:%s Output_list is empty", node->GetName().c_str());
    return PARAM_INVALID;
  }
  GE_CHECK_NOTNULL(var_node->GetOpDesc());
  const GeTensorDesc var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0U);
  rtMemType_t memory_type = RT_MEMORY_HBM;
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  GE_CHK_STATUS_RET(
      VarManager::Instance(session_id)->GetVarAddr(var_node->GetName(), var_tensor_desc, dev_ptr, memory_type));

  const size_t out_list_size = output_list.size();
  if (index >= out_list_size) {
    REPORT_INNER_ERR_MSG("E19999", "param index:%zu >= output_list.size() %zu in node %s, check invalid",
                       index, out_list_size, node->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] index %zu >= output_list.size() %zu in node %s", index, out_list_size,
           node->GetName().c_str());
    return FAILED;
  }

  output_list[index] = static_cast<int64_t>(PtrToValue(dev_ptr));
  GELOGD("Assign node outputOffset[index] is: %ld", output_list[index]);
  node->GetOpDesc()->SetOutputOffset(output_list);

  return SUCCESS;
}

Status VarMemAssignUtil::DealExportVariableNode(const ge::NodePtr &node, const ge::NodePtr &var_node,
                                                const uint64_t session_id, const uint32_t depth) {
  if (depth >= kVarMemMaxRecursion) {
    GELOGE(FAILED, "[Invoke][DealExportVariableNode]There are too much recursion:%u > max:%u", depth,
           kVarMemMaxRecursion);
    REPORT_INNER_ERR_MSG("E19999", "[DealExportVariableNode]There are too much recursion:%u > max:%u", depth,
                       kVarMemMaxRecursion);
    return FAILED;
  }
  const ge::OutDataAnchorPtr var_out_anchor = node->GetOutDataAnchor(0);
  GE_IF_BOOL_EXEC(var_out_anchor == nullptr, return FAILED);
  for (const ge::InDataAnchorPtr &dst_in_var_anchor : var_out_anchor->GetPeerInDataAnchors()) {
    const ge::NodePtr dst_node = dst_in_var_anchor->GetOwnerNode();
    if ((dst_node->GetType() == ASSIGN) || (dst_node->GetType() == ASSIGNADD) || (dst_node->GetType() == ASSIGNSUB)) {
      if (dst_in_var_anchor == dst_node->GetInDataAnchor(0)) {
        GE_CHK_STATUS_RET(DealExportVariableNode(dst_node, var_node, session_id, depth + 1U));
      }
    }
  }
  GE_CHK_STATUS_RET(SetOutVariableAttr(node, var_node, 0U, session_id));
  return SUCCESS;
}

Status VarMemAssignUtil::DealBroadCastNode(const uint32_t graph_id, const ge::NodePtr &node,
                                           const ge::InDataAnchorPtr &in_data_anchor, const ge::NodePtr &var_node,
                                           const uint64_t session_id) {
  VarBroadCastInfo broad_cast_info;
  broad_cast_info.idx = in_data_anchor->GetIdx();
  broad_cast_info.var_name = var_node->GetName();
  broad_cast_info.broadcast_name = node->GetName();

  const auto op_desc = node->GetOpDesc();
  GE_CHK_BOOL_RET_STATUS(op_desc != nullptr, FAILED,
                         "[Check][Param] Get broadcast op %s desc is nullptr", node->GetName().c_str());

  GE_IF_BOOL_EXEC(broad_cast_info.idx < 0,
                  GELOGI("Broadcast input index must be positive, actual %d", broad_cast_info.idx);
                  return INTERNAL_ERROR);

  const auto broad_cast_index = static_cast<size_t>(broad_cast_info.idx);
  auto input_tensor_desc_ptr_vistor = op_desc->GetAllInputsDescPtr();
  if (input_tensor_desc_ptr_vistor.size() <= broad_cast_index) {
    REPORT_INNER_ERR_MSG("E19999", "Get broadcast op %s input tensor desc size [%zu] < idx [%d]",
                       node->GetName().c_str(), input_tensor_desc_ptr_vistor.size(), broad_cast_info.idx);
    GELOGE(FAILED, "[Check][Param] Get broadcast op %s input tensor desc size [%zu] < idx [%d]",
           node->GetName().c_str(), input_tensor_desc_ptr_vistor.size(), broad_cast_info.idx);
    return FAILED;
  }
  const ge::GeTensorDescPtr input_tensor_desc =
      input_tensor_desc_ptr_vistor.at(static_cast<size_t>(broad_cast_info.idx));
  int64_t input_size = 0;
  GE_CHK_STATUS(TensorUtils::GetSize(*input_tensor_desc, input_size), "get input size failed.");
  broad_cast_info.input_size = static_cast<uint64_t>(input_size);

  const std::vector<int64_t> output_list = op_desc->GetOutputOffset();
  GE_CHK_BOOL_RET_STATUS(output_list.size() > broad_cast_index, FAILED,
                         "[Check][Param] Get broadcast op %s output_list size [%zu] < idx [%d]",
                         node->GetName().c_str(), output_list.size(), broad_cast_info.idx);
  broad_cast_info.input_offset = output_list[static_cast<size_t>(broad_cast_info.idx)];
  broad_cast_info.output_offset = output_list[static_cast<size_t>(broad_cast_info.idx)];

  op_desc->SetInputOffset(output_list);

  auto output_tensor_desc_ptr_vistor = op_desc->GetAllOutputsDescPtr();
  GE_CHK_BOOL_RET_STATUS(output_tensor_desc_ptr_vistor.size() > broad_cast_index, FAILED,
                         "[Check][Param] Get broadcast op %s output tensor desc size [%zu] < idx [%d]",
                         node->GetName().c_str(), output_tensor_desc_ptr_vistor.size(), broad_cast_info.idx);
  const ge::GeTensorDescPtr output_tensor_desc =
      output_tensor_desc_ptr_vistor.at(static_cast<size_t>(broad_cast_info.idx));
  int64_t output_size = 0;
  GE_CHK_STATUS(TensorUtils::GetSize(*output_tensor_desc, output_size), "[Check][Param] get output size failed.");
  broad_cast_info.output_size = static_cast<uint64_t>(output_size);
  GE_CHK_BOOL_RET_STATUS(broad_cast_info.output_size == broad_cast_info.input_size, FAILED,
                         "[Check][Param] Broadcast op input size[%" PRIu64 "] is not equal output size[%" PRIu64 "]",
                         broad_cast_info.input_size, broad_cast_info.output_size);

  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  GE_CHK_STATUS_RET(VarManager::Instance(session_id)->SaveBroadCastInfo(graph_id, broad_cast_info));
  return SUCCESS;
}

Status VarMemAssignUtil::DealVariableNode(const uint32_t graph_id, const ge::NodePtr &node, const uint64_t session_id) {
  GE_CHK_STATUS_RET(SetOutVariableAttr(node, node, 0U, session_id));

  for (const ge::OutDataAnchorPtr &var_out_data_anchor : node->GetAllOutDataAnchors()) {
    for (const ge::InDataAnchorPtr &dst_in_data_anchor : var_out_data_anchor->GetPeerInDataAnchors()) {
      const ge::NodePtr dst_node = dst_in_data_anchor->GetOwnerNode();
      if ((dst_node->GetType() == HCOMBROADCAST) || (dst_node->GetType() == HVDCALLBACKBROADCAST)) {
        GE_CHK_STATUS_RET(DealBroadCastNode(graph_id, dst_node, dst_in_data_anchor, node, session_id));
        continue;
      }

      if ((dst_node->GetType() == ASSIGN) || (dst_node->GetType() == ASSIGNADD) || (dst_node->GetType() == ASSIGNSUB)) {
        if (dst_in_data_anchor == dst_node->GetInDataAnchor(0)) {
          GE_CHK_STATUS_RET(DealExportVariableNode(dst_node, node, session_id));
        }
      }
      const auto dst_type = dst_node->GetType();
      const bool is_trans_node =
          (dst_type == TRANSDATA) || (dst_type == CAST) || (dst_type == TRANSPOSE) || (dst_type == PERMUTE);
      if (is_trans_node) {
        const NodePtr final_trans_node = GetFinalTransNode(dst_node);
        GE_CHK_STATUS_RET(DealTransNode(final_trans_node));
      }
    }
  }
  return SUCCESS;
}

ge::NodePtr VarMemAssignUtil::GetFinalTransNode(const ge::NodePtr &trans_node, const uint32_t depth) {
  NodePtr final_ref_node = trans_node;
  if (depth >= kVarMemMaxRecursion) {
    GELOGE(FAILED, "[Invoke][GetFinalTransNode]There are too much recursion:%u > max:%u", depth, kVarMemMaxRecursion);
    REPORT_INNER_ERR_MSG("E19999", "[GetFinalTransNode]There are too much recursion:%u > max:%u", depth,
                       kVarMemMaxRecursion);
    return final_ref_node;
  }
  const OutDataAnchorPtr trans_out_data_anchor = trans_node->GetOutDataAnchor(0);
  GE_IF_BOOL_EXEC(trans_out_data_anchor == nullptr, return final_ref_node);
  for (const auto &dst_in_anchor : trans_out_data_anchor->GetPeerInDataAnchors()) {
    const NodePtr dst_node = dst_in_anchor->GetOwnerNode();
    const auto dst_type = dst_node->GetType();
    const bool is_trans_node =
        (dst_type == TRANSDATA) || (dst_type == CAST) || (dst_type == TRANSPOSE) || (dst_type == PERMUTE);
    if (is_trans_node && (dst_in_anchor->GetIdx() == 0)) {
      final_ref_node = GetFinalTransNode(dst_node, depth + 1U);
    }
  }
  GELOGI("Final writable node is %s", final_ref_node->GetName().c_str());
  return final_ref_node;
}

Status VarMemAssignUtil::DealTransNode(const ge::NodePtr &final_trans_node) {
  const ge::OutDataAnchorPtr final_trans_out_anchor = final_trans_node->GetOutDataAnchor(0);
  GE_IF_BOOL_EXEC(final_trans_out_anchor == nullptr, return SUCCESS);
  for (const ge::InDataAnchorPtr &dst_in_var_anchor : final_trans_out_anchor->GetPeerInDataAnchors()) {
    const ge::NodePtr dst_node = dst_in_var_anchor->GetOwnerNode();
    if ((dst_node->GetType() == ASSIGN) || (dst_node->GetType() == ASSIGNADD) || (dst_node->GetType() == ASSIGNSUB)) {
      GE_CHK_STATUS_RET(DealExportTransNode(dst_node, final_trans_node));
    }
  }
  return SUCCESS;
}

Status VarMemAssignUtil::DealExportTransNode(const ge::NodePtr &node, const ge::NodePtr &final_trans_node,
                                             const uint32_t depth) {
  if (depth >= kVarMemMaxRecursion) {
    GELOGE(FAILED, "[Invoke][DealExportTransNode]There are too much recursion:%u > max:%u", depth, kVarMemMaxRecursion);
    REPORT_INNER_ERR_MSG("E19999", "[DealExportTransNode]There are too much recursion:%u > max:%u", depth,
                       kVarMemMaxRecursion);
    return FAILED;
  }
  const ge::OutDataAnchorPtr node_out_anchor = node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(node_out_anchor);
  for (const ge::InDataAnchorPtr &dst_in_var_anchor : node_out_anchor->GetPeerInDataAnchors()) {
    const ge::NodePtr dst_node = dst_in_var_anchor->GetOwnerNode();
    if ((dst_node->GetType() == ASSIGN) || (dst_node->GetType() == ASSIGNADD) || (dst_node->GetType() == ASSIGNSUB)) {
      GE_CHK_STATUS_RET(DealExportTransNode(dst_node, final_trans_node, depth + 1U));
    }
  }
  GE_CHK_STATUS_RET(SetOutTransNodeToAssign(node, final_trans_node, 0U));
  return SUCCESS;
}

Status VarMemAssignUtil::SetOutTransNodeToAssign(const ge::NodePtr &node, const ge::NodePtr &final_trans_node,
                                                 const size_t index) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  GE_CHECK_NOTNULL(final_trans_node->GetOpDesc());
  // get final_trans_node outputOffset
  const std::vector<int64_t> final_trans_output_list = final_trans_node->GetOpDesc()->GetOutputOffset();
  GE_CHECK_SIZE(final_trans_output_list.size());

  // get assign_node outputOffset
  std::vector<int64_t> output_list = node->GetOpDesc()->GetOutputOffset();
  const auto out_list_size = output_list.size();
  GE_CHECK_SIZE(out_list_size);
  GE_CHK_BOOL_RET_STATUS(index < out_list_size, FAILED,
                         "[Check][Param] index %zu >= output_list.size() %zu, node:%s",
                         index, out_list_size, node->GetName().c_str());

  // final_trans_node outputOffset[0] to assign_node outputOffset[0]
  GELOGI("final_trans_node %s output offset[0] is: %ld", final_trans_node->GetNamePtr(), final_trans_output_list[0U]);

  output_list[index] = final_trans_output_list[0U];
  GELOGI("Assign node %s output offset[0] is: %ld", node->GetNamePtr(), output_list[index]);
  node->GetOpDesc()->SetOutputOffset(output_list);

  return SUCCESS;
}

Status VarMemAssignUtil::AssignMemory2HasRefAttrNode(const ge::ComputeGraphPtr &compute_graph) {
  GraphToNodeMap graph_to_node;
  for (const ge::NodePtr &n : compute_graph->GetAllNodes()) {
    std::string ref_var_src_var_name;
    const auto op_desc = n->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (uint32_t idx = 0U; idx < op_desc->GetOutputsSize(); idx += 1U) {
      const auto out_desc = op_desc->MutableOutputDesc(idx);
      if (ge::AttrUtils::GetStr(out_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name)) {
        GE_CHK_STATUS_RET(
            AssignData2VarRef(n, ref_var_src_var_name, compute_graph->GetSessionID(), idx, graph_to_node));
      }
    }
  }
  return SUCCESS;
}

Status VarMemAssignUtil::AssignData2VarRef(const ge::NodePtr &has_ref_attr_node, const std::string &src_var_name,
                                           const uint64_t session_id, const uint32_t out_index,
                                           GraphToNodeMap &graph_to_node) {
  // Get ref_var_src_var address
  const auto root_graph = GraphUtils::FindRootGraph(has_ref_attr_node->GetOwnerComputeGraph());
  GE_CHECK_NOTNULL(root_graph);
  // Cache mapping (name to nodeptr) simproves query performance
  auto &name_to_node = graph_to_node[root_graph];
  if (name_to_node.empty()) {
    for (const ge::NodePtr &n : root_graph->GetDirectNode()) {
      (void)name_to_node.emplace(n->GetName(), n);
    }
    for (const auto &sub_graph : root_graph->GetAllSubgraphs()) {
      auto &name_to_node_sub = graph_to_node[sub_graph];
      if (name_to_node_sub.empty()) {
        for (const ge::NodePtr &n : sub_graph->GetDirectNode()) {
          (void)name_to_node_sub.emplace(n->GetName(), n);
        }
      }
    }
  }

  ge::NodePtr var_ref_src_var = nullptr;
  auto it = name_to_node.find(src_var_name);
  if ((it != name_to_node.end()) && (it->second != nullptr)) {
    var_ref_src_var = it->second;
  } else {
    for (const auto &sub_graph : root_graph->GetAllSubgraphs()) {
      auto &name_to_node_sub = graph_to_node[sub_graph];
      it = name_to_node_sub.find(src_var_name);
      if ((it != name_to_node_sub.end()) && (it->second != nullptr)) {
          var_ref_src_var = it->second;
          break;
        }
    }
  }
  GE_IF_BOOL_EXEC((var_ref_src_var == nullptr) || (var_ref_src_var->GetOpDesc() == nullptr), return FAILED);
  const GeTensorDesc src_tensor_desc = var_ref_src_var->GetOpDesc()->GetOutputDesc(0U);
  if (var_ref_src_var->GetType() == REFDATA) {
    GELOGD("Find refdata named %s. Skip AssignData2VarRef.", src_var_name.c_str());
    return SUCCESS;
  }
  uint8_t *dev_ptr = nullptr;
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  GE_CHK_STATUS_RET(VarManager::Instance(session_id)->GetVarAddr(src_var_name, src_tensor_desc, dev_ptr));
  GE_CHECK_NOTNULL(has_ref_attr_node->GetOpDesc());
  std::vector<int64_t> ref_attr_node_output_list = has_ref_attr_node->GetOpDesc()->GetOutputOffset();
  GE_CHECK_SIZE(ref_attr_node_output_list.size());
  GE_CHK_BOOL_RET_STATUS(out_index < ref_attr_node_output_list.size(), FAILED,
                         "[Check][Param] out_index %u >= ref_attr_node_output_list.size() %zu", out_index,
                         ref_attr_node_output_list.size());

  ref_attr_node_output_list[static_cast<size_t>(out_index)] = static_cast<int64_t>(PtrToValue(dev_ptr));
  has_ref_attr_node->GetOpDesc()->SetOutputOffset(ref_attr_node_output_list);
  GELOGI("Refresh address successfully, ref node: [%s], addr: [%ld]", has_ref_attr_node->GetName().c_str(),
         ref_attr_node_output_list[static_cast<size_t>(out_index)]);
  return SUCCESS;
}
}  // namespace ge
