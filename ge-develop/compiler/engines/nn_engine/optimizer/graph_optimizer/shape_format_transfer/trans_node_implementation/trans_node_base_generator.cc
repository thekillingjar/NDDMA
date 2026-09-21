/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "common/fe_error_code.h"
#include "common/fe_op_info_common.h"
#include "common/fe_utils.h"
#include "common/unknown_shape_util.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/insert_trans_node_strategy.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"

namespace fe {
namespace {
const std::string kAssign = "Assign";
const std::string kStageAddEgFreshTransInfo = "[GraphOptJdgInst][ShapeTrans][AddEgFreshTransInfo]";
const std::unordered_set<ge::Format> kBaseFormatOf3D = {
  ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_DHWCN, ge::FORMAT_DHWNC};
}
TransNodeBaseGenerator::TransNodeBaseGenerator(FEOpsKernelInfoStorePtr fe_ops_store_ptr, TransInfoPtr trans_info_ptr)
    : trans_info_ptr_(trans_info_ptr), fe_ops_store_info_ptr_(fe_ops_store_ptr) {}

TransNodeBaseGenerator::~TransNodeBaseGenerator() {}

ge::OpDescPtr TransNodeBaseGenerator::CreateBasicOpDescForTransNode(const string &op_type) const {
  stringstream op_name_temp;
  // The atomic id of trans nodes must be unique.(start from 0)
  op_name_temp << "trans_" << op_type << "_" << GetTransAtomicId();

  ge::OpDescPtr op_desc_ptr = nullptr;
  FE_MAKE_SHARED(op_desc_ptr = std::make_shared<ge::OpDesc>(op_name_temp.str().c_str(), op_type), return nullptr);
  FE_CHECK(op_desc_ptr == nullptr,
           REPORT_FE_ERROR("[ShapeTrans][AddEgFreshTransInfo] Create op desc failed."), return nullptr);
  FE_LOGD("Create op [%s].", op_desc_ptr->GetName().c_str());
  return op_desc_ptr;
}

void TransNodeBaseGenerator::GetGroupIdFromSrcOrDst(ge::OpDescPtr &op_desc_ptr,
    uint32_t &parallel_group_id, uint32_t &heavy_op_parallel_group_id) {
  if (IsHeavyOp(op_desc_ptr)) {
    ge::AttrUtils::GetInt(op_desc_ptr, ge::ATTR_NAME_PARALLEL_GROUP_ID, heavy_op_parallel_group_id);
  } else {
    ge::AttrUtils::GetInt(op_desc_ptr, ge::ATTR_NAME_PARALLEL_GROUP_ID, parallel_group_id);
  }
}

void TransNodeBaseGenerator::SetNodeParallelGroupId(ge::OpDescPtr &op_desc_ptr, ge::OpDescPtr &src_op_desc,
      ge::OpDescPtr &dst_op_desc) {
  uint32_t parallel_group_id = DefaultGroupID;
  uint32_t heavy_op_parallel_group_id = DefaultGroupID;
  GetGroupIdFromSrcOrDst(src_op_desc, parallel_group_id, heavy_op_parallel_group_id);
  GetGroupIdFromSrcOrDst(dst_op_desc, parallel_group_id, heavy_op_parallel_group_id);
  if (heavy_op_parallel_group_id != static_cast<uint32_t>(DefaultGroupID)) {
    FE_LOGD("Create op [%s], set group ID from heavy op; group ID is %u.",
        op_desc_ptr->GetName().c_str(), heavy_op_parallel_group_id);
    ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_PARALLEL_GROUP_ID, heavy_op_parallel_group_id);
  } else if (parallel_group_id != static_cast<uint32_t>(DefaultGroupID)) {
    FE_LOGD("Create op [%s], set group Id from general op; the group Id is %u.",
        op_desc_ptr->GetName().c_str(), parallel_group_id);
    ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_PARALLEL_GROUP_ID, parallel_group_id);
  }
}

Status TransNodeBaseGenerator::SetTensorDescInfo(ge::OpDescPtr &op_desc_ptr) const {
  FE_CHECK_NOTNULL(op_desc_ptr);
  for (auto input_tensor : op_desc_ptr->GetAllInputsDescPtr()) {
    input_tensor->SetOriginFormat(trans_info_ptr_->src_out_original_format);
    input_tensor->SetOriginShape(trans_info_ptr_->src_out_original_shape);
  }
  bool is_from_const_op = false;
  (void)ge::AttrUtils::GetBool(trans_info_ptr_->src_op_desc, kIsComeFromConstOp, is_from_const_op);
  bool is_src_from_const_op = is_from_const_op || trans_info_ptr_->src_op_desc->GetType() == CONSTANT ||
          trans_info_ptr_->src_op_desc->GetType() == CONSTANTOP;
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kIsComeFromConstOp, is_src_from_const_op);
  for (auto output_tensor : op_desc_ptr->GetAllOutputsDescPtr()) {
    output_tensor->SetOriginFormat(trans_info_ptr_->src_out_original_format);
    output_tensor->SetOriginShape(trans_info_ptr_->src_out_original_shape);
    if (!is_src_from_const_op) {
      GraphPassUtil::SetOutputDescAttr(trans_info_ptr_->src_out_tensor_desc_ptr,
                                       static_cast<int64_t>(trans_info_ptr_->src_anchor->GetIdx()),
                                       trans_info_ptr_->src_op_desc,
                                       output_tensor);
    }
  }
  return SUCCESS;
}

Status TransNodeBaseGenerator::SetTensorRealDimCountAndNewShape(ge::OpDescPtr &op_desc_ptr,
                                                                std::vector<ge::GeShape> inputs_shape,
                                                                ge::GeShape output_shape) const {
  FE_CHECK_NOTNULL(op_desc_ptr);
  uint32_t index = 0;
  for (auto &input_tensor : op_desc_ptr->GetAllInputsDescPtr()) {
    if (index >= inputs_shape.size()) {
      break;
    }
    if (index < inputs_shape[index].GetDims().size()) {
      ge::TensorUtils::SetRealDimCnt(*input_tensor.get(), static_cast<uint32_t>(inputs_shape[index].GetDims().size()));
      input_tensor->SetShape(inputs_shape[index]);
    }
    index++;
  }
  for (auto &output_tensor : op_desc_ptr->GetAllOutputsDescPtr()) {
    ge::TensorUtils::SetRealDimCnt(*output_tensor.get(), static_cast<uint32_t>(output_shape.GetDims().size()));
    output_tensor->SetShape(output_shape);
  }
  return SUCCESS;
}

Status TransNodeBaseGenerator::SetNewShapeRange(const ge::OpDescPtr &op_desc_ptr,
                                                vector<std::pair<int64_t, int64_t>> &inputs_range,
                                                vector<std::pair<int64_t, int64_t>> &output_range) const {
  FE_CHECK_NOTNULL(op_desc_ptr);
  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr)) {
    uint32_t index = 0;
    for (auto &input_tensor : op_desc_ptr->GetAllInputsDescPtr()) {
      if (index < inputs_range.size()) {
        input_tensor->SetShapeRange(inputs_range);
      }
      index++;
    }
    for (auto &output_tensor : op_desc_ptr->GetAllOutputsDescPtr()) {
      output_tensor->SetShapeRange(output_range);
    }
  }
  return SUCCESS;
}

Status TransNodeBaseGenerator::AddNecessaryPeerNodes(ge::ComputeGraph &fused_graph, ge::NodePtr new_node) const {
  (void) fused_graph;
  (void) new_node;
  return SUCCESS;
}

Status TransNodeBaseGenerator::AddEdgesAndFreshTransInfo(ge::ComputeGraph &fused_graph,
                                                         const ge::OpDescPtr &op_desc_ptr) {
  ge::OutDataAnchorPtr src_anchor = trans_info_ptr_->src_anchor;
  ge::InDataAnchorPtr dst_anchor = trans_info_ptr_->dst_anchor;
  ge::NodePtr new_trans_node = nullptr;
  /*
   * Variable ---> Assign ---> X
   *                          / (control edges)
   *     ...AssignAdd -------
   *
   * Because operator Assign has the following characteristic:
   * The variable from the output of Assign will depend on the
   * AssignAdd or AssignSub. Only after calculating AssignAdd and
   * AssignSub we can execute X.
   * If Insert a node A between Assgin and X.
   * Every control edge to the node X should be
   * moved to the node A. So we use InsertNodeBefore. */
  if (trans_info_ptr_->src_op_desc_type == kAssign || InsertNodeBeforeJudge()) {
    new_trans_node = ge::GraphUtils::InsertNodeBefore(dst_anchor, op_desc_ptr);
  } else {
    new_trans_node = ge::GraphUtils::InsertNodeAfter(src_anchor, {dst_anchor}, op_desc_ptr);
  }
  FE_CHECK_NOTNULL(new_trans_node);
  GraphPassUtil::InheritGraphRelatedAttr({trans_info_ptr_->src_node_ptr}, {new_trans_node},
                                         BackWardInheritMode::kInsertNode);

  Status ret = AddNecessaryPeerNodes(fused_graph, new_trans_node);
  if (ret != SUCCESS) {
    FE_LOGD("Add edges between src_node[%s] and node[%s] unsuccessful.",
            trans_info_ptr_->src_op_desc->GetName().c_str(), new_trans_node->GetName().c_str());
    return ret;
  }
  /* After inserting new TransData or Permute op, re-write the
   * shape of det_op_desc. */
  /* After inserting trans node, src will become trans node. */
  RefreshSourceTransInfo(new_trans_node);
  return SUCCESS;
}

bool TransNodeBaseGenerator::InsertNodeBeforeJudge() const {
  std::unordered_set<string> ref_origin_name_set;
  auto in_ctrl_anchor = trans_info_ptr_->dst_node_ptr->GetInControlAnchor();
  if (in_ctrl_anchor == nullptr || in_ctrl_anchor->GetPeerOutControlAnchors().size() == 0) {
    return false;
  }
  for (const auto &peer_out_control_anchor : in_ctrl_anchor->GetPeerOutControlAnchors()) {
    auto peer_node = peer_out_control_anchor->GetOwnerNode();
    ge::Node::Vistor<ge::NodePtr> out_nodes = peer_node->GetOutDataNodes();
    if (out_nodes.empty()) {
      for (auto &in_anchor : peer_node->GetAllInDataAnchors()) {
        auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
        if (peer_out_anchor == nullptr) {
          continue;
        }
        auto peer_in_node = peer_out_anchor->GetOwnerNode();
        const std::string &peer_in_type = peer_in_node->GetType();
        bool is_var = peer_in_type == "Variable" || peer_in_type == "RefData";
        if (is_var && peer_in_node == trans_info_ptr_->src_node_ptr) {
          FE_LOGD("Dst node [%s, %s], OutCtrl peer node [%s, %s], size [%u] meets BeforeJudge condition.",
                  trans_info_ptr_->dst_node_ptr->GetNamePtr(), trans_info_ptr_->dst_node_ptr->GetTypePtr(),
                  peer_node->GetNamePtr(), peer_node->GetTypePtr(), peer_node->GetOutDataNodesSize());
          return true;
        }
      }
    }
    string ref_origin_name;
    (void)ge::AttrUtils::GetStr(peer_node->GetOpDesc(), ge::REF_VAR_SRC_VAR_NAME, ref_origin_name);
    ref_origin_name_set.emplace(ref_origin_name);
    for (auto &node : out_nodes) {
      (void)ge::AttrUtils::GetStr(node->GetOpDesc(), ge::REF_VAR_SRC_VAR_NAME, ref_origin_name);
      ref_origin_name_set.emplace(ref_origin_name);
    }
    if (peer_node->GetType() != NO_OP) {
      continue;
    }
    if (trans_info_ptr_->dst_node_ptr->GetType() == NETOUTPUT) {
      continue;
    }
    auto no_op_in_ctrl_anchor = peer_node->GetInControlAnchor();
    if (no_op_in_ctrl_anchor == nullptr || no_op_in_ctrl_anchor->GetPeerOutControlAnchors().size() == 0) {
      continue;
    }
    for (const auto &no_op_peer_out_ctrl_anchor : no_op_in_ctrl_anchor->GetPeerOutControlAnchors()) {
      auto no_op_peer_node = no_op_peer_out_ctrl_anchor->GetOwnerNode();
      if (no_op_peer_node == nullptr) {
        continue;
      }
      (void)ge::AttrUtils::GetStr(no_op_peer_node->GetOpDesc(), ge::REF_VAR_SRC_VAR_NAME, ref_origin_name);
      ref_origin_name_set.emplace(ref_origin_name);
    }
  }
  bool insert_node_before_flage = false;
  const auto src_peer_input_nodes = trans_info_ptr_->src_node_ptr->GetInDataNodes();
  if (!src_peer_input_nodes.empty()) {
    if (src_peer_input_nodes.at(0)->GetType() == TRANSDATA) {
      const auto transdata_input_nodes = src_peer_input_nodes.at(0)->GetInDataNodes();
      insert_node_before_flage = ref_origin_name_set.count(transdata_input_nodes.at(0)->GetName()) != 0;
    } else {
      insert_node_before_flage = ref_origin_name_set.count(src_peer_input_nodes.at(0)->GetName()) != 0;
    }
  }
  insert_node_before_flage = insert_node_before_flage ||
      ref_origin_name_set.count(trans_info_ptr_->src_node_ptr->GetName());
  return insert_node_before_flage;
}

void TransNodeBaseGenerator::RefreshSourceTransInfo(ge::NodePtr src_node) const {
  trans_info_ptr_->src_op_desc = src_node->GetOpDesc();
  trans_info_ptr_->src_node_ptr = src_node;
  trans_info_ptr_->src_anchor = src_node->GetOutDataAnchor(0);

  uint32_t src_anchor_index = static_cast<uint32_t>(trans_info_ptr_->src_anchor->GetIdx());
  trans_info_ptr_->src_out_tensor_desc_ptr = trans_info_ptr_->src_op_desc->GetOutputDescPtr(src_anchor_index);
  if (trans_info_ptr_->src_out_tensor_desc_ptr == nullptr) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][RefreshSourceTransInfo] src_out_tensor_desc_ptr is null.");
    return;
  }

  /* Not need to update c0, cause we don't set c0 to node like cast,
   * so src node may don't have c0 */
  auto src_out_format = trans_info_ptr_->src_out_tensor_desc_ptr->GetFormat();
  trans_info_ptr_->src_out_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(src_out_format));
  if (FE_ORIGIN_FORMAT_SET.count(trans_info_ptr_->src_out_primary_format) != 0) {
    trans_info_ptr_->src_out_c0_format = 0;
  }
  trans_info_ptr_->src_out_sub_format = static_cast<ge::Format>(ge::GetSubFormat(src_out_format));
  trans_info_ptr_->src_out_data_type = trans_info_ptr_->src_out_tensor_desc_ptr->GetDataType();
  trans_info_ptr_->src_out_shape = trans_info_ptr_->src_out_tensor_desc_ptr->GetShape();
  trans_info_ptr_->src_out_range = GetShapeRange(*trans_info_ptr_->src_out_tensor_desc_ptr);

  trans_info_ptr_->src_op_desc_type = trans_info_ptr_->src_op_desc->GetType();

  trans_info_ptr_->is_source_weight = CheckOpConstOrVariableInOriGraph(trans_info_ptr_->src_op_desc);

  uint64_t strategy_ext_val = CalcStrategyIdExtraVal(trans_info_ptr_);
  uint64_t strategy_id =
      CalcStrategyId(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->dst_in_primary_format, strategy_ext_val);

  trans_info_ptr_->strategy_id = strategy_id;
}

Status TransNodeBaseGenerator::TransformDimsWithFormat(bool increasing_flag) const {
  if (increasing_flag) {
    if (trans_info_ptr_->src_out_shape.IsUnknownDimNum()) {
      FE_LOGD("The shape of output [%d] of op (name [%s] type [%s]) is unknown, do not need to pad shape to 4 dims.",
              trans_info_ptr_->src_anchor->GetIdx(), trans_info_ptr_->src_op_desc->GetName().c_str(),
              trans_info_ptr_->src_op_desc->GetType().c_str());
      return SUCCESS;
    }
    ExpandDimension(trans_info_ptr_->src_out_primary_format, trans_info_ptr_->dst_in_primary_format,
                    trans_info_ptr_->dst_reshape_type, trans_info_ptr_->src_out_shape);
    trans_info_ptr_->src_out_range = GetShapeRange(*trans_info_ptr_->src_out_tensor_desc_ptr);
    FE_LOGD(
        "The size of output [%d] of op (name [%s] type [%s]) is less than 4. Size is [%lu]. "
        "Now fill in the dims with value[1] until size reaches 4.",
        trans_info_ptr_->src_anchor->GetIdx(), trans_info_ptr_->src_op_desc->GetName().c_str(),
        trans_info_ptr_->src_op_desc->GetType().c_str(), trans_info_ptr_->src_out_shape.GetDimNum());
  }

  return SUCCESS;
}

bool TransNodeBaseGenerator::TransNodeCheckAccuracySupported(const ge::OpDescPtr &op_desc_ptr,
                                                             bool real_query) const {
  FE_CHECK_NOTNULL(fe_ops_store_info_ptr_);

  /* Check trans-nodes supported in cache */
  if (fe_ops_store_info_ptr_->CheckAccuracySupportByCache(op_desc_ptr)) {
    return true;
  }
  std::string un_supported_reason;
  bool ret = fe_ops_store_info_ptr_->CheckAccuracySupported(op_desc_ptr, un_supported_reason, real_query);
  /* Store the result of check accuracy support for trans-nodes. */
  fe_ops_store_info_ptr_->StoreCheckSuportResultForTransNodes(op_desc_ptr, ret);
  return ret;
}

uint64_t TransNodeBaseGenerator::GetTransAtomicId() {
  static std::atomic<uint64_t> global_trans_atomic_id(0);
  return global_trans_atomic_id.fetch_add(1, std::memory_order_relaxed);
}

bool TransNodeBaseGenerator::Is3DFormat(const ge::Format &format) {
  return kBaseFormatOf3D.find(format) != kBaseFormatOf3D.end();
}

bool TransNodeBaseGenerator::TransNodeCheckSupportedByFormatTune(ge::ComputeGraph &fused_graph,
                                                                 const ge::OpDescPtr &op_desc_ptr) {
  bool is_need_re_precompile = false;
  if (ge::AttrUtils::GetBool(fused_graph, NEED_RE_PRECOMPILE, is_need_re_precompile) &&
      is_need_re_precompile) {
    FE_LOGD("[FormatTune][ShapeTrans][CheckSupported] Graph[%s] get attr NEED_RE_PRECOMPILE.",
            fused_graph.GetName().c_str());
    if (!TransNodeCheckAccuracySupported(op_desc_ptr, true)) {
      return false;
    } else {
      (void)ge::AttrUtils::SetBool(op_desc_ptr, NEED_RE_PRECOMPILE, true);
    }
  }
  return true;
}
}  // namespace fe
