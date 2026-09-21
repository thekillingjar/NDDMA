/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generate_cmo_type_invalid.h"
#include "common/fe_log.h"
#include "common/fe_op_info_common.h"
#include "common/configuration.h"
#include "common/aicore_util_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "proto/task.pb.h"

namespace fe {
GenerateCMOTypeInvalid::GenerateCMOTypeInvalid()
    : GenerateCMOTypeBase() {
}

bool GenerateCMOTypeInvalid::CheckReadDistance(const ge::OpDescPtr &op_desc,
                                               const ge::InDataAnchorPtr &in_anchor) const {
  auto op_name    = op_desc->GetName();
  auto op_type    = op_desc->GetType();
  auto in_idx     = in_anchor->GetIdx();
  auto input_desc = op_desc->MutableInputDesc(in_idx);
  if (!ge::AttrUtils::HasAttr(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE)) {
    return false;
  }

  vector<int32_t> data_visit_dist_vec;
  (void)ge::AttrUtils::GetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, data_visit_dist_vec);
  int32_t data_vist_dist_from_pre_node = INT_MAX;
  if (!data_visit_dist_vec.empty()) {
    data_vist_dist_from_pre_node = data_visit_dist_vec[0];
  }
  FE_LOGD("Op[name=%s,type=%s,input=%s]: visit pre node distance:%d, threshold:%d", op_name.c_str(), op_type.c_str(),
          op_desc->GetName().c_str(), data_vist_dist_from_pre_node, kDataVisitDistThreshold);
  return data_vist_dist_from_pre_node < kDataVisitDistThreshold;
}

bool GenerateCMOTypeInvalid::IsLastPeerOutNode(const ge::InDataAnchorPtr &inanchor, const ge::NodePtr &node) const {
  int64_t cur_stream_index = -1;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, cur_stream_index);
  auto peer_out_anchor = inanchor->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(peer_out_anchor);
  for (auto &peer_in_node_anchor : peer_out_anchor->GetPeerInDataAnchors()) {
    auto owner_node = peer_in_node_anchor->GetOwnerNode();
    if (owner_node == nullptr) {
      continue;
    }
    int64_t tmp_stream_index = -1;
    (void)ge::AttrUtils::GetInt(owner_node->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, tmp_stream_index);
    if (tmp_stream_index > cur_stream_index) {
      return false;
    }
  }
  return true;
}

void GenerateCMOTypeInvalid::LabeledInvalidOrBarrier(const ge::NodePtr &src_node,
                                                     const CmoAttr &attr,
                                                     const std::string &cmo_type) const {
  std::vector<CmoAttr> vec_attr{};
  auto src_op_desc = src_node->GetOpDesc();
  vec_attr.emplace_back(attr);
  AddToNodeCmoAttr(src_op_desc, cmo_type, vec_attr);
  FE_LOGD("Op[name=%s, type=%s] for Op[name=%s, object=%s, index=%d], add label:%s",
          src_op_desc->GetName().c_str(), src_op_desc->GetType().c_str(),
          attr.node->GetOpDesc()->GetName().c_str(),
          (attr.object == CmoTypeObject::OUTPUT ? "output" : "workspace"),
          attr.object_index, cmo_type.c_str());
}

bool GenerateCMOTypeInvalid::CheckDiffStreamReuseDistance(const int64_t &src_stream_id, const int64_t &dst_stream_id,
                                                          const ge::NodePtr &src_node, const ge::NodePtr &dst_node,
                                                          const StreamCtrlMap &stream_ctrls) const {
  int64_t src_node_id = -1;
  int64_t dst_node_id = -1;
  (void)ge::AttrUtils::GetInt(src_node->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, src_node_id);
  (void)ge::AttrUtils::GetInt(dst_node->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, dst_node_id);
  int64_t max_reuse_distance = 0;
  for (auto &stream_ctl : stream_ctrls) {
    if (stream_ctl.second.src_stream_id == src_stream_id && stream_ctl.second.dst_stream_id == dst_stream_id) {
      if (stream_ctl.second.src_ctrl_node == nullptr || stream_ctl.second.dst_ctrl_node == nullptr) {
        continue;
      }
      int64_t src_ctlnode_id = -1;
      int64_t dst_ctlnode_id = -1;
      (void)ge::AttrUtils::GetInt(stream_ctl.second.src_ctrl_node->GetOpDesc(),
                                  ge::ATTR_NAME_OP_READ_WRITE_INDEX, src_ctlnode_id);
      (void)ge::AttrUtils::GetInt(stream_ctl.second.dst_ctrl_node->GetOpDesc(),
                                  ge::ATTR_NAME_OP_READ_WRITE_INDEX, dst_ctlnode_id);
      if (src_ctlnode_id >= src_node_id && dst_ctlnode_id <= dst_node_id) {
        FE_LOGD("Src node:%s, dst node:%s. dst node id:%ld, dst ctlnode id:%ld, src node id:%ld, src ctlnode id:%ld",
                src_node->GetName().c_str(), dst_node->GetName().c_str(),
                dst_node_id, dst_ctlnode_id, src_node_id, src_ctlnode_id);
        int64_t cur_distance = dst_node_id - dst_ctlnode_id + src_ctlnode_id - src_node_id;
        max_reuse_distance = max(max_reuse_distance, cur_distance);
      }
    }
  }
  FE_LOGD("Max reuse distance is %ld.", max_reuse_distance);
  int32_t reuse_threshold = Configuration::Instance(AI_CORE_NAME).GetMemReuseDistThreshold();
  return max_reuse_distance >= static_cast<int64_t>(reuse_threshold);
}

bool GenerateCMOTypeInvalid::CheckReuseDistance(const ge::NodePtr &src_node,
                                                const ge::NodePtr &dst_node,
                                                const ge::NodePtr &last_use_node,
                                                const StreamCtrlMap &stream_ctrls) const {
  int64_t src_stream_index = -1;
  int64_t dst_stream_index = -1;
  (void)ge::AttrUtils::GetInt(last_use_node->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, src_stream_index);
  (void)ge::AttrUtils::GetInt(dst_node->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, dst_stream_index);
  int32_t reuse_threshold = Configuration::Instance(AI_CORE_NAME).GetMemReuseDistThreshold();
  FE_LOGD("SrcOp[name=%s, type=%s] has been reused by DstOp[name=%s, type=%s], UseOp[name=%s, type=%s]",
          src_node->GetOpDesc()->GetName().c_str(), src_node->GetOpDesc()->GetType().c_str(),
          dst_node->GetOpDesc()->GetName().c_str(), dst_node->GetOpDesc()->GetType().c_str(),
          last_use_node->GetOpDesc()->GetName().c_str(), last_use_node->GetOpDesc()->GetType().c_str());

  uint32_t src_stream_id = last_use_node->GetOpDesc()->GetStreamId();
  uint32_t dst_stream_id = dst_node->GetOpDesc()->GetStreamId();
  FE_LOGD("UseOp stream_index:%ld, stream_id:%u, DstOp stream_index:%ld, stream_id:%u, threshold:%d",
          src_stream_index, src_stream_id, dst_stream_index, dst_stream_id, reuse_threshold);

  if (src_stream_id != dst_stream_id) {
    return CheckDiffStreamReuseDistance(src_stream_id, dst_stream_id, last_use_node, dst_node, stream_ctrls);
  }
  return (dst_stream_index - src_stream_index) >= reuse_threshold;
}

void GenerateCMOTypeInvalid::CheckReuseDistanceAndLabeled(const ge::NodePtr &node,
                                                          const ge::NodePtr &pre_node,
                                                          const CmoTypeObject cmo_type_obj,
                                                          const int32_t index,
                                                          const StreamCtrlMap &stream_ctrls) const {
  const ge::NodePtr& dst_node = (cmo_type_obj == CmoTypeObject::OUTPUT ? pre_node : node);
  CmoAttr attr{dst_node, cmo_type_obj, index};
  auto dst_op_desc = dst_node->GetOpDesc();
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info{};
  mem_reuse_info = dst_op_desc->TryGetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);
  if (mem_reuse_info.empty()) {
    FE_LOGD("Op[name=%s, type=%s] no mem reuse attr", dst_op_desc->GetName().c_str(),
            dst_op_desc->GetType().c_str());
    LabeledInvalidOrBarrier(node, attr, kCmoInvalid);
    return;
  }
  std::string tmp_key = (cmo_type_obj == CmoTypeObject::OUTPUT ? "output" : "workspace");
  std::string mem_info_key = tmp_key + std::to_string(index);
  if (mem_reuse_info.find(mem_info_key) == mem_reuse_info.end() || mem_reuse_info[mem_info_key].empty()) {
    FE_LOGD("Op[name=%s, type=%s, key=%s] gets no mem reuse", dst_op_desc->GetName().c_str(),
            dst_op_desc->GetType().c_str(), mem_info_key.c_str());
    LabeledInvalidOrBarrier(node, attr, kCmoInvalid);
    return;
  }
  auto &all_reuse_info = mem_reuse_info[mem_info_key];
  ge::MemReuseInfo &first_reuse_info = all_reuse_info[0];
  ge::NodePtr &mem_reuse_node = first_reuse_info.node;
  for (auto &reuse_info : all_reuse_info) {
    ge::NodePtr &reuse_node = reuse_info.node;
    std::vector<ge::NodePtr> continues_output_node {reuse_node};
    auto out_nodes = reuse_node->GetOutDataNodes();
    if (reuse_info.index < static_cast<uint32_t>(out_nodes.size())) {
      ge::NodePtr next_node = out_nodes.at(reuse_info.index);
      if (IsNoTaskOp(next_node)) {
        for (auto &in_node : next_node->GetInDataNodes()) {
          continues_output_node.emplace_back(in_node);
        }
      }
    }
    for (auto &tmp_node : continues_output_node) {
      if (tmp_node->GetOpDesc()->GetId() < mem_reuse_node->GetOpDesc()->GetId()) {
        mem_reuse_node = tmp_node;
      }
    }
  }
  if (!IsAiCoreOp(mem_reuse_node)) {
    return;
  }
  if (!CheckReuseDistance(dst_node, mem_reuse_node, node, stream_ctrls)) {
    FE_LOGD("Not set invalid or barrier label.");
    return;
  }
  for (auto &tmp_node : mem_reuse_node->GetOutAllNodes()) {
    if (IsNoTaskOp(tmp_node)) {
      FE_LOGD("Node:%s with continues memory, not set invalid or barrier label.", tmp_node->GetName().c_str());
      return;
    }
  }
  LabeledInvalidOrBarrier(node, attr, kCmoInvalid);
  LabeledInvalidOrBarrier(mem_reuse_node, attr, kCmoBarrier);
}

bool GenerateCMOTypeInvalid::CheckInputNeedGenerate(const ge::NodePtr &node,
                                                    const ge::InDataAnchorPtr &in_anchor) const {
  auto op_desc_ptr = node->GetOpDesc();
  if (!CheckParentOpIsAiCore(in_anchor)) {
    FE_LOGD("Current Op[name=%s, type=%s,input=%d] has no parent or parent is not aicore op",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), in_anchor->GetIdx());
    return false;
  }

  if (IsPeerOutNoTaskNode(in_anchor)) {
    FE_LOGD("Op[name=%s,type=%s] pre node:%d has attr no_task", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(), in_anchor->GetIdx());
    return false;
  }

  if (Configuration::Instance(AI_CORE_NAME).IsEnableReuseMemory()) {
    if (!ReadIsLifeCycleEnd(node, in_anchor)) {
      FE_LOGD("Op[name=%s,type=%s] life_end_cycle flag is false", op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str());
      return false;
    }
  }

  if (!CheckReadDistance(op_desc_ptr, in_anchor)) {
    return false;
  }

  if (!IsLastPeerOutNode(in_anchor, node)) {
    FE_LOGD("Op[name=%s,type=%s] one output multi input, not the last one", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str());
    return false;
  }
  return true;
}

void GenerateCMOTypeInvalid::GenerateInput(const ge::NodePtr &node, const StreamCtrlMap &stream_ctrls) const {
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    if (!CheckInputNeedGenerate(node, in_data_anchor)) {
      continue;
    }

    auto out_anchor = in_data_anchor->GetPeerOutAnchor();
    auto out_node   = out_anchor->GetOwnerNode();
    auto out_idx    = out_anchor->GetIdx();
    CheckReuseDistanceAndLabeled(node, out_node, CmoTypeObject::OUTPUT, out_idx, stream_ctrls);
  }
  return;
}

void GenerateCMOTypeInvalid::GenerateWorkSpace(const ge::NodePtr &node, const StreamCtrlMap &stream_ctrls) const {
  auto op_desc_ptr = node->GetOpDesc();
  for (size_t work_idx = 0; work_idx < op_desc_ptr->GetWorkspace().size(); work_idx++) {
    CheckReuseDistanceAndLabeled(node, nullptr, CmoTypeObject::WORKSPACE, static_cast<int>(work_idx), stream_ctrls);
  }
  return;
}

bool GenerateCMOTypeInvalid::IsNeedGenerate(const ge::NodePtr &node) const {
  if (!IsAiCoreOp(node)) {
    FE_LOGD("Current Op[name=%s] is not Aicore Op", node->GetName().c_str());
    return false;
  }
  if (IsNoTaskOp(node)) {
    FE_LOGD("Current Op[name=%s] is no task op", node->GetName().c_str());
    return false;
  }
  if (node->GetType() == MEMSET_OP_TYPE || node->GetType() == ATOMIC_CLEAN_OP_TYPE) {
    FE_LOGD("Current Op[name=%s] is memset op.", node->GetName().c_str());
    return false;
  }
  return true;
}

void GenerateCMOTypeInvalid::GenerateType(const ge::NodePtr &node,
                                          const StreamCtrlMap &stream_ctrls,
                                          std::unordered_map<ge::NodePtr, ge::NodePtr> &prefetch_cache_map,
                                          std::map<uint32_t, std::map<int64_t, ge::NodePtr>> &stream_node_map) {
  (void)prefetch_cache_map;
  (void)stream_node_map;
  /**
   * input check the parent output mem reuse info
   * workspace check current node mem reuse info
   */
  FE_LOGD("Begin to generate invalid data for node: [name=%s, type=%s]", node->GetName().c_str(), node->GetType().c_str());
  if (!IsNeedGenerate(node)) {
    FE_LOGD("No need to generate invalid data for node:[name=%s, type=%s]", node->GetName().c_str(), node->GetType().c_str());
    return;
  }
  GenerateInput(node, stream_ctrls);
  GenerateWorkSpace(node, stream_ctrls);
  FE_LOGD("Ending generate invalid data for node:[name=%s, type=%s]", node->GetName().c_str(), node->GetType().c_str());
  return;
}
} // namespace fe
