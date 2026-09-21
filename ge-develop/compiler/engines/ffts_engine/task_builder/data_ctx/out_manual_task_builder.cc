/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "out_manual_task_builder.h"
#include "graph/debug/ge_attr_define.h"

namespace ffts {
OutTaskBuilder::OutTaskBuilder() {}

OutTaskBuilder::OutTaskBuilder(CACHE_OPERATION operation)
    : DataTaskBuilder(operation) {}

OutTaskBuilder::~OutTaskBuilder() {}

Status OutTaskBuilder::UptSuccListOfRelatedNode(const ge::NodePtr &node, const std::vector<uint32_t> &succ_list,
                                                int &data_ctx_id,
                                                domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  /* If original context size is 9 and after generate this out data context,
   * the total size is 10. So the context id of this out data context is 10 - 1 = 9; */
  data_ctx_id = ffts_plus_task_def->ffts_plus_ctx_size() - 1;
  FFTS_LOGD("Data context ID is %d.", data_ctx_id);
  vector<uint32_t> target_ctx_ids;
  if (operation_ == CACHE_OPERATION::WRITE_BACK) {
    auto op = node->GetOpDesc();
    int ctx_id = -1;
    if (ge::AttrUtils::GetInt(op, kContextId, ctx_id)) {
      target_ctx_ids.emplace_back(ctx_id);
    }
  } else {
    target_ctx_ids = succ_list;
  }
  for (auto target_id : target_ctx_ids) {
    UpdateSuccList(data_ctx_id, target_id, ffts_plus_task_def);
  }
  return SUCCESS;
}
Status OutTaskBuilder::GetSuccessorContextIdSpecial(uint32_t out_anchor_index, const ge::NodePtr &node,
                                                    std::vector<uint32_t> &succ_list, uint32_t &cons_cnt) {
  auto anchors = node->GetAllOutDataAnchors();
  auto output_size = anchors.size();
  if (out_anchor_index >= output_size) {
    REPORT_FFTS_ERROR("[GenTask][DataTskBuilder][GetSuccessorContextIdSpecial]Output "
        "anchor index %u >= output size %zu of %s.", out_anchor_index, output_size, node->GetName().c_str());
    return FAILED;
  }

  auto output_anchor = anchors.at(out_anchor_index);
  if (output_anchor == nullptr) {
    return SUCCESS;
  }
  for (const auto &peer_in_anchor : output_anchor->GetPeerInDataAnchors()) {
    FFTS_CHECK_NOTNULL(peer_in_anchor);
    auto peer_in_node = peer_in_anchor->GetOwnerNode();
    FFTS_CHECK_NOTNULL(peer_in_node);
    bool no_padding_continuous_input_flag = false;
    (void)ge::AttrUtils::GetBool(peer_in_node->GetOpDesc(), ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                                 no_padding_continuous_input_flag);
    if (no_padding_continuous_input_flag) {
      FFTS_LOGD("Node %s's output %u is memory-continuous operation", node->GetName().c_str(), out_anchor_index);
      for (auto &py_in_anchor : peer_in_node->GetAllInDataAnchors()) {
        FFTS_CHECK_NOTNULL(py_in_anchor);
        auto src_out_anchor = py_in_anchor->GetPeerOutAnchor();
        FFTS_CHECK_NOTNULL(src_out_anchor);
        auto py_in_node = src_out_anchor->GetOwnerNode();
        auto py_in_node_idx = src_out_anchor->GetIdx();
        FFTS_LOGD("Py op %s' peer input op %s's output %d need become the pred of invalidate which node %s's output %u.",
                  peer_in_node->GetName().c_str(),
                  py_in_node->GetName().c_str(), py_in_node_idx, node->GetName().c_str(), out_anchor_index);
        GetSuccessorContextId(py_in_node_idx, py_in_node, succ_list, cons_cnt);
      }
    } else {
      GetSuccessorContextId(out_anchor_index, node, succ_list, cons_cnt);
    }
  }
  return SUCCESS;
}

Status OutTaskBuilder::GetPeerInputContextId(const ge::NodePtr &node, const ge::NodePtr &peer_in_node,
                                             vector<uint32_t> &peer_in_context_id) {
  uint32_t ctx_id_tmp = 0;
  if (IsPhonyOp(peer_in_node->GetOpDesc())) {
    FFTS_LOGD("Peer input operation for %s is PhonyConcat %s.", node->GetName().c_str(), peer_in_node->GetName().c_str());
    // special process for concat(no task)->phonyConnnat->mish->transdata
    //                     concat(no task) - /              \ - transdata
    auto peer_nodes = peer_in_node->GetOutDataNodes();
    bool has_two_layer_phony_op = false;
    if (!peer_nodes.empty()) {
      ge::NodePtr peer_peer_node = peer_nodes.at(0);
      FFTS_CHECK_NOTNULL(peer_peer_node);
      has_two_layer_phony_op = (kPhonyTypes.count(peer_peer_node->GetType()) != 0);
    }
    FFTS_LOGD("The has_two_layer_phony_op flag is %d.", has_two_layer_phony_op);
    if (!has_two_layer_phony_op) {
      peer_nodes = peer_in_node->GetOutAllNodes();
    }
    for (const auto &peer_node_of_pc : peer_nodes) {
      auto peer_op_of_pc = peer_node_of_pc->GetOpDesc();
      if (!ge::AttrUtils::GetInt(peer_op_of_pc, kContextId, ctx_id_tmp)) {
        FFTS_LOGI("PhonyConcat %s peer op %s requires a successor list but does not have a context ID.",
                  peer_in_node->GetName().c_str(), peer_op_of_pc->GetName().c_str());
        if (has_two_layer_phony_op) {
          for (const auto &peer_peer_node_of_pc : peer_node_of_pc->GetOutDataNodes()) {
            auto peer_peer_op_of_pc = peer_peer_node_of_pc->GetOpDesc();
            if (!ge::AttrUtils::GetInt(peer_peer_op_of_pc, kContextId, ctx_id_tmp)) {
              FFTS_LOGI("The 2nd layer PhonyConcat %s peer op %s needs a successor list but does not "
                        "have a context id.", peer_op_of_pc->GetName().c_str(), peer_peer_op_of_pc->GetName().c_str());
              continue;
            }
            FFTS_LOGD("Peer input op for the 2nd layer PhonyConcat is %s; context ID is %u.",
                      peer_peer_op_of_pc->GetName().c_str(), ctx_id_tmp);
            peer_in_context_id.emplace_back(ctx_id_tmp);
          }
        }
        continue;
      }
      FFTS_LOGD("Peer input op for PhonyConcat is '%s', with context ID %u.",
                peer_op_of_pc->GetName().c_str(), ctx_id_tmp);
      peer_in_context_id.emplace_back(ctx_id_tmp);
    }
  } else {
    if (!ge::AttrUtils::GetInt(peer_in_node->GetOpDesc(), kContextId, ctx_id_tmp)) {
      FFTS_LOGI("Node %s needs a successor list but its current successor %s does not have a context ID.",
                node->GetName().c_str(), peer_in_node->GetName().c_str());
      return NOT_CHANGED;
    }
    FFTS_LOGD("Peer input operation for %s is %s, with context ID %u.",
              node->GetName().c_str(), peer_in_node->GetName().c_str(), ctx_id_tmp);
    peer_in_context_id.emplace_back(ctx_id_tmp);
  }
  return SUCCESS;
}

Status OutTaskBuilder::GetSuccessorContextId(uint32_t out_anchor_index, const ge::NodePtr &node,
                                             std::vector<uint32_t> &succ_list, uint32_t &cons_cnt) {
  auto anchors = node->GetAllOutDataAnchors();
  auto output_size = anchors.size();
  if (out_anchor_index >= output_size) {
    REPORT_FFTS_ERROR("[GenTask][DataTskBuilder][GetSuccList] Output anchor index %u is greater than or equal to the output size %zu for %s.",
                      out_anchor_index, output_size, node->GetName().c_str());
    return FAILED;
  }

  auto output_anchor = anchors.at(out_anchor_index);
  if (output_anchor == nullptr) {
    return SUCCESS;
  }
  for (const auto &peer_in_anchor : output_anchor->GetPeerInDataAnchors()) {
    FFTS_CHECK_NOTNULL(peer_in_anchor);
    auto peer_in_node = peer_in_anchor->GetOwnerNode();
    FFTS_CHECK_NOTNULL(peer_in_node);
    vector<uint32_t> peer_in_context_id;
    auto peer_op = peer_in_node->GetOpDesc();
    /* PhonyConcat is no-task op. We need to penetrate it
     * and find its successors. */
    Status ret = GetPeerInputContextId(node, peer_in_node, peer_in_context_id);
    if (ret == NOT_CHANGED) {
      continue;
    } else if (ret != SUCCESS) {
      FFTS_LOGW("Failed to get peer input with context ID: %s.", peer_op->GetNamePtr());
      return FAILED;
    }

    for (auto ele : peer_in_context_id) {
      succ_list.emplace_back(ele);
      cons_cnt++;
    }
    FFTS_LOGD("Total successors(%zu) for node %s is %s.", succ_list.size(), node->GetName().c_str(),
              fe::StringUtils::IntegerVecToString(succ_list).c_str());
  }
  return SUCCESS;
}

bool OutTaskBuilder::CheckManualDataAttr(const ge::OpDescPtr &op_desc, vector<int64_t> &output_addrs) {
  uint32_t context_id = 0;
  if (!ge::AttrUtils::GetInt(op_desc, kContextId, context_id)) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][FillCtxt][node %s, type %s] Unable to obtain context ID for this node.",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return false;
  }
  if (!ge::AttrUtils::GetListInt(op_desc, "output_addrs", output_addrs)) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][FillCtxt][node %s, type %s] Attr output_addrs is empty.",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return false;
  }
  return true;
}

Status OutTaskBuilder::FillManualDataCtx(size_t out_anchor_index, const ge::NodePtr &node,
                                         const DataContextParam &param, domi::FftsPlusTaskDef *ffts_plus_task_def,
                                         domi::FftsPlusDataCtxDef *data_ctx_def) {
  auto op_desc = node->GetOpDesc();
  vector<int64_t> output_addrs;
  if (!CheckManualDataAttr(op_desc, output_addrs)) {
    return FAILED;
  }
  data_ctx_def->set_aten(0);
  std::vector<uint32_t> succ_list;
  if (operation_ == CACHE_OPERATION::WRITE_BACK) {
    /* In write back case, the write back data context can only be
     * executed when the only producer has already been executed. */
    data_ctx_def->set_cnt_init(1);
    data_ctx_def->set_cnt(1);
    FFTS_LOGD("Fill write back for node %s, consumer count is 1.", node->GetName().c_str());
  } else {
    /* In invalidate case, the invalidate data context can only be
     * executed when all the consumers have already been executed. */
    uint32_t cons_cnt = 0;
    bool is_first_slice = false;
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kFftsFirstSliceFlag, is_first_slice);
    if (is_first_slice) {
      GetSuccessorContextIdSpecial(out_anchor_index, node, succ_list, cons_cnt);
    } else {
      GetSuccessorContextId(out_anchor_index, node, succ_list, cons_cnt);
    }

    uint32_t size = static_cast<uint32_t>(succ_list.size());
    data_ctx_def->set_cnt_init(size);
    data_ctx_def->set_cnt(size);
    data_ctx_def->set_orig_consumer_counter(size);
    data_ctx_def->set_run_consumer_counter(size);
    FFTS_LOGD("Fill invalidate for node %s, with consumer count of %u.", node->GetName().c_str(), cons_cnt);
  }
  int32_t data_ctx_id = -1;
  UptSuccListOfRelatedNode(node, succ_list, data_ctx_id, ffts_plus_task_def);

  if (out_anchor_index >= output_addrs.size()) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][FillCtxt][node %s, type %s] out anchor %zu is greater than or equal to the size of output_addrs %zu.",
                      node->GetName().c_str(), node->GetType().c_str(), out_anchor_index, output_addrs.size());
    return FAILED;
  }
  data_ctx_def->set_addr_base(output_addrs[out_anchor_index]);
  data_ctx_def->set_addr_offset(param.base_addr_offset);
  FFTS_LOGD("Base Addr for output %zu is %ld, offset is %ld", out_anchor_index, output_addrs[out_anchor_index],
            param.base_addr_offset);

  if (operation_ == CACHE_OPERATION::INVALIDATE &&
      GenInvalidSuccListWithMemReuse(node, out_anchor_index, ffts_plus_task_def, data_ctx_id, 0) != SUCCESS) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][FillCtxt][node %s, type %s] GenInvalidSuccList failed, data_ctx_id: %d.",
                      node->GetName().c_str(), node->GetType().c_str(), data_ctx_id);
    return FAILED;
  }
  FillManualThreadingParam(param, data_ctx_def);
  FFTS_LOGI("Finished generating %d contexts for node %s.", static_cast<int>(operation_), node->GetName().c_str());
  return SUCCESS;
}

Status OutTaskBuilder::UpdateInvalidCtxWithMemReuse(const ge::NodePtr &node, int &data_ctx_id,
                                                    const size_t &window_id,
                                                    domi::FftsPlusTaskDef *ffts_plus_task_def) {
  (void)window_id;
  ge::NodePtr phony_op = nullptr;
  FFTS_LOGD("[GenTsk][DataTsk][UpdateInvalidCtxWithMemReuse] [node %s, type %s] Entering UpdateInvalidCtxWithMemReuse.",
            node->GetName().c_str(), node->GetType().c_str());

  for (auto &out_node : node->GetOutDataNodes()) {
    bool is_nopading_input_continuous = false;
    (void)ge::AttrUtils::GetBool(out_node->GetOpDesc(), "_no_padding_continuous_input", is_nopading_input_continuous);
    if (is_nopading_input_continuous) {
      FFTS_LOGI("[GenTsk][DataTsk][UpdateInvalidCtxWithMemReuse][node %s, type %s] has _no_padding_continuous_input.",
                node->GetName().c_str(), node->GetType().c_str());
      phony_op = out_node;
      break;
    }
  }
  if (phony_op != nullptr) {
    for (auto &reuse_node : phony_op->GetInDataNodes()) {
      int ctx_id = -1;
      if (!ge::AttrUtils::GetInt(reuse_node->GetOpDesc(), kContextId, ctx_id)) {
        FFTS_LOGI("[GenTsk][DataTsk][UpdateInvalidCtxWithMemReuse][node %s, type %s] has no context id.",
                  node->GetName().c_str(), node->GetType().c_str());
        return FAILED;
      }

      UpdateSuccList(ctx_id, data_ctx_id, ffts_plus_task_def);
      UpdatePreCnt(ctx_id, ffts_plus_task_def, 1);
    }
  } else {
    int ctx_id = -1;
    auto reuse_op_desc = node->GetOpDesc();
    if (!ge::AttrUtils::GetInt(reuse_op_desc, kContextId, ctx_id)) {
      FFTS_LOGI("[GenTsk][DataTsk][UpdateInvalidCtxWithMemReuse][node %s, type %s] has no context id.",
                node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }

    UpdateSuccList(ctx_id, data_ctx_id, ffts_plus_task_def);
    UpdatePreCnt(ctx_id, ffts_plus_task_def, 1);
  }
  return SUCCESS;
}
}
