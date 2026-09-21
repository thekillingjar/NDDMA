/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fftsplus_task_builder.h"
#include <securec.h>
#include <string>
#include "inc/ffts_log.h"
#include "inc/ffts_type.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "rt_error_codes.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"
#include "inc/ffts_utils.h"

namespace ffts {
namespace {
void FillComCtxForNode(const ge::NodePtr &up_node, uint32_t recurise_cnt,
                       FftsPlusComCtx_t &sub_ffts_plus_context_elem) {
  if (recurise_cnt == 0) {
    FFTS_LOGI("Recursion count reduced to 0, will not continue processing.");
    return;
  }
  ge::OpDescPtr up_op_desc = up_node->GetOpDesc();
  if (up_op_desc == nullptr) {
    return;
  }
  bool has_ctx_id = up_op_desc->HasAttr(kContextId);
  if (IsPhonyOp(up_op_desc) || !has_ctx_id) {
    FFTS_LOGD("Node name:%s, type:%s, has_ctx_id:%d.", up_op_desc->GetName().c_str(), up_op_desc->GetType().c_str(),
              has_ctx_id);
    for (const auto &peer_node : up_node->GetOutAllNodes()) {
      FFTS_CHECK(peer_node == nullptr, FFTS_LOGW("The peer_node is nullptr."), return);
      FFTS_LOGD("Node name:%s, type:%s, out node name:%s, type:%s.", up_op_desc->GetName().c_str(),
                up_op_desc->GetType().c_str(), peer_node->GetName().c_str(), peer_node->GetType().c_str());
      FillComCtxForNode(peer_node, recurise_cnt - 1, sub_ffts_plus_context_elem);
    }
    return;
  }

  if (kHCCLOpType.count(up_op_desc->GetType()) > 0) {
    return;
  }
  uint32_t context_id = 0;
  if (ge::AttrUtils::GetInt(up_op_desc, kContextId, context_id)) {
    FFTS_LOGD("Out node name: %s, node type: %s, added succlist with context_id: %u.", up_op_desc->GetName().c_str(),
              up_op_desc->GetType().c_str(), context_id);
    sub_ffts_plus_context_elem.succ_list.emplace_back(context_id);
  } else {
    FFTS_LOGI("Out node name: %s, node type: %s. get context id unsuccessful.", up_op_desc->GetName().c_str(),
              up_op_desc->GetType().c_str());
  }

  return;
}
} //namespace

FFTSPlusTaskBuilder::FFTSPlusTaskBuilder() {}

FFTSPlusTaskBuilder::~FFTSPlusTaskBuilder() {}

Status FFTSPlusTaskBuilder::GetFirstAvailableLabel(domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                   domi::FftsPlusLabelCtxDef *pred_label_ctx,
                                                   domi::FftsPlusLabelCtxDef **avl_label_context,
                                                   uint32_t &recursion_count) {
  recursion_count++;
  if (recursion_count > kGetFirstAvailableLabel) {
    FFTS_LOGE("The count of GetFirstAvailableLabel recursions has reached %d; now stopping recursion.", recursion_count);
    return FAILED;
  }
  FFTS_LOGD("pred_label_ctx successor_num is %u", pred_label_ctx->successor_num());
  if (pred_label_ctx->successor_num() == RT_CTX_SUCCESSOR_NUM) {
    uint32_t last_succ_id = pred_label_ctx->successor_list(RT_CTX_SUCCESSOR_NUM - 1);
    FFTS_LOGD("The pred label context is full; its last successor ID is %u.", last_succ_id);
    uint32_t ctx_size = ffts_plus_task_def->ffts_plus_ctx_size();
    if (last_succ_id >= ctx_size) {
      REPORT_FFTS_ERROR("[FFTSPlusTaskBuilder][GetFirstLabel] last_succ_id: %u, ctx_size: %u", last_succ_id, ctx_size);
      return FAILED;
    }
    domi::FftsPlusCtxDef* last_succ_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(last_succ_id));
    FFTS_CHECK_NOTNULL(last_succ_ctx);
    if (last_succ_ctx->context_type() == RT_CTX_TYPE_LABEL) {
      GetFirstAvailableLabel(ffts_plus_task_def, last_succ_ctx->mutable_label_ctx(),
                             avl_label_context, recursion_count);
    } else {
      FFTS_LOGD("The last successor is unlabeled; stop the search and generate a new label.");
      Status ret = GenerateNewLabelCtx(ffts_plus_task_def, last_succ_id, pred_label_ctx, avl_label_context);
      if (ret != SUCCESS) {
        return ret;
      }
    }
  } else {
    *avl_label_context = pred_label_ctx;
    FFTS_LOGD("Stop searching for label context.");
  }

  return SUCCESS;
}

Status FFTSPlusTaskBuilder::UpdatePreCnt(uint32_t curr_id, domi::FftsPlusTaskDef *ffts_plus_task_def,
                                         const int32_t gradient) const {
  domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(curr_id);
  FFTS_CHECK_NOTNULL(ffts_plus_ctx);
  uint32_t type = ffts_plus_ctx->context_type();
  FFTS_LOGD("Update pred cnt for context id:[%u] type:[%u], gradient:[%d].", curr_id, type, gradient);
  Status ret = SUCCESS;
  switch (type) {
    case RT_CTX_TYPE_AT_START:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_at_start_ctx(), gradient);
      break;
    case RT_CTX_TYPE_AICORE:
    case RT_CTX_TYPE_AIV:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_aic_aiv_ctx(), gradient);
      break;
    case RT_CTX_TYPE_MIX_AIC:
    case RT_CTX_TYPE_MIX_AIV:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_mix_aic_aiv_ctx(), gradient);
      break;
    case RT_CTX_TYPE_AICPU:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_aicpu_ctx(), gradient);
      break;
    case RT_CTX_TYPE_SDMA:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_sdma_ctx(), gradient);
      break;
    case RT_CTX_TYPE_NOTIFY_WAIT:
    case RT_CTX_TYPE_NOTIFY_RECORD:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_notify_ctx(), gradient);
      break;
    case RT_CTX_TYPE_WRITE_VALUE:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_write_value_ctx(), gradient);
      break;
    case RT_CTX_TYPE_CASE_SWITCH:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_case_switch_ctx(), gradient);
      break;
    case RT_CTX_TYPE_COND_SWITCH:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_cond_switch_ctx(), gradient);
      break;
    case RT_CTX_TYPE_LABEL:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_label_ctx(), gradient);
      break;
    case RT_CTX_TYPE_DSA:
      ret = UpdateCtxPredCnt(ffts_plus_ctx->mutable_dsa_ctx(), gradient);
      break;
    case RT_CTX_TYPE_INVALIDATE_DATA:
    case RT_CTX_TYPE_WRITEBACK_DATA:
      ret = UpdateDataPredCnt(ffts_plus_ctx->mutable_data_ctx(), gradient);
      break;
    default:
      FFTS_LOGW("Type %u does not support updating pred cnt.", type);
  }
  return ret;
}

uint32_t FFTSPlusTaskBuilder::GetPreCnt(uint32_t curr_id, domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(curr_id);
  FFTS_CHECK_NOTNULL(ffts_plus_ctx);
  uint32_t type = ffts_plus_ctx->context_type();
  uint32_t pred_cnt = 0;
  switch (type) {
    case RT_CTX_TYPE_AT_START:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_at_start_ctx());
      break;
    case RT_CTX_TYPE_AICORE:
    case RT_CTX_TYPE_AIV:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_aic_aiv_ctx());
      break;
    case RT_CTX_TYPE_MIX_AIC:
    case RT_CTX_TYPE_MIX_AIV:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_mix_aic_aiv_ctx());
      break;
    case RT_CTX_TYPE_AICPU:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_aicpu_ctx());
      break;
    case RT_CTX_TYPE_SDMA:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_sdma_ctx());
      break;
    case RT_CTX_TYPE_NOTIFY_WAIT:
    case RT_CTX_TYPE_NOTIFY_RECORD:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_notify_ctx());
      break;
    case RT_CTX_TYPE_WRITE_VALUE:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_write_value_ctx());
      break;
    case RT_CTX_TYPE_CASE_SWITCH:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_case_switch_ctx());
      break;
    case RT_CTX_TYPE_COND_SWITCH:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_cond_switch_ctx());
      break;
    case RT_CTX_TYPE_LABEL:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_label_ctx());
      break;
    case RT_CTX_TYPE_DSA:
      pred_cnt = GetCtxPredCnt(ffts_plus_ctx->mutable_dsa_ctx());
      break;
    case RT_CTX_TYPE_INVALIDATE_DATA:
    case RT_CTX_TYPE_WRITEBACK_DATA:
      pred_cnt = GetDataPredCnt(ffts_plus_ctx->mutable_data_ctx());
      break;
    default:
      FFTS_LOGW("Type %u does not support get pred cnt.", type);
  }
  return pred_cnt;
}

Status FFTSPlusTaskBuilder::ReplaceSuccList(uint32_t succ_id, uint32_t new_succ_id, uint32_t curr_id,
                                            domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  domi::FftsPlusCtxDef* ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(curr_id);
  FFTS_CHECK_NOTNULL(ffts_plus_ctx);
  uint32_t type = ffts_plus_ctx->context_type();
  Status ret = SUCCESS;
  FFTS_LOGD("Current type[%u], attempting to replace succ_id[%u] with [%u] from context[%u].", type, succ_id, new_succ_id, curr_id);
  switch (type) {
    case RT_CTX_TYPE_AT_START:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_at_start_ctx());
      break;
    case RT_CTX_TYPE_AICORE:
    case RT_CTX_TYPE_AIV:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_aic_aiv_ctx());
      break;
    case RT_CTX_TYPE_DSA:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_dsa_ctx());
      break;
    case RT_CTX_TYPE_MIX_AIC:
    case RT_CTX_TYPE_MIX_AIV:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_mix_aic_aiv_ctx());
      break;
    case RT_CTX_TYPE_AICPU:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_aicpu_ctx());
      break;
    case RT_CTX_TYPE_SDMA:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_sdma_ctx());
      break;
    case RT_CTX_TYPE_NOTIFY_WAIT:
    case RT_CTX_TYPE_NOTIFY_RECORD:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_notify_ctx());
      break;
    case RT_CTX_TYPE_WRITE_VALUE:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_write_value_ctx());
      break;
    case RT_CTX_TYPE_CASE_SWITCH:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_case_switch_ctx());
      break;
    case RT_CTX_TYPE_LABEL:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_label_ctx());
      break;
    case RT_CTX_TYPE_INVALIDATE_DATA:
      ret = ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, ffts_plus_ctx->mutable_data_ctx());
      break;
    default:
      FFTS_LOGW("Type %u does not require removal of its successor list.", type);
      return FAILED;
  }
  return ret;
}

Status FFTSPlusTaskBuilder::UpdateSuccList(uint32_t succ_id, uint32_t curr_id,
    domi::FftsPlusTaskDef *ffts_plus_task_def, size_t thread_id, bool is_auto) const {
  domi::FftsPlusCtxDef* ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(curr_id);
  FFTS_CHECK_NOTNULL(ffts_plus_ctx);
  uint32_t type = ffts_plus_ctx->context_type();
  FFTS_LOGD("Current type: %u wants to add succ_id: %u in curr_id: %u's context.", type, succ_id, curr_id);
  switch (type) {
    case RT_CTX_TYPE_AT_START:
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_at_start_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_AICORE:
    case RT_CTX_TYPE_AIV:
      FFTS_LOGD("Update succlist for aic/aiv context with ID %u, successful ID %u.", curr_id, succ_id);
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_aic_aiv_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_DSA:
      FFTS_LOGD("Update succlist for dsa context %u, succ id %u", curr_id, succ_id);
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_dsa_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_MIX_AIC:
    case RT_CTX_TYPE_MIX_AIV:
      FFTS_LOGD("Update succlist for mix aic/aiv context with ID %u, successful item ID %u.", curr_id, succ_id);
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_mix_aic_aiv_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_AICPU:
      FFTS_LOGD("Update succlist for aicpu context %u with succ id %u", curr_id, succ_id);
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_aicpu_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_SDMA:
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_sdma_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_NOTIFY_WAIT:
    case RT_CTX_TYPE_NOTIFY_RECORD:
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_notify_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_WRITE_VALUE:
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_write_value_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_CASE_SWITCH:
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_case_switch_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_COND_SWITCH:
      {
        auto cond_switch_ctx_def = ffts_plus_ctx->mutable_cond_switch_ctx();
        FFTS_CHECK_NOTNULL(cond_switch_ctx_def);
        auto succ_num = cond_switch_ctx_def->true_successor_num();
        ++succ_num;
        FFTS_LOGD("Added one successor %u. Successor count is %u.", succ_id, succ_num);
        cond_switch_ctx_def->set_true_successor_num(succ_num);
        cond_switch_ctx_def->add_true_successor_list(succ_id);
        break;
      }
    case RT_CTX_TYPE_LABEL:
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_label_ctx(), thread_id, is_auto);
      break;
    case RT_CTX_TYPE_INVALIDATE_DATA:
      AddOneId(ffts_plus_task_def, succ_id, ffts_plus_ctx->mutable_data_ctx(), thread_id, is_auto);
      break;
    default:
      FFTS_LOGI("type %u does not require updating its successor list.", type);
      return FAILED;
  }
  return SUCCESS;
}

Status FFTSPlusTaskBuilder::GenerateTaskDef(const ge::NodePtr &node,
                                            domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("TaskBuilder::GenerateTask begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());

  ge::OpDescPtr op_desc = node->GetOpDesc();
  Status status = GenContextDef(node, ffts_plus_task_def);
  if (status != SUCCESS) {
    FFTS_LOGE("GenSubFftsTaskCommonInfo failed. Op[%s, optype[%s]].",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return status;
  }
  return SUCCESS;
}

void FFTSPlusTaskBuilder::FillSingleProducersInfo(const ge::NodePtr &pre_node, uint32_t &pred_cnt,
                                                  uint32_t recurise_cnt) const {
  ge::OpDescPtr pre_op_desc = pre_node->GetOpDesc();
  if (pre_op_desc == nullptr) {
    return;
  }
  FFTS_LOGD("Up node name:%s, node type:%s, recurise_cnt:%u.", pre_op_desc->GetName().c_str(),
            pre_op_desc->GetType().c_str(), recurise_cnt);
  if (recurise_cnt == 0) {
    FFTS_LOGI("Recursion count reduced to 0, will not continue processing.");
    return;
  }
  if (kHCCLOpType.count(pre_op_desc->GetType()) > 0) {
    int32_t hccl_output_degree_0_num = 0;
    (void)ge::AttrUtils::GetInt(pre_op_desc, kHcclOutDegree0Num, hccl_output_degree_0_num);
    pred_cnt += static_cast<uint32_t>(hccl_output_degree_0_num);
    FFTS_LOGD("FFTSPlusTaskBuilder node name:%s, node type:%s, hccl_output_degree_0_num:%d success.",
              pre_op_desc->GetName().c_str(), pre_op_desc->GetType().c_str(), hccl_output_degree_0_num);
  } else if (IsPhonyOp(pre_op_desc) || (!pre_op_desc->HasAttr(kContextId) && !pre_op_desc->HasAttr(kAutoCtxIdList))) {
    FFTS_LOGD("Node name: %s, Node type: %s, has attribute kContextId: %d, has attribute kAutoCtxIdList: %d.",
              pre_op_desc->GetName().c_str(), pre_op_desc->GetType().c_str(),
              pre_op_desc->HasAttr(kContextId), pre_op_desc->HasAttr(kAutoCtxIdList));
    for (const auto &in_node : pre_node->GetInAllNodes()) {
      FFTS_LOGD("Node name: %s, node type: %s, in-node name: %s, in-node type: %s.", pre_op_desc->GetName().c_str(),
                pre_op_desc->GetType().c_str(), in_node->GetName().c_str(), in_node->GetType().c_str());
      FillSingleProducersInfo(in_node, pred_cnt, recurise_cnt - 1);
    }
  } else if (pre_op_desc->HasAttr(kTypeFFTSPlus) || pre_op_desc->GetType() == kAtomicAddrClean) {
    FFTS_LOGD("Non-phony node name: %s, node type: %s, increment precnt by 1.", pre_op_desc->GetName().c_str(),
              pre_op_desc->GetType().c_str());
    ++pred_cnt;
  }
}

void FFTSPlusTaskBuilder::JudgeAutoStratCtxIdListInfo(const ge::NodePtr &node,
                                                      uint32_t &pred_cnt) const {
  vector<uint32_t> at_start_ctx_id_list;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), kAutoAtStartCtxIdList, at_start_ctx_id_list);
  if (!at_start_ctx_id_list.empty()) {
    pred_cnt += 1;
  }
  FFTS_LOGD("FFTSPlusTaskBuilder node name:%s, node type:%s, predCnt:%u success.",
            node->GetName().c_str(), node->GetType().c_str(), pred_cnt);
}

Status FFTSPlusTaskBuilder::FillCommonProducersInfo(const ge::NodePtr &node,
                                                    uint32_t &pred_cnt,
                                                    FftsPlusComCtx_t &ffts_plus_context) const {
  for (const auto &up_node : node->GetInAllNodes()) {
    FillSingleProducersInfo(up_node, pred_cnt, kRecuriseCntMax);
  }
  JudgeAutoStratCtxIdListInfo(node, pred_cnt);
  ffts_plus_context.pred_cnt = pred_cnt;
  return SUCCESS;
}

Status FFTSPlusTaskBuilder::FillProducersInfoForLabelX(const ge::NodePtr &node,
                                                       FftsPlusComCtx_t &ffts_plus_context,
                                                       uint32_t &pred_cnt,
                                                       const std::string &node_type,
                                                       const ge::OpDescPtr &op_desc) const {
  if (node_type == "LabelSet") {
    FFTS_LOGD("dealing with Producers' labelset");
    PrintNode(node);
    bool lastlabelset = false;
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), "_labelsetleave", lastlabelset) && lastlabelset) {
      pred_cnt = 1;
      ffts_plus_context.pred_cnt = pred_cnt;
      return SUCCESS;
    }
    PrintNodeAttrExtNode(node, ATTR_NAME_LABELSET_PRE_LABEL);
    ge::NodePtr labelset_pre_label = nullptr;
    labelset_pre_label = op_desc->TryGetExtAttr(ATTR_NAME_LABELSET_PRE_LABEL, labelset_pre_label);
    if (labelset_pre_label != nullptr) {
      FillSingleProducersInfo(labelset_pre_label, pred_cnt, kRecuriseCntMax);
    }
    return FillCommonProducersInfo(node, pred_cnt, ffts_plus_context);
  } else if (node_type == "LabelGoto" || node_type == "LabelGotoEx") {
    FFTS_LOGD("deal with Producers labelgoto/LabelGotoEx.");
    PrintNode(node);
    return FillCommonProducersInfo(node, pred_cnt, ffts_plus_context);
  } else if (node_type == "LabelSwitchByIndex") {
    FFTS_LOGD("Dealing with Producers LabelSwitchByIndex.");
    PrintNode(node);
    return FillCommonProducersInfo(node, pred_cnt, ffts_plus_context);
  } else {
    return SUCCESS;
  }
}

Status FFTSPlusTaskBuilder::FillProducersInfo(const ge::NodePtr &node, FftsPlusComCtx_t &ffts_plus_context) const {
  auto node_type = node->GetType();
  auto op_desc = node->GetOpDesc();
  uint32_t pred_cnt = 0;
  FFTS_LOGD("Dealing with FillProducersInfo.");
  PrintNode(node);
  PrintNodeAttrExtNodes(node, ATTR_NAME_PARENT_PRE_NODES);
  std::shared_ptr<std::vector<ge::NodePtr>> parent_inputnodes  = nullptr;
  parent_inputnodes = op_desc->TryGetExtAttr(ATTR_NAME_PARENT_PRE_NODES, parent_inputnodes);
  if (parent_inputnodes != nullptr && (*parent_inputnodes).size() != 0) {
    for (const auto &up_node : (*parent_inputnodes)) {
      FillSingleProducersInfo(up_node, pred_cnt, kRecuriseCntMax);
      FFTS_LOGD("prenodedealwith has prent inputnode count = %u.", pred_cnt);
    }
  }

  parent_inputnodes = op_desc->TryGetExtAttr(kNonEdgePreNodes, parent_inputnodes);
  if (parent_inputnodes != nullptr && (*parent_inputnodes).size() != 0) {
    for (const auto &up_node : (*parent_inputnodes)) {
      FillSingleProducersInfo(up_node, pred_cnt, kRecuriseCntMax);
      FFTS_LOGD("Add non-edge pre nodes with parent inputnode count = %u.", pred_cnt);
    }
  }

  if (LABELX_NODE_TYPE.count(node_type) != 0) {
    FFTS_LOGD("deal with Producers' LabelX");
    return FillProducersInfoForLabelX(node, ffts_plus_context, pred_cnt, node_type, op_desc);
  } else {
    FFTS_LOGD("deal with other producers.");
    PrintNodeAttrExtNode(node, ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES);
    ge::NodePtr inner_graph_outputs_node = nullptr;
    inner_graph_outputs_node = op_desc->TryGetExtAttr(ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES, inner_graph_outputs_node);
    if (inner_graph_outputs_node == nullptr) {
      FFTS_LOGD("deal with other producers.s no labelx attr.");
      return FillCommonProducersInfo(node, pred_cnt, ffts_plus_context);
    } else {
      FFTS_LOGD("deal with other producers.s has labelx attr.");
      FillSingleProducersInfo(inner_graph_outputs_node, pred_cnt, kRecuriseCntMax);
      return FillCommonProducersInfo(node, pred_cnt, ffts_plus_context);
    }
  }
}

bool FFTSPlusTaskBuilder::GetJumpLabelSetContextid(const ge::NodePtr &node,
                                                   uint32_t &jumplabel_context_id,
                                                   bool &has_jumpnode) const {
  ge::NodePtr labelset = nullptr;
  ge::OpDescPtr op_desc = node->GetOpDesc();
  FFTS_LOGD("GetJumpLabelSetContextId");
  PrintNode(node);
  PrintNodeAttrExtNode(node, ATTR_NAME_LABEL_JUMP_NODE);
  jumplabel_context_id = 0;
  labelset = op_desc->TryGetExtAttr(ATTR_NAME_LABEL_JUMP_NODE, labelset);
  if (labelset != nullptr) {
      (void)ge::AttrUtils::GetInt(labelset->GetOpDesc(), kContextId, jumplabel_context_id);
      FFTS_LOGD("Out node name: %s, node type: %s, context_id: %u.", labelset->GetName().c_str(),
                labelset->GetType().c_str(), jumplabel_context_id);
      has_jumpnode = true;
      (void)ge::AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_JUMP_NODE_INDEX, jumplabel_context_id);
    return true;
  }
  has_jumpnode = false;
  return false;
}

void FFTSPlusTaskBuilder::FillManualCustomersInfoCommon(const ge::NodePtr &up_node,
                                                        ge::OpDescPtr up_op_desc,
                                                        FftsPlusComCtx_t &sub_ffts_plus_context_elem) const {
  if (up_node == nullptr) {
    return;
  }
  if (!up_node->GetOpDesc()->HasAttr(kTypeFFTSPlus)) {
    FFTS_LOGD("Node name: %s does not have the _ffts_plus attribute.", up_op_desc->GetName().c_str());
    return;
  }

  FillComCtxForNode(up_node, kRecuriseCntMax, sub_ffts_plus_context_elem);
  return;
}

void FFTSPlusTaskBuilder::FillManualCustomersInfoForLabelX(const ge::NodePtr &node,
                                                           FftsPlusComCtx_t &sub_ffts_plus_context_elem,
                                                           uint32_t &jumplabel_context_id) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  auto node_type = node->GetType();
  if (node_type == "LabelGoto" || node_type == "LabelGotoEx") {
    return FillManualCustomersInfoForLabelGoto(node, sub_ffts_plus_context_elem, jumplabel_context_id);
  } else if (node_type == "LabelSwitchByIndex") {
    return FillManualCustomersInfoForLabelSwitch(node, sub_ffts_plus_context_elem);
  } else if (node_type == "LabelSet") {
    return FillManualCustomersInfoForLabelSet(node, sub_ffts_plus_context_elem);
  }
  return;
}

void FFTSPlusTaskBuilder::FillManualCustomersInfoForLabelSet(const ge::NodePtr &node,
                                                             FftsPlusComCtx_t &sub_ffts_plus_context_elem) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  FFTS_LOGD("handle customer LabelSet");
  PrintNode(node);
  PrintNodeAttrExtNodes(node, ATTR_NAME_LASTLABELSET_OUT_NODES);
  std::shared_ptr<std::vector<ge::NodePtr>> label_outgraph_out_nodes = nullptr;
  label_outgraph_out_nodes = op_desc->TryGetExtAttr(ATTR_NAME_LASTLABELSET_OUT_NODES, label_outgraph_out_nodes);
  if (label_outgraph_out_nodes != nullptr && (*label_outgraph_out_nodes).size() != 0) {
    for (const auto &up_node : (*label_outgraph_out_nodes)) {
      FFTS_CHECK(up_node == nullptr, FFTS_LOGW("The up_node is nullptr."), return);
      ge::OpDescPtr up_op_desc = up_node->GetOpDesc();
      FFTS_LOGD("label_outgraph_out_nodes.");
      PrintNode(node);
      FillManualCustomersInfoCommon(up_node, up_op_desc, sub_ffts_plus_context_elem);
    }
  }
  for (const auto &up_node : node->GetOutAllNodes()) {
    FFTS_LOGD("Out node name: %s, node type: %s.", up_node->GetName().c_str(),
              up_node->GetType().c_str());
    ge::OpDescPtr up_op_desc = up_node->GetOpDesc();
    FillManualCustomersInfoCommon(up_node, up_op_desc, sub_ffts_plus_context_elem);
  }
  return;
}

void FFTSPlusTaskBuilder::FillManualCustomersInfoForLabelSwitch(const ge::NodePtr &node,
                                                                FftsPlusComCtx_t &sub_ffts_plus_context_elem) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  FFTS_LOGD("Dealing with customer LabelSwitchByIndex.");
  PrintNode(node);
  PrintNodeAttrExtNodes(node, ATTR_NAME_LABEL_JUMP_NODES);
  std::shared_ptr<std::vector<ge::NodePtr>> labelsets_nodes  = nullptr;
  labelsets_nodes = op_desc->TryGetExtAttr(ATTR_NAME_LABEL_JUMP_NODES, labelsets_nodes);
  if (labelsets_nodes != nullptr && (*labelsets_nodes).size() != 0) {
    for (const auto &up_node : (*labelsets_nodes)) {
      uint32_t success_list_contextid = 0;
      (void)ge::AttrUtils::GetInt(up_node->GetOpDesc(), kContextId, success_list_contextid);
      FFTS_LOGD("Out node name: %s, node type: %s, context_id: %u.", up_node->GetName().c_str(),
                up_node->GetType().c_str(), success_list_contextid);
      FFTS_LOGD("List jumplabel_context_id = %u.", success_list_contextid);
      sub_ffts_plus_context_elem.succ_list.emplace_back(success_list_contextid);
    }
  }
  FFTS_LOGD("Dealing with customer LabelSwitchByIndex. successlist size = %zu.",
            sub_ffts_plus_context_elem.succ_list.size());
  for (auto iter : sub_ffts_plus_context_elem.succ_list) {
    FFTS_LOGD("Dealing with customer LabelSwitchByIndex. contextid = %u.", iter);
  }
}

void FFTSPlusTaskBuilder::FillManualCustomersInfoForLabelGoto(const ge::NodePtr &node,
                                                              FftsPlusComCtx_t &sub_ffts_plus_context_elem,
                                                              uint32_t &jumplabel_context_id) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  bool has_jumpnode = false;
  FFTS_LOGD("Dealing with customers LabelGoto.");
  if (GetJumpLabelSetContextid(node, jumplabel_context_id, has_jumpnode)) {
    FFTS_LOGD("jumplabel_context_id = %u.", jumplabel_context_id);
    sub_ffts_plus_context_elem.succ_list.emplace_back(jumplabel_context_id);
  }
  return;
}


Status FFTSPlusTaskBuilder::FillManualCustomersInfo(const ge::NodePtr &node,
                                                    FftsPlusComCtx_t &sub_ffts_plus_context_elem) const {
  uint32_t context_id = 0;
  uint32_t jumplabel_context_id = 0;
  ge::NodePtr subgraphnode = nullptr;
  ge::OpDescPtr op_desc = node->GetOpDesc();
  PrintNode(node);
  PrintNodeAttrExtNode(node, ATTR_NAME_PARENT_OUTPUTS_OUTPUT_NODE);
  subgraphnode = op_desc->TryGetExtAttr(ATTR_NAME_PARENT_OUTPUTS_OUTPUT_NODE, subgraphnode);
  if (subgraphnode != nullptr) {
    FFTS_LOGD("subgraphnode manualinfo.");
    (void)ge::AttrUtils::GetInt(subgraphnode->GetOpDesc(), kContextId, context_id);
    sub_ffts_plus_context_elem.succ_list.emplace_back(context_id);
  }

  std::shared_ptr<std::vector<ge::NodePtr>> succ_nodes  = nullptr;
  succ_nodes = op_desc->TryGetExtAttr(kNonEdgeSuccList, succ_nodes);
  if (succ_nodes != nullptr && (*succ_nodes).size() != 0) {
    for (const auto &succ_node : (*succ_nodes)) {
      FFTS_CHECK_NOTNULL(succ_node);
      auto succ_node_desc = succ_node->GetOpDesc();
      if (!ge::AttrUtils::GetInt(succ_node_desc, kContextId, context_id)) {
        REPORT_FFTS_ERROR("[FFTSPLUS][TASKBUILDER][node %s, type %s] does not have a context_id.",
                          succ_node->GetName().c_str(), succ_node->GetType().c_str());
        return FAILED;
      }
      sub_ffts_plus_context_elem.succ_list.emplace_back(context_id);
    }
  }
  auto node_type = node->GetType();
  if (LABELX_NODE_TYPE.count(node_type) != 0) {
    FFTS_LOGD("LABELX_NODE_TYPE: manualinfo.");
    FillManualCustomersInfoForLabelX(node, sub_ffts_plus_context_elem, jumplabel_context_id);
    return SUCCESS;
  } else {
    FFTS_LOGD("other manual information.");
    for (const auto &up_node : node->GetOutAllNodes()) {
      FFTS_CHECK_NOTNULL(up_node);
      FFTS_LOGD("Out node name: %s, node type: %s.", up_node->GetName().c_str(),
                up_node->GetType().c_str());
      ge::OpDescPtr up_op_desc = up_node->GetOpDesc();
      FillManualCustomersInfoCommon(up_node, up_op_desc, sub_ffts_plus_context_elem);
    }
  }
  return SUCCESS;
}

Status FFTSPlusTaskBuilder::FillCustomersInfo(const ge::NodePtr &node, FftsPlusComCtx_t &sub_ffts_plus_context_elem,
                                              vector<FftsPlusComCtx_t> &sub_ffts_plus_context) const {
  FFTS_CHECK_NOTNULL(node);

  // Confirm whether it is automatic or manual
  bool thread_manual_mode = true;
  ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  if (slice_info_ptr && slice_info_ptr->thread_mode == kAutoMode) {
    thread_manual_mode = false;
  }
  if (thread_manual_mode) {
    // manual
    if (FillManualCustomersInfo(node, sub_ffts_plus_context_elem)) {
      return FAILED;
    }
    sub_ffts_plus_context_elem.successorNum = sub_ffts_plus_context_elem.succ_list.size();
    FFTS_LOGD("[TaskBuilder][Fftsplus][CommonTask] Node name: %s, Node type: %s, Successor Num: %u.",
              node->GetName().c_str(), node->GetType().c_str(), sub_ffts_plus_context_elem.successorNum);
    sub_ffts_plus_context.push_back(sub_ffts_plus_context_elem);
  } else {
    // auto thread
    vector<uint32_t> context_id_list;
    (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), kAutoCtxIdList, context_id_list);
    for (size_t i = 0; i < context_id_list.size(); i++) {
      sub_ffts_plus_context.emplace_back(sub_ffts_plus_context_elem);
    }
  }
  return SUCCESS;
}

Status FFTSPlusTaskBuilder::GenFftsPlusDependencyInfo(const ge::NodePtr &node,
                                                      vector<FftsPlusComCtx_t> &sub_ffts_plus_context) const {
  FFTS_CHECK_NOTNULL(node);
  FftsPlusComCtx_t sub_ffts_plus_context_elem = {0, 0, 0, {}};
  FillProducersInfo(node, sub_ffts_plus_context_elem);
  FillCustomersInfo(node, sub_ffts_plus_context_elem, sub_ffts_plus_context);
  return SUCCESS;
}

Status FFTSPlusTaskBuilder::GenFftsPlusTaskCommonInfo(const ge::NodePtr &node,
                                                      vector<FftsPlusComCtx_t> &sub_ffts_plus_context) const {
  auto status = GenFftsPlusDependencyInfo(node, sub_ffts_plus_context);
  return status;
}

void FFTSPlusTaskBuilder::FillContextData(const domi::FftsPlusAicAivCtxDef *aicore_ctx_def,
                                          domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def) const {
  if (aicore_ctx_def == nullptr || aic_aiv_ctx_def == nullptr) {
    return;
  }
  aic_aiv_ctx_def->set_aten(aicore_ctx_def->aten());

  aic_aiv_ctx_def->set_prefetch_once_bitmap(aicore_ctx_def->prefetch_once_bitmap());
  aic_aiv_ctx_def->set_prefetch_enable_bitmap(aicore_ctx_def->prefetch_enable_bitmap());
  aic_aiv_ctx_def->set_atm(aicore_ctx_def->atm());
  aic_aiv_ctx_def->set_thread_dim(aicore_ctx_def->thread_dim());
  aic_aiv_ctx_def->set_non_tail_block_dim(aicore_ctx_def->non_tail_block_dim());
  aic_aiv_ctx_def->set_tail_block_dim(aicore_ctx_def->tail_block_dim());
  aic_aiv_ctx_def->set_input_output_count(aicore_ctx_def->input_output_count());
  aic_aiv_ctx_def->set_schem(aicore_ctx_def->schem());
  for (auto task_addr : aicore_ctx_def->task_addr()) {
    aic_aiv_ctx_def->add_task_addr(task_addr);
  }
  for (auto task_addr_offset : aicore_ctx_def->task_addr_offset()) {
    aic_aiv_ctx_def->add_task_addr_offset(task_addr_offset);
  }
  for (auto kernel_name : aicore_ctx_def->kernel_name()) {
    aic_aiv_ctx_def->add_kernel_name(kernel_name);
  }
  aic_aiv_ctx_def->set_task_param_ptr_offset(aicore_ctx_def->task_param_ptr_offset());
}

void FFTSPlusTaskBuilder::FillContextData(const domi::FftsPlusMixAicAivCtxDef *aicore_ctx_def,
                                          domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def) const {
  if (aicore_ctx_def == nullptr || mix_aic_aiv_ctx_def == nullptr) {
    return;
  }
  mix_aic_aiv_ctx_def->set_aten(aicore_ctx_def->aten());

  mix_aic_aiv_ctx_def->set_prefetch_once_bitmap(aicore_ctx_def->prefetch_once_bitmap());
  mix_aic_aiv_ctx_def->set_prefetch_enable_bitmap(aicore_ctx_def->prefetch_enable_bitmap());
  mix_aic_aiv_ctx_def->set_atm(aicore_ctx_def->atm());
  mix_aic_aiv_ctx_def->set_thread_dim(aicore_ctx_def->thread_dim());
  mix_aic_aiv_ctx_def->set_non_tail_block_dim(aicore_ctx_def->non_tail_block_dim());
  mix_aic_aiv_ctx_def->set_tail_block_dim(aicore_ctx_def->tail_block_dim());
  mix_aic_aiv_ctx_def->set_input_output_count(aicore_ctx_def->input_output_count());
  mix_aic_aiv_ctx_def->set_schem(aicore_ctx_def->schem());
  for (auto task_addr : aicore_ctx_def->task_addr()) {
    mix_aic_aiv_ctx_def->add_task_addr(task_addr);
  }
  for (auto kernel_name : aicore_ctx_def->kernel_name()) {
    mix_aic_aiv_ctx_def->add_kernel_name(kernel_name);
  }

  mix_aic_aiv_ctx_def->set_non_tail_block_ratio_n(aicore_ctx_def->non_tail_block_ratio_n());
  mix_aic_aiv_ctx_def->set_tail_block_ratio_n(aicore_ctx_def->tail_block_ratio_n());
}

Status FFTSPlusTaskBuilder::FillContextData(const domi::FftsPlusAicpuCtxDef *aicpu_ctx_def_ptr,
                                            domi::FftsPlusAicpuCtxDef *aicpu_ctx_def) const {
  FFTS_CHECK_NOTNULL(aicpu_ctx_def_ptr);
  FFTS_CHECK_NOTNULL(aicpu_ctx_def);
  aicpu_ctx_def->set_atm(aicpu_ctx_def_ptr->atm());
  aicpu_ctx_def->set_kernel_type(aicpu_ctx_def_ptr->kernel_type());
  aicpu_ctx_def->set_bm(aicpu_ctx_def_ptr->bm());
  aicpu_ctx_def->set_topic_type(aicpu_ctx_def_ptr->topic_type());
  aicpu_ctx_def->set_thread_id(aicpu_ctx_def_ptr->thread_id());
  aicpu_ctx_def->set_thread_dim(aicpu_ctx_def_ptr->thread_dim());
  aicpu_ctx_def->set_non_tail_block_dim(aicpu_ctx_def_ptr->non_tail_block_dim());
  aicpu_ctx_def->set_tail_block_dim(aicpu_ctx_def_ptr->tail_block_dim());
  aicpu_ctx_def->set_sub_topic_id(aicpu_ctx_def_ptr->sub_topic_id());
  aicpu_ctx_def->set_topic_id(aicpu_ctx_def_ptr->topic_id());
  aicpu_ctx_def->set_group_id(aicpu_ctx_def_ptr->group_id());
  aicpu_ctx_def->set_task_param_offset(aicpu_ctx_def_ptr->task_param_offset());
  domi::aicpuKernelDef *kernel_def = aicpu_ctx_def->mutable_kernel();
  FFTS_CHECK_NOTNULL(kernel_def);
  kernel_def->set_args_size(aicpu_ctx_def_ptr->kernel().args_size());
  kernel_def->set_args(aicpu_ctx_def_ptr->kernel().args());
  kernel_def->set_so_name(aicpu_ctx_def_ptr->kernel().so_name());
  kernel_def->set_kernel_name(aicpu_ctx_def_ptr->kernel().kernel_name());
  kernel_def->set_kernel_ext_info(aicpu_ctx_def_ptr->kernel().kernel_ext_info());
  kernel_def->set_kernel_ext_info_size(aicpu_ctx_def_ptr->kernel().kernel_ext_info_size());
  return SUCCESS;
}
}  // namespace ffts
